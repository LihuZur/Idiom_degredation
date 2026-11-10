"""v3 validation tooling: Label Studio tasks / form / import, the AI checker and the repair loop (all offline)."""

import csv
import json
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from augmentation.checker import (
    TEMPLATES,
    compare,
    disagreements,
    parse_verdict,
    render_checker_prompt,
    split_dev_test,
)
from augmentation.labelstudio import (
    V3_FIELDS,
    flatten_v3_export,
    highlight_v3,
    load_v3,
    nominal_kappa,
    v3_agreement,
    v3_config_xml,
    v3_tasks,
)
from augmentation.repair import (
    RepairTask,
    item_from_v3,
    render_repair_prompt,
    repair_reason,
    repair_round,
)
from tests._generation_helpers import setup_inventory

COLS = ["id", "variant", "x", "y", "meta", "augmenter_model", "prompt_hash", "validators"]
ORIG = "The team won the final on Sunday."
IDI = "The team won the final with flying colours on Sunday."
LIT = "The team won the final very successfully on Sunday."


def _write_run(tmp: Path, dataset: str = "mmlu") -> Path:
    d = tmp / "run" / dataset
    d.mkdir(parents=True)
    meta = {"choices": ["a", "b", "c", "d"], "answer_index": 2} if dataset == "mmlu" else {}
    rec = {
        "idiom": "with flying colours",
        "midas_ids": "1|9",
        "definition": "Very successfully.",
        "idiom_text": "with flying colours",
        "literal_text": "very successfully",
        "idiomatic_span": [IDI.index("with"), IDI.index("with") + len("with flying colours")],
        "literal_span": [LIT.index("very"), LIT.index("very") + len("very successfully")],
        "self_check": "SECRET GENERATOR VERDICT",
    }
    for name, x in (("original", ORIG), ("idiomatic", IDI), ("span_literal", LIT)):
        m = dict(meta) if name == "original" else meta | {"idiom": rec}
        with (d / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerow(
                {
                    "id": "q1",
                    "variant": name,
                    "x": x,
                    "y": "2",
                    "meta": json.dumps(m),
                    "augmenter_model": "",
                    "prompt_hash": "",
                    "validators": "",
                }
            )
    return d


def test_load_v3_drops_the_generator_verdict_and_keeps_context(tmp_path: Path) -> None:
    items = load_v3(_write_run(tmp_path), "mmlu")
    assert len(items) == 1
    i = items[0]
    assert i["key"] == "mmlu:q1" and i["label"] == "C) c" and i["idiom_id"] == "1"
    assert "SECRET" not in json.dumps(i)
    tasks = v3_tasks(items, seed=0)
    assert "SECRET" not in json.dumps(tasks)
    assert "Answer choices" in tasks[0]["data"]["context_html"]


def test_highlight_marks_idiom_inserted_and_dropped_words() -> None:
    item = {
        "original": "The team won the final on Sunday.",
        "idiomatic": "The team won the final with flying colours.",
        "literal": "The team won the final very successfully.",
        "idiomatic_span": [23, 42],
        "literal_span": [23, 40],
    }
    h = highlight_v3(item)
    assert "<mark>with</mark> <mark>flying</mark> <mark>colours.</mark>" in h["idiomatic_html"]
    assert "<s>Sunday.</s>" in h["original_html"]  # the dropped detail is visible to the annotator
    assert "<mark>very</mark>" in h["literal_html"]


def test_config_is_valid_xml_and_names_every_field() -> None:
    root = ET.fromstring(v3_config_xml())
    names = {e.get("name") for e in root.iter()}
    assert set(V3_FIELDS) <= names
    assert {"context", "original", "idiomatic", "literal"} <= names


def _export(annotator: int, action: str, nat: str = "3") -> list[dict[str, Any]]:
    result = [
        {"from_name": s, "value": {"choices": ["pass"]}}
        for s in ("step1_label", "step2_idiom", "step3_literal", "step4_rest")
    ]
    result += [
        {"from_name": "naturalness", "value": {"choices": [nat]}},
        {"from_name": "action", "value": {"choices": [action]}},
        {"from_name": "note", "value": {"text": ["ok"]}},
    ]
    return [
        {
            "data": {"key": "mmlu:q1", "dataset": "mmlu", "id": "q1"},
            "annotations": [{"completed_by": annotator, "result": result}],
        }
    ]


def test_flatten_and_agreement() -> None:
    rows = flatten_v3_export(_export(1, "valid")) + flatten_v3_export(_export(2, "valid", "2"))
    assert rows[0]["action"] == "valid" and rows[0]["note"] == "ok"
    agree = {r["field"]: r for r in v3_agreement(rows)}
    assert agree["action"]["agreement"] == 1.0
    assert agree["naturalness"]["agreement"] == 0.0


def test_nominal_kappa() -> None:
    assert nominal_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(nominal_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9


def test_checker_prompt_shows_context_and_label(tmp_path: Path) -> None:
    item = load_v3(_write_run(tmp_path), "mmlu")[0]
    for t in TEMPLATES:
        p = render_checker_prompt(item, t)
        assert "<answer_choices_never_changed>A) a B) b C) c D) d" in p
        assert "<correct_label>C) c</correct_label>" in p
        assert "{" not in p.split("Answer with one JSON object")[0]  # no unfilled slot


GOOD = {
    s: {"verdict": "pass", "reason": "fine"}
    for s in ("step1_label", "step2_idiom", "step3_literal", "step4_rest")
}


def test_parse_verdict() -> None:
    v = parse_verdict(
        json.dumps(GOOD | {"naturalness": "2", "action": "valid", "primary_reason": "none"})
    )
    assert v.ok and not v.rejects
    bad = GOOD | {"step1_label": {"verdict": "fail", "reason": "hint"}}
    v = parse_verdict(
        json.dumps(bad | {"naturalness": "n/a", "action": "fix", "primary_reason": "answer_hint"})
    )
    assert v.ok and v.rejects
    assert parse_verdict("nope").rejects  # unparsable counts as a rejection
    assert parse_verdict(None).error == "no_reply"


def test_split_spreads_bad_items_and_compare_counts() -> None:
    human = {
        f"sst2:{n}": {"dataset": "sst2", "action": "fix" if n < 4 else "valid"} for n in range(20)
    }
    split = split_dev_test(human, seed=0)
    assert sum(split[f"sst2:{n}"] == "dev" for n in range(4)) == 2  # bad items in both parts
    ok = parse_verdict(
        json.dumps(GOOD | {"naturalness": "3", "action": "valid", "primary_reason": "none"})
    )
    verdicts = dict.fromkeys(human, ok)  # a checker that passes everything
    m = compare(verdicts, human, list(human))
    assert m["bad_passed"] == 4 and m["recall_bad"] == 0.0 and m["good_rejected"] == 0
    assert disagreements(verdicts, human, list(human))[0]["kind"] == "bad_passed"


def test_repair_prompt_and_round(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    v3 = {
        "dataset": "sst2",
        "id": "a",
        "original": "They stopped fighting after the long war.",
        "y": "1",
        "original_meta": {},
    }
    item = item_from_v3(v3)
    bad = parse_verdict(
        json.dumps(
            GOOD
            | {
                "step2_idiom": {"verdict": "fail", "reason": "wrong sense"},
                "naturalness": "n/a",
                "action": "fix",
                "primary_reason": "wrong_sense",
            }
        )
    )
    reason = repair_reason(bad)
    assert "wrong sense" in reason
    p = render_repair_prompt(
        item, inv, previous_idiomatic="X1", previous_literal="X2", reason=reason
    )
    assert "<earlier_idiomatic>X1</earlier_idiomatic>" in p and "REPAIR" in p
    reply = json.dumps(
        {
            "choice": "1",
            "idiomatic": "They buried the hatchet after the long war.",
            "idiom_text": "buried the hatchet",
            "literal_text": "made peace",
            "literal": "They made peace after the long war.",
            "check": "ok",
            "verdict": "KEEP",
        }
    )
    task = RepairTask("sst2:a", item, "X1", "X2", "1", reason)
    out = repair_round([task], inv, index, None, lambda ps: [reply] * len(ps), seed=0, round_=1)
    assert out[0].inserted.ok and out[0].idiom is not None and out[0].idiom.idiom_id == "1"


def test_run_repair_passes_code_check_failure_on_and_accepts_later(tmp_path: Path) -> None:
    from augmentation.repair import run_repair  # noqa: PLC0415

    inv, index = setup_inventory(tmp_path)
    x = "They stopped fighting after the long war."
    base = {
        "sst2:a": {
            "key": "sst2:a",
            "dataset": "sst2",
            "id": "a",
            "original": x,
            "y": "1",
            "original_meta": {},
            "idiomatic": "X1",
            "literal": "X2",
            "idiom_id": "1",
        }
    }
    item = item_from_v3(base["sst2:a"])
    good = json.dumps(
        {
            "choice": "1",
            "idiomatic": "They buried the hatchet after the long war.",
            "idiom_text": "buried the hatchet",
            "literal_text": "made peace",
            "literal": "They made peace after the long war.",
            "check": "ok",
            "verdict": "KEEP",
        }
    )
    seen: list[str] = []

    def gen_for(_d: str):
        def complete(ps: Sequence[str]) -> list[str | None]:
            seen.extend(ps)
            return ["not json" if len(seen) == 1 else good for _ in ps]

        return complete

    accept = parse_verdict(
        json.dumps(GOOD | {"naturalness": "3", "action": "valid", "primary_reason": "none"})
    )
    out = run_repair(
        [RepairTask("sst2:a", item, "X1", "X2", "1", "step2 idiom: wrong sense")],
        base,
        inv,
        index,
        None,
        gen_for,
        lambda items: {i["key"]: accept for i in items},
        seed=0,
    )
    assert [r["code_check"] for r in out.log] == ["bad_json", "ok"]
    assert "failed an automatic check (bad_json)" in seen[1]
    assert len(out.repaired) == 1 and out.repaired[0]["round"] == 2 and not out.unresolved
