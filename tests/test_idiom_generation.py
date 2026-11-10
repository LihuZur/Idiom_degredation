import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from augmentation.idiom_generation import (
    Item,
    Parsed,
    build_shortlist,
    build_variants,
    generate,
    opacity_cuts,
    outside_span_identical,
    prompt_hash,
    render_prompt,
    stratified_sample,
    validate,
    write_outputs,
)
from tests._generation_helpers import GOOD, X, setup_inventory


def test_validate_accepts_good_and_names_each_failure(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    ids = {i.idiom_id for i in inv}

    def bad(**over: str) -> str:
        return json.loads(GOOD) | over

    assert validate(GOOD, X, ids, index).ok
    cases = {
        "bad_json": "not json",
        "none": json.dumps(bad(choice="NONE")),
        "unknown_choice": json.dumps(bad(choice="99")),
        "empty_field": json.dumps(bad(literal_text=" ")),
        "span_not_unique": json.dumps(bad(span="not in text")),
        "no_change": json.dumps(bad(literal_text="buried the hatchet")),
        "idiom_not_present": json.dumps(bad(idiom_text="made peace", literal_text="ended it")),
        "idiom_in_literal": json.dumps(bad(literal_text="bury the hatchet")),
        "context_repeated": json.dumps(bad(idiom_text="buried the hatchet after the long war")),
        "replacement_too_long": json.dumps(bad(literal_text=" ".join(["peace"] * 17))),
    }
    for reason, raw in cases.items():
        assert validate(raw, X, ids, index).reason == reason, reason
    fenced = "```json\n" + GOOD + "\n```"
    assert validate(fenced, X, ids, index).ok


def test_variants_differ_only_inside_the_span() -> None:
    p = Parsed(True, "ok", "1", "stopped fighting", "buried the hatchet", "made peace")
    idiomatic, literal, start = build_variants(X, p)
    assert idiomatic == "They buried the hatchet after the long war."
    assert literal == "They made peace after the long war."
    assert outside_span_identical(X, idiomatic, start, len(p.span), len(p.idiom_text))
    assert not outside_span_identical(
        X, idiomatic.replace("long", "short"), start, len(p.span), len(p.idiom_text)
    )


def test_shortlist_respects_cap_and_domain_condition(tmp_path: Path) -> None:
    inv, _ = setup_inventory(tmp_path)
    cuts = opacity_cuts(inv)
    rng = random.Random(0)
    capped, _ = build_shortlist(inv, 3, rng, Counter({"1": 15}), 15, cuts)
    assert "1" not in {i.idiom_id for i in capped}
    cross, used = build_shortlist(
        inv, 1, rng, Counter(), 15, cuts, domain_condition="cross", domain="sports"
    )
    assert used == "cross"
    assert cross[0].domain in {"war-politics"}
    _, fallback = build_shortlist(
        inv, 3, rng, Counter(), 15, cuts, domain_condition="cross", domain="sports"
    )
    assert fallback == "cross_fallback"


def test_failed_item_is_retried_with_fresh_shortlist(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    items = [Item("sst2", "a", X, "1", {})]
    shown: list[str] = []
    texts = {"1": "buried the hatchet", "2": "spilled the beans", "3": "broke the ice"}

    def complete(prompts: Any) -> list[str | None]:
        shown.extend(prompts)
        if len(shown) < 3:
            return [json.dumps({"choice": "NONE"})]
        pick = re.findall(r"^(\d+): ", prompts[0], re.M)[0]
        reply = {
            "choice": pick,
            "span": "stopped fighting",
            "idiom_text": texts[pick],
            "literal_text": "ended the conflict",
        }
        return [json.dumps(reply)]

    out = generate(
        items, inv, index, complete, model="m", p_hash="ph", k=1, max_attempts=3, wave=10
    )
    assert [(o.ok, o.attempts, o.history) for o in out] == [(True, 3, ("none", "none"))]
    ids_per_try = [set(re.findall(r"^(\d+): ", p, re.M)) for p in shown]
    assert len(shown) == 3
    assert all(not (ids_per_try[a] & ids_per_try[b]) for a in range(3) for b in range(a))


def test_prompt_contains_idioms_and_context(tmp_path: Path) -> None:
    inv, _ = setup_inventory(tmp_path)
    item = Item("mnli", "a", X, "0", {"hypothesis": "Fighting ended.", "label_name": "entailment"})
    prompt = render_prompt(item, inv)
    assert "1: bury the hatchet - To make peace" in prompt
    assert "Fighting ended." in prompt
    assert "entailment" in prompt
    assert len(prompt_hash("mnli")) == 16


def test_generate_end_to_end_and_outputs(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    items = [Item("sst2", f"id{n}", X, "1", {}) for n in range(3)]
    replies = [GOOD, json.dumps({"choice": "NONE"}), None]

    def complete(prompts: Any) -> list[str | None]:
        return [replies[n % 3] for n, _ in enumerate(prompts)]

    outcomes = generate(items, inv, index, complete, model="fake/m", p_hash="ph", k=3, wave=10)
    assert [o.reason for o in outcomes] == ["ok", "none", "empty_response"]
    rec = outcomes[0].record
    assert rec is not None
    assert (rec["idiom"], rec["definition"], rec["definition_source"]) == (
        "Bury the hatchet",
        "To make peace.",
        "midas",
    )
    assert rec["idiomatic_span"] == [
        rec["span_start"],
        rec["span_start"] + len("buried the hatchet"),
    ]
    pool = {
        i.id: {
            "id": i.id,
            "variant": "original",
            "x": i.x,
            "y": i.y,
            "meta": "{}",
            "augmenter_model": "",
            "prompt_hash": "",
            "validators": "",
        }
        for i in items
    }
    counts = write_outputs(tmp_path / "out", outcomes, pool, model="fake/m", p_hash="ph", meta={})
    assert counts == {"ok": 1, "none": 1, "empty_response": 1}
    for variant in ("original", "idiomatic", "span_literal"):
        with (tmp_path / "out" / f"{variant}.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["id"] for r in rows] == ["id0"]
    with (tmp_path / "out" / "idiomatic.csv").open(encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["x"] == "They buried the hatchet after the long war."
    assert json.loads(row["meta"])["idiom"]["span"] == "stopped fighting"


def test_stratified_sample_is_deterministic_and_balanced() -> None:
    items = [Item("sst2", str(n), "word " * (5 + n % 9), str(n % 2), {}) for n in range(120)]
    a = stratified_sample(items, 30, 0)
    assert [i.id for i in a] == [i.id for i in stratified_sample(items, 30, 0)]
    assert len(a) == 30
    assert abs(sum(i.y == "1" for i in a) - 15) <= 3
