import json
from collections import Counter
from pathlib import Path

from click.testing import CliRunner

from augmentation.selection import Pair, allocate, length_bin, select_pairs
from scripts.checker import main, position_bins


def test_allocate_largest_remainder():
    assert sum(allocate(10, {"a": 1, "b": 1, "c": 1}).values()) == 10
    assert allocate(7, {"x": 0.5, "y": 0.25, "z": 0.25}) == {
        "x": 3,
        "y": 2,
        "z": 2,
    }  # 3.5 / 1.75 / 1.75: the larger remainders win
    assert allocate(0, {"a": 1}) == {"a": 0}


def test_length_bin():
    assert [length_bin(w, (10, 20)) for w in (5, 10, 19, 20, 30)] == [
        "short",
        "medium",
        "medium",
        "long",
        "long",
    ]


def _pairs() -> list[Pair]:
    out: list[Pair] = []
    n = 0
    for lab, count in (("0", 40), ("1", 30), ("2", 60)):
        for k in range(count):
            n += 1
            out.append(
                Pair(
                    id=f"i{n:04d}",
                    label=lab,
                    stratum=("short", "long")[k % 2],
                    idiom_id=f"idiom{n % 25}",
                    words_kept=0.5 if k % 10 == 0 else 0.95,
                )
            )
    return out


SHARES = {lab: {"short": 0.5, "long": 0.5} for lab in "012"}


def test_select_applies_threshold_cap_and_equal_labels():
    sel = select_pairs(_pairs(), 60, SHARES, min_kept=0.8, cap=5, seed=1)
    pairs = {p.id: p for p in _pairs()}
    chosen = [pairs[i] for i in sel.ids]
    assert sel.complete and len(chosen) == 60
    assert Counter(p.label for p in chosen) == {"0": 20, "1": 20, "2": 20}
    assert all(p.words_kept >= 0.8 for p in chosen)
    assert max(Counter(p.idiom_id for p in chosen).values()) <= 5
    assert sel.report["dropped_words_kept"] == 13  # every 10th of 40 / 30 / 60
    for lab in "012":
        strata = Counter(p.stratum for p in chosen if p.label == lab)
        assert abs(strata["short"] - strata["long"]) <= 1


def test_select_reports_a_short_label_and_is_deterministic():
    sel = select_pairs(_pairs(), 90, SHARES, cap=100, seed=1)
    assert not sel.complete
    assert sel.report["per_label"]["1"]["short_by"] == 3  # 27 usable (3 below 80% kept) vs quota 30
    again = select_pairs(_pairs(), 90, SHARES, cap=100, seed=1)
    assert sel.ids == again.ids


def test_short_stratum_is_filled_from_the_same_label():
    pairs = [Pair(f"a{k}", "0", "short", f"i{k}", 1.0) for k in range(2)]
    pairs += [Pair(f"b{k}", "0", "long", f"j{k}", 1.0) for k in range(10)]
    sel = select_pairs(pairs, 8, {"0": {"short": 0.5, "long": 0.5}}, seed=0)
    assert sel.complete and len(sel.ids) == 8
    assert sum(i.startswith("a") for i in sel.ids) == 2


def test_combine_passes_only_when_checker_and_review_both_pass(tmp_path: Path) -> None:
    base = {"ok": True, "reasons": {}, "naturalness": "3", "error": ""}
    steps_ok = dict.fromkeys(("step1_label", "step2_idiom", "step3_literal", "step4_rest"), "pass")
    rows = [
        {"key": "mnli:a", "steps": steps_ok, "action": "valid", "primary_reason": "none"},
        {"key": "mnli:b", "steps": steps_ok, "action": "valid", "primary_reason": "none"},
        {
            "key": "mnli:c",
            "steps": {**steps_ok, "step1_label": "fail"},
            "action": "invalid",
            "primary_reason": "label_changed",
        },
    ]
    v = tmp_path / "verdicts.jsonl"
    v.write_text("".join(json.dumps(base | r) + "\n" for r in rows))
    rev = tmp_path / "review.csv"
    rev.write_text("key,action\nmnli:a,valid\nmnli:b,fix\nmnli:c,valid\n")
    out = tmp_path / "combined.jsonl"
    res = CliRunner().invoke(
        main, ["combine", "--verdicts", str(v), "--review", str(rev), "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    got = {
        json.loads(line)["key"]: json.loads(line)["action"] for line in out.read_text().splitlines()
    }
    assert got == {"mnli:a": "valid", "mnli:b": "fix", "mnli:c": "invalid"}


def test_position_bins_expect_block_sizes() -> None:
    obs, exp = position_bins([(0, 30), (29, 30), (14, 30)])
    assert obs == [1, 0, 1, 0, 1]
    assert [round(e, 6) for e in exp] == [0.6] * 5
    obs, exp = position_bins([(0, 8)] * 8, n_bins=4)  # 8 positions in 4 blocks of 2
    assert obs == [8, 0, 0, 0] and [round(e, 6) for e in exp] == [2.0] * 4
