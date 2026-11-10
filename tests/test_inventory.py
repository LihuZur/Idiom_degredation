import json
from pathlib import Path

import pytest

from data.norms.inventory import (
    MidasIdiom,
    NormRow,
    apply_review,
    build_rows,
    content_key,
    dedupe_shared_ids,
    load_midas,
    match_norms,
    surface_form,
)


def _norm(idiom: str) -> NormRow:
    return NormRow(idiom, "bulkes_tanner_2017", 3.0, 4.0, 0.4, 1.5)


def _midas(mid: str, *forms: str) -> MidasIdiom:
    return MidasIdiom(mid, forms, (f"meaning {mid}",), ("a sentence",))


def test_content_key_aligns_inflection_and_slots() -> None:
    assert content_key("They buried the hatchet") == content_key("bury the hatchet")
    assert content_key("Take someone's breath away") == content_key("take one's breath away")
    assert content_key("He took the veil") == content_key("take the veil")


def test_surface_form_ignores_case_and_punctuation() -> None:
    assert surface_form("Break the Ice!") == "break the ice"


def test_match_status_tiers() -> None:
    midas = [
        _midas("1", "break the ice"),
        _midas("2", "bury the hatchet"),
        _midas("3", "up one's sleeve"),
        _midas("4", "cook someone's goose"),
        _midas("7", "someone's goose is cooked"),
        _midas("5", "pass the buck"),
        _midas("6", "buck-passing"),
    ]
    norms = [
        _norm("Break the ice"),
        _norm("They buried the hatchet"),
        _norm("Have something up one's sleeve"),
        _norm("She cooked his goose"),
        _norm("He passed the buck"),
        _norm("Spill the beans"),
    ]
    got = {m.norm.idiom: m for m in match_norms(norms, midas)}
    assert got["Break the ice"].status == "exact"
    assert got["They buried the hatchet"].status == "key"
    assert got["Have something up one's sleeve"].status == "near"
    assert got["She cooked his goose"].status == "ambiguous"
    assert got["He passed the buck"].status == "ambiguous"
    assert "Spill the beans" not in got


def test_load_midas_merges_senses(tmp_path: Path) -> None:
    raw = [
        {"ID": "7-1", "Idiom": ["a b"], "Meaning": "first", "Sentence": ["s1"]},
        {"ID": "7-2", "Idiom": ["a b", "a c"], "Meaning": "second", "Sentence": ["s2"]},
        {"ID": "8", "Idiom": ["x y"], "Meaning": "m", "Sentence": []},
    ]
    path = tmp_path / "EN.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    by_id = {m.midas_id: m for m in load_midas(path)}
    assert by_id["7"].forms == ("a b", "a c")
    assert by_id["7"].meanings == ("first", "second")
    assert by_id["7"].sentences == ("s1", "s2")
    assert by_id["8"].sentences == ()


def test_build_rows_columns_and_frequency() -> None:
    midas = [_midas("1", "break the ice")]
    matches = match_norms([_norm("Break the ice")], midas)
    (row,) = build_rows(matches, midas, {"break": 100, "the": 5000, "ice": 40})
    assert row["match_status"] == "exact"
    assert row["midas_primary_form"] == "break the ice"
    assert row["min_word_count_enwiki"] == "40"
    assert row["match_checked"] == "accept (auto)"
    assert json.loads(row["definitions"]) == ["meaning 1"]


def _review_rows(
    checked: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, str]], list[MidasIdiom]]:
    midas = [
        _midas("4", "cook someone's goose"),
        _midas("7", "someone's goose is cooked"),
        _midas("1", "break the ice"),
        _midas("2", "bury the hatchet"),
        _midas("3", "up one's sleeve"),
    ]
    norms = [
        _norm("She cooked his goose"),
        _norm("Break the ice"),
        _norm("They buried the hatchet"),
        _norm("Have something up one's sleeve"),
    ]
    rows = build_rows(match_norms(norms, midas), midas, {})
    for r in rows:
        if r["norm_idiom"] in checked:
            r["match_checked"], r["match_note"] = checked[r["norm_idiom"]]
    return rows, midas


def test_apply_review_accept_fix_reject_and_unreviewed() -> None:
    rows, midas = _review_rows(
        {
            "She cooked his goose": ("fix", "4"),
            "They buried the hatchet": ("reject", ""),
        }
    )
    final, counts, unreviewed = apply_review(rows, midas, {})
    by_idiom = {r["norm_idiom"]: r for r in final}
    assert by_idiom["She cooked his goose"]["midas_ids"] == "4"
    assert by_idiom["Break the ice"]["match_checked"] == "accept"
    assert "They buried the hatchet" not in by_idiom
    assert unreviewed == ["Have something up one's sleeve"]
    assert counts[("ambiguous", "fix")] == 1
    assert counts[("exact", "accept")] == 1
    assert counts[("key", "reject")] == 1


def test_apply_review_accept_on_ambiguous_keeps_all_candidates() -> None:
    rows, midas = _review_rows({"She cooked his goose": ("accept", "")})
    final, _, _ = apply_review(rows, midas, {})
    row = next(r for r in final if r["norm_idiom"] == "She cooked his goose")
    assert row["midas_ids"] == "4|7"
    assert json.loads(row["definitions"]) == ["meaning 4", "meaning 7"]


def test_apply_review_rejects_bad_fix_and_unknown_value() -> None:
    rows, midas = _review_rows({"She cooked his goose": ("fix", "999")})
    with pytest.raises(ValueError, match="valid MIDAS ids"):
        apply_review(rows, midas, {})
    rows, midas = _review_rows({"She cooked his goose": ("maybe", "")})
    with pytest.raises(ValueError, match="unknown match_checked"):
        apply_review(rows, midas, {})


def test_apply_review_reads_fix_ids_typed_in_match_checked() -> None:
    rows, midas = _review_rows({"She cooked his goose": ("fix with 4", "2 candidate MIDAS ids")})
    final, _, _ = apply_review(rows, midas, {})
    row = next(r for r in final if r["norm_idiom"] == "She cooked his goose")
    assert row["midas_ids"] == "4"


def test_dedupe_shared_ids_prefers_bulkes_tanner_and_records_the_rest() -> None:
    midas = [_midas("1", "break the ice"), _midas("2", "bury the hatchet")]
    lt = NormRow("He broke the ice", "libben_titone_2008", None, None, 0.2, None)
    bt = _norm("Break the ice")
    other = _norm("They buried the hatchet")
    rows = build_rows(match_norms([lt, bt, other], midas), midas, {})
    kept, merged = dedupe_shared_ids(rows)
    assert merged == 1
    assert {r["norm_idiom"] for r in kept} == {"Break the ice", "They buried the hatchet"}
    ice = next(r for r in kept if r["norm_idiom"] == "Break the ice")
    assert json.loads(ice["also_normed_as"])[0]["norm_idiom"] == "He broke the ice"
