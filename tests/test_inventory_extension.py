"""MAGPIE extension tier, its kind/domain tags, and the generator changes that come with it."""

import csv
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from augmentation.idiom_generation import (
    Item,
    build_shortlist,
    load_inventory,
    opacity_cuts,
    render_prompt,
    validate,
)
from augmentation.idiom_retrieval import SpanProposal
from augmentation.labelstudio import stratified_gold_sample
from data.norms import extension_tags
from data.norms.idiom_text import IdiomIndex
from data.norms.inventory import MidasIdiom, magpie_extension_rows
from tests._generation_helpers import FIELDS, GOOD, X, write_inventory

MIDAS = [
    MidasIdiom("10", ("take a back seat",), ("To adopt a passive role.",), ()),
    MidasIdiom("11", ("bury the hatchet",), ("To make peace.",), ()),
    MidasIdiom("12", ("under lock and key",), ("Securely locked away.",), ()),
    MidasIdiom("13", ("at sea",), ("Confused.",), ()),
    MidasIdiom("14", ("at sea",), ("On a ship.",), ()),
    MidasIdiom("15", ("toast",), ("Doomed.",), ()),
]


def test_magpie_extension_rows_filters_and_records() -> None:
    uses = Counter(
        {
            "take a back seat": 6,
            "bury the hatchet": 9,  # already in the inventory
            "under lock and key": 2,  # below min uses
            "at sea": 7,  # two MIDAS idioms share the key
            "toast": 8,  # single word: the generator could never find it
            "no such idiom here": 5,
        }
    )
    rows, counts = magpie_extension_rows(
        MIDAS, uses, {}, exclude_ids={"11"}, exclude_names=set(), min_uses=5
    )
    assert [r["norm_idiom"] for r in rows] == ["take a back seat"]
    row = rows[0]
    assert row["norm_source"] == "magpie_bnc"
    assert row["opacity"] == "" and row["familiarity"] == ""
    assert row["magpie_idiomatic_uses"] == "6"
    assert counts == Counter(
        {
            "added": 1,
            "already_in_inventory": 1,
            "below_min_uses": 1,
            "single_word": 1,
            "magpie_type_ambiguous": 1,
            "magpie_type_unmatched": 1,
        }
    )


def test_parse_tags_keeps_only_valid_pairs() -> None:
    raw = (
        '```json\n{"1": {"kind": "figurative", "domain": "sports"},'
        ' "2": {"kind": "fixed_phrase", "domain": "everyday"},'
        ' "3": {"kind": "literal", "domain": "everyday"},'
        ' "4": {"kind": "figurative", "domain": "cooking"},'
        ' "9": {"kind": "figurative", "domain": "sports"}}\n```'
    )
    assert extension_tags.parse_tags(raw, ["1", "2", "3", "4"]) == {
        "1": ("figurative", "sports"),
        "2": ("fixed_phrase", "everyday"),
    }


def _write_extra(tmp_path: Path) -> Path:
    extra = tmp_path / "extra.csv"
    with extra.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerow(
            {
                "norm_idiom": "take a back seat",
                "norm_source": "magpie_bnc",
                "midas_ids": "10",
                "midas_forms": json.dumps(["take a back seat"]),
                "definitions": json.dumps(["To adopt a passive role."]),
                "example_sentences": "[]",
                "opacity": "",
                "familiarity": "",
                "min_word_count_enwiki": "",
            }
        )
    return extra


def test_load_inventory_with_unnormed_tier(tmp_path: Path) -> None:
    final, domains = write_inventory(tmp_path)
    extra = _write_extra(tmp_path)
    extra_domains = tmp_path / "extra_domains.csv"
    extra_domains.write_text("idiom_id,idiom,domain\n10,take a back seat,everyday\n")
    inv, rows = load_inventory(
        final, domains_csv=[domains, extra_domains, tmp_path / "missing.csv"], extra_csvs=[extra]
    )
    assert len(inv) == 4 and len(rows) == 4
    unnormed = next(i for i in inv if i.idiom_id == "10")
    assert unnormed.opacity is None and unnormed.domain == "everyday"
    assert opacity_cuts(inv) == opacity_cuts([i for i in inv if i.opacity is not None])
    short, _ = build_shortlist(inv, 4, random.Random(0), Counter(), 15, opacity_cuts(inv))
    assert {i.idiom_id for i in short} == {"1", "2", "3", "10"}


def test_v4_prompt_shows_candidate_spans_and_v2_ignores_them(tmp_path: Path) -> None:
    final, _ = write_inventory(tmp_path)
    inv, _ = load_inventory(final)
    item = Item("sst2", "a", X, "1", {})
    props = [SpanProposal("stopped fighting", "ended the conflict")]
    v4 = render_prompt(item, inv, template="idiom_span_v4.txt", proposals=props)
    assert '- "stopped fighting": ended the conflict' in v4
    v2 = "idiom_span_v2.txt"
    assert render_prompt(item, inv, template=v2, proposals=props) == render_prompt(
        item, inv, template=v2
    )


def test_self_rejected_verdict(tmp_path: Path) -> None:
    final, _ = write_inventory(tmp_path)
    _, rows = load_inventory(final)
    index = IdiomIndex(rows, min_tokens=2, min_content=1)
    ids = {"1", "2", "3"}
    assert validate(GOOD, X, ids, index).ok
    keep = json.dumps({**json.loads(GOOD), "verdict": "KEEP"})
    assert validate(keep, X, ids, index).ok
    drop = json.dumps({**json.loads(GOOD), "verdict": "NONE"})
    assert validate(drop, X, ids, index).reason == "self_rejected"


def test_gold_sample_puts_unnormed_idioms_in_their_own_stratum() -> None:
    items = [
        {
            "id": str(n),
            "label": "1",
            "original": "word " * (5 + n % 3),
            "meta": {"idiom": {"opacity": None if n % 2 else 0.1 * (n % 10)}},
        }
        for n in range(40)
    ]
    sample = stratified_gold_sample(items, 10, seed=0)
    assert len(sample) == 10
    assert any(i["meta"]["idiom"]["opacity"] is None for i in sample)
    assert any(i["meta"]["idiom"]["opacity"] is not None for i in sample)


def test_literal_must_reword_the_span(tmp_path: Path) -> None:
    final, _ = write_inventory(tmp_path)
    _, rows = load_inventory(final)
    index = IdiomIndex(rows, min_tokens=2, min_content=1)
    same = json.dumps({**json.loads(GOOD), "literal_text": "stopped fighting"})
    assert validate(same, X, {"1", "2", "3"}, index).reason == "no_change"


def test_v8_lists_every_sense_and_v5_only_the_first(tmp_path: Path) -> None:
    final, _ = write_inventory(tmp_path)
    inv, _ = load_inventory(final)
    two = replace(inv[0], definitions=("To begin traveling.", "To leave a place."))
    item = Item("sst2", "a", X, "1", {})
    v8 = render_prompt(item, [two], template="idiom_span_v8.txt")
    v5 = render_prompt(item, [two], template="idiom_span_v5.txt")
    assert "To begin traveling / To leave a place" in v8
    assert "To leave a place" not in v5
