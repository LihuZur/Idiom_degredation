"""Model-suggested idioms (Kai 4a, "leave it up to its own decision"), verified against MIDAS."""

import json
from collections import Counter
from pathlib import Path

from augmentation.idiom_generation import Item, Suggester, finish, load_inventory, prepare
from augmentation.idiom_retrieval import SpanProposal, parse_proposals
from data.norms.idiom_text import IdiomIndex
from data.norms.inventory import MidasIdiom
from tests._generation_helpers import X, write_inventory

MIDAS = [
    MidasIdiom("1", ("bury the hatchet",), ("To make peace.",), ()),
    MidasIdiom("50", ("keep an eye on",), ("To monitor closely.",), ()),
    MidasIdiom("60", ("pound the pavement",), ("To look for work.",), ()),
    MidasIdiom("61", ("pound the pavement",), ("To walk the streets.",), ()),
    MidasIdiom("70", ("at sea",), ("Confused.",), ()),
    MidasIdiom("71", ("be at sea",), ("On a ship.",), ()),
]


class _Retriever:
    def rank(self, meaning: str) -> list[str]:
        del meaning
        return ["2", "3", "1"]


def _setup(tmp_path: Path) -> tuple[list, IdiomIndex, Suggester]:  # type: ignore[type-arg]
    final, _ = write_inventory(tmp_path)
    inv, rows = load_inventory(final)
    return inv, IdiomIndex(rows, min_tokens=2, min_content=1), Suggester(MIDAS, inv)


def test_lookup_inventory_midas_only_duplicates_and_ambiguous(tmp_path: Path) -> None:
    _, _, s = _setup(tmp_path)
    inventory_hit = s.lookup("buried the hatchet")
    assert inventory_hit is not None and inventory_hit.norm_source == "bulkes_tanner_2017"
    midas_hit = s.lookup("keep an eye on")
    assert midas_hit is not None and midas_hit.norm_source == "midas_only"
    assert midas_hit.opacity is None
    duplicate = s.lookup("pound the pavement")
    assert duplicate is not None and duplicate.idiom_id == "60"
    assert s.lookup("at sea") is None  # two different idioms share the key
    assert s.lookup("shot in the arm") is None
    assert s.lookup("") is None


def test_parse_proposals_reads_the_suggested_idiom() -> None:
    raw = json.dumps(
        [{"span": "stopped fighting", "meaning": "made peace", "idiom": "bury the hatchet"}]
    )
    assert parse_proposals(raw, X) == [
        SpanProposal("stopped fighting", "made peace", "bury the hatchet")
    ]


def test_prepare_adds_suggestions_and_finish_records_them(tmp_path: Path) -> None:
    inv, index, s = _setup(tmp_path)
    item = Item("sst2", "a", "They kept watching the long war.", "1", {})
    props = {"a": [SpanProposal("kept watching", "monitored", "keep an eye on")]}
    (prep,) = prepare(
        [item],
        inv,
        k=2,
        cap=15,
        seed=0,
        usage=Counter(),
        proposals=props,
        retriever=_Retriever(),  # type: ignore[arg-type]
        suggester=s,
    )
    assert {i.idiom_id for i in prep.shortlist} == {"2", "3", "50"}
    assert prep.suggested == frozenset({"50"})
    reply = json.dumps(
        {
            "choice": "50",
            "span": "kept watching",
            "idiom_text": "kept an eye on",
            "literal_text": "kept monitoring",
        }
    )
    (no_presence,) = finish([prep], [reply], index, model="m", p_hash="h")
    assert no_presence.reason == "idiom_not_present"  # inventory index alone cannot see it
    (out,) = finish([prep], [reply], index, model="m", p_hash="h", presence_index=s.presence_index)
    assert out.ok and out.record is not None
    assert out.record["inventory_source"] == "midas_only"
    assert out.record["suggested_by_model"] is True
    assert out.idiomatic == "They kept an eye on the long war."


def test_suggestions_respect_the_reuse_cap(tmp_path: Path) -> None:
    inv, _, s = _setup(tmp_path)
    item = Item("sst2", "a", "They kept watching the long war.", "1", {})
    props = {"a": [SpanProposal("kept watching", "monitored", "keep an eye on")]}
    (prep,) = prepare(
        [item],
        inv,
        k=2,
        cap=1,
        seed=0,
        usage=Counter({"50": 1}),
        proposals=props,
        retriever=_Retriever(),  # type: ignore[arg-type]
        suggester=s,
    )
    assert "50" not in {i.idiom_id for i in prep.shortlist}
    assert prep.suggested == frozenset()


def test_idiom_equal_to_span_up_to_case_is_no_change(tmp_path: Path) -> None:
    _, index, s = _setup(tmp_path)
    x = "They buried the hatchet after the long war."
    reply = json.dumps(
        {
            "choice": "1",
            "span": "buried the hatchet",
            "idiom_text": "Buried the Hatchet",
            "literal_text": "made peace",
        }
    )
    from augmentation.idiom_generation import validate  # noqa: PLC0415

    assert validate(reply, x, {"1"}, index, s.presence_index).reason == "no_change"
