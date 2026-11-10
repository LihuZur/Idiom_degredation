"""Meaning-based shortlisting: proposals, retrieval ranking, and the end-to-end retrieval path."""

import json
import re
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from augmentation.idiom_generation import Item, generate, validate
from augmentation.idiom_retrieval import IdiomRetriever, Matrix, parse_proposals
from tests._generation_helpers import GOOD, X, setup_inventory

_DIM = 128


def fake_embed(texts: Sequence[str], query: bool) -> Matrix:
    """Hashed bag of words, L2-normalised: enough to rank by shared words."""
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            out[row, zlib.crc32(word.encode()) % _DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def test_parse_proposals_keeps_only_unique_spans_with_meaning() -> None:
    raw = json.dumps(
        [
            {"span": "stopped fighting", "meaning": "ended the conflict"},
            {"span": "a", "meaning": "letter"},
            {"span": "after", "meaning": ""},
            {"span": "not in text", "meaning": "x"},
        ]
    )
    assert [p.span for p in parse_proposals(f"```json\n{raw}\n```", X)] == ["stopped fighting"]
    assert parse_proposals("not json", X) == []
    assert parse_proposals(None, X) == []


def test_retriever_ranks_by_definition_and_caches_document_embeddings(tmp_path: Path) -> None:
    docs = ["bury the hatchet: To make peace", "spill the beans: To reveal a secret"]
    calls: list[bool] = []

    def counting_embed(texts: Sequence[str], query: bool) -> Matrix:
        calls.append(query)
        return fake_embed(texts, query)

    first = IdiomRetriever(["1", "2"], docs, counting_embed, tmp_path)
    assert first.rank("to reveal a secret")[0] == "2"
    assert first.rank("make peace with enemies")[0] == "1"
    again = IdiomRetriever(["1", "2"], docs, counting_embed, tmp_path)
    assert again.rank("reveal a secret")[0] == "2"
    assert calls.count(False) == 1


def test_retrieval_shortlist_feeds_the_pick_step(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    retriever = IdiomRetriever(
        [i.idiom_id for i in inv],
        [f"{i.name}: {' / '.join(i.definitions)}" for i in inv],
        fake_embed,
    )
    pick_prompts: list[str] = []

    def complete(prompts: Any) -> list[str | None]:
        out: list[str | None] = []
        for p in prompts:
            if p.startswith("Below is a short text"):
                out.append(json.dumps([{"span": "stopped fighting", "meaning": "make peace"}]))
            else:
                pick_prompts.append(p)
                out.append(GOOD)
        return out

    items = [Item("sst2", "a", X, "1", {})]
    outcomes = generate(
        items, inv, index, complete, model="m", p_hash="ph", k=1, retriever=retriever
    )
    assert [o.reason for o in outcomes] == ["ok"]
    assert outcomes[0].record is not None
    assert outcomes[0].record["shortlist_mode"] == "retrieval"
    assert "1: bury the hatchet" in pick_prompts[0]
    assert len(re.findall(r"^\d+: ", pick_prompts[0], re.M)) == 1


def test_items_without_usable_proposals_fall_back_to_a_random_shortlist(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    retriever = IdiomRetriever([i.idiom_id for i in inv], [i.name for i in inv], fake_embed)

    def complete(prompts: Any) -> list[str | None]:
        return [None if p.startswith("Below is a short text") else GOOD for p in prompts]

    out = generate(
        [Item("sst2", "a", X, "1", {})],
        inv,
        index,
        complete,
        model="m",
        p_hash="ph",
        k=3,
        retriever=retriever,
        skip_empty=False,
    )
    assert out[0].record is not None
    assert out[0].record["shortlist_mode"] == "random"


def test_items_without_usable_proposals_are_skipped_by_default(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    retriever = IdiomRetriever([i.idiom_id for i in inv], [i.name for i in inv], fake_embed)
    pick_prompts: list[str] = []

    def complete(prompts: Any) -> list[str | None]:
        pick_prompts.extend(p for p in prompts if not p.startswith("Below is a short text"))
        return [None if p.startswith("Below is a short text") else GOOD for p in prompts]

    out = generate(
        [Item("sst2", "a", X, "1", {})],
        inv,
        index,
        complete,
        model="m",
        p_hash="ph",
        k=3,
        retriever=retriever,
    )
    assert [o.reason for o in out] == ["no_candidate"]
    assert pick_prompts == []  # no call 2 for an item without candidates


def test_span_that_already_contains_an_inventory_idiom_is_rejected(tmp_path: Path) -> None:
    inv, index = setup_inventory(tmp_path)
    x = "They buried the hatchet after the long war."
    raw = json.dumps(
        {
            "choice": "2",
            "span": "buried the hatchet",
            "idiom_text": "spilled the beans",
            "literal_text": "made peace",
        }
    )
    ids = {i.idiom_id for i in inv}
    assert validate(raw, x, ids, index).reason == "span_has_idiom"
