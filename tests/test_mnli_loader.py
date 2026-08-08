"""MNLI loader tests — stubbed HF splits, no network (MNLI_DATASET_PLAN §5.7)."""

from typing import Any

import pytest

import data.mnli as mnli_module
from data.mnli import MnliLoader

_HF_REVISION = "da70db2af9d09693783c3320c4249840212ee221"


def _pair(
    *,
    prompt_id: int,
    pair_id: str,
    premise: str,
    hypothesis: str,
    label: int,
    genre: str,
) -> dict[str, Any]:
    """One raw HF row, with the parse columns the loader ignores."""
    return {
        "promptID": prompt_id,
        "pairID": pair_id,
        "premise": premise,
        "premise_binary_parse": "( ignored )",
        "premise_parse": "(ROOT ignored)",
        "hypothesis": hypothesis,
        "hypothesis_binary_parse": "( ignored )",
        "hypothesis_parse": "(ROOT ignored)",
        "genre": genre,
        "label": label,
    }


# Two promptIDs per split, each with several hypotheses, in HF row order.
_MATCHED = [
    _pair(
        prompt_id=63735,
        pair_id="63735n",
        premise="The new rights are nice enough",
        hypothesis="Everyone really likes the newest benefits",
        label=1,
        genre="slate",
    ),
    _pair(
        prompt_id=63735,
        pair_id="63735e",
        premise="The new rights are nice enough",
        hypothesis="The new rights are acceptable",
        label=0,
        genre="slate",
    ),
    _pair(
        prompt_id=63735,
        pair_id="63735c",
        premise="The new rights are nice enough",
        hypothesis="Nobody was granted any new rights",
        label=2,
        genre="slate",
    ),
    _pair(
        prompt_id=91383,
        pair_id="91383c",
        premise="He turned and smiled at Vrenna",
        hypothesis="He smiled at Vrenna who was walking away",
        label=1,
        genre="fiction",
    ),
    _pair(
        prompt_id=91383,
        pair_id="91383e",
        premise="He turned and smiled at Vrenna",
        hypothesis="He smiled at Vrenna",
        label=0,
        genre="fiction",
    ),
]

_MISMATCHED = [
    _pair(
        prompt_id=31193,
        pair_id="31193n",
        premise="Your contribution helped make it possible",
        hypothesis="Your contributions were of no help",
        label=2,
        genre="letters",
    ),
    _pair(
        prompt_id=31193,
        pair_id="31193e",
        premise="Your contribution helped make it possible",
        hypothesis="Your contribution was helpful",
        label=0,
        genre="letters",
    ),
]


@pytest.fixture
def stub_hf(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Replace `load_dataset` with in-memory splits; record how it was called."""
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(dataset_id: str, *, revision: str) -> dict[str, list[dict[str, Any]]]:
        calls.append((dataset_id, revision))
        return {
            "train": [],  # must never be read (D2: validation splits only)
            "validation_matched": _MATCHED,
            "validation_mismatched": _MISMATCHED,
        }

    monkeypatch.setattr(mnli_module, "load_dataset", fake_load_dataset)
    return calls


def _load() -> list[Any]:
    loader = MnliLoader(hf_revision=_HF_REVISION, normalize=["nfc", "collapse_whitespace"])
    return list(loader.load())


def test_mnli_loader_pins_the_revision(stub_hf: list[tuple[str, str]]) -> None:
    _load()
    assert stub_hf == [("nyu-mll/multi_nli", _HF_REVISION)]


def test_mnli_loader_yields_first_pair_per_prompt_id_across_both_splits(
    stub_hf: list[tuple[str, str]],
) -> None:
    del stub_hf
    rows = _load()

    # R5: exactly one row per promptID, deterministically the first in HF row
    # order, with `validation_matched` read before `validation_mismatched`.
    assert [row.id for row in rows] == ["63735n", "91383c", "31193n"]
    assert [row.meta["prompt_id"] for row in rows] == [63735, 91383, 31193]
    assert len({row.meta["prompt_id"] for row in rows}) == len(rows)


def test_mnli_loader_maps_fields_per_the_row_contract(stub_hf: list[tuple[str, str]]) -> None:
    del stub_hf
    rows = _load()
    first = rows[0]

    # §4.1: x = premise (the only field Stage 2 rewrites), y = the label code.
    assert first.x == "The new rights are nice enough"
    assert first.y == 1
    assert first.meta["hypothesis"] == "Everyone really likes the newest benefits"
    assert first.meta["label_name"] == "neutral"
    assert first.meta["genre"] == "slate"


def test_mnli_loader_label_names_match_the_class_label_order(
    stub_hf: list[tuple[str, str]],
) -> None:
    del stub_hf
    expected = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for row in _load():
        assert row.y in expected
        assert row.meta["label_name"] == expected[row.y]


def test_mnli_loader_is_deterministic(stub_hf: list[tuple[str, str]]) -> None:
    del stub_hf
    first = [(row.id, row.x, row.y) for row in _load()]
    second = [(row.id, row.x, row.y) for row in _load()]
    assert first == second


def test_mnli_loader_skips_unlabeled_pairs_without_claiming_the_prompt_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `label == -1` row must not shadow a labeled sibling, nor index from the end."""
    unlabeled = _pair(
        prompt_id=63735,
        pair_id="63735x",
        premise="The new rights are nice enough",
        hypothesis="unlabeled",
        label=-1,
        genre="slate",
    )

    def fake_load_dataset(dataset_id: str, *, revision: str) -> dict[str, list[dict[str, Any]]]:
        del dataset_id, revision
        return {
            "validation_matched": [unlabeled, *_MATCHED],
            "validation_mismatched": [],
        }

    monkeypatch.setattr(mnli_module, "load_dataset", fake_load_dataset)
    rows = _load()

    assert [row.id for row in rows] == ["63735n", "91383c"]
    assert all(row.y != -1 for row in rows)


def test_mnli_loader_meta_carries_no_unexpected_keys(stub_hf: list[tuple[str, str]]) -> None:
    """`meta` is written into every variant CSV verbatim — keep it to the §4.1 set."""
    del stub_hf
    for row in _load():
        assert set(row.meta) == {"hypothesis", "label_name", "prompt_id", "genre"}
