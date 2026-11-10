"""MNLI (MultiNLI) raw dataset loader (MNLI_DATASET_PLAN §5.1)."""

from collections.abc import Iterable
from itertools import chain, islice
from typing import Any, TypedDict, cast

from datasets import load_dataset

from data.base import DatasetRow
from data.registry import register_dataset

# MNLI's `label` ClassLabel order. Recorded verbatim in `meta.label_name` (R8) so
# the augmenter and its judges see the relation in words rather than the bare
# integer `{y}` renders (plan §4.4).
_LABEL_NAMES = ("entailment", "neutral", "contradiction")

# D2: both validation halves. `validation_mismatched` adds 5 held-out genres, so
# genre shift is a documented confound (R6) — `meta.genre` records it.
_SPLITS = ("validation_matched", "validation_mismatched")

_HF_DATASET_ID = "nyu-mll/multi_nli"
_WIDE_TRAIN_PAIRS = 20000


class _SelectedPair(TypedDict):
    """One selected MNLI pair, before either loader assigns x/meta roles."""

    pair_id: str
    premise: str
    hypothesis: str
    label: int
    label_name: str
    prompt_id: int
    genre: str


def _iter_selected_pairs(
    hf_revision: str, splits: tuple[str, ...] = _SPLITS
) -> Iterable[_SelectedPair]:
    """Yield one pair per premise — the first pair seen for each `promptID` (R5).

    Shared by `MnliLoader` (RQ1 chapter: rewrites the premise) and
    `MnliHypothesisLoader` (RQ3 §4: rewrites the hypothesis instead), so both
    loaders select the exact same 4,709 pairs/ids — required for RQ3's
    same-row-id comparison against Ch. 1.

    MNLI pairs ~3 hypotheses with every premise. Emitting all of them would put
    the same premise in the dataset repeatedly, breaking the independence
    assumption of Stage 4's paired significance test and tripling Stage 2's API
    spend for no extra premise coverage. The selection is deliberately
    hardcoded rather than configurable, so Stage 1 stays dataset-agnostic
    (`cleaning/pipeline.py`); changing it means editing this module.

    Selection is deterministic under the pinned `hf_revision`: splits are read
    in `_SPLITS` order and rows in HF row order.
    """
    ds = load_dataset(_HF_DATASET_ID, revision=hf_revision)
    seen_prompt_ids: set[int] = set()
    for split_name in splits:
        for raw_row in ds[split_name]:
            row = cast(dict[str, Any], raw_row)
            label = cast(int, row["label"])
            # HF marks unlabeled pairs with -1. Both validation splits are fully
            # labeled, but skip before claiming the promptID so an anomalous row
            # can't shadow a labeled sibling (and can't index `_LABEL_NAMES`
            # from the end).
            if not 0 <= label < len(_LABEL_NAMES):
                continue
            prompt_id = cast(int, row["promptID"])
            if prompt_id in seen_prompt_ids:
                continue
            seen_prompt_ids.add(prompt_id)
            yield _SelectedPair(
                pair_id=cast(str, row["pairID"]),
                premise=cast(str, row["premise"]),
                hypothesis=cast(str, row["hypothesis"]),
                label=label,
                label_name=_LABEL_NAMES[label],
                prompt_id=prompt_id,
                genre=cast(str, row["genre"]),
            )


@register_dataset("mnli")
class MnliLoader:
    """Loads `nyu-mll/multi_nli`, merging both HF validation splits (D2)."""

    name = "mnli"
    hf_dataset_id = _HF_DATASET_ID

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        # MNLI ids are the source `pairID` and don't depend on normalization.
        del normalize

    def load(self) -> Iterable[DatasetRow]:
        """Yield one row per premise (§4.1: x = premise, the field Stage 2 rewrites).

        Consequence of the shared selection: the Stage 1 sidecar's
        `row_counts.raw_loaded` reports the post-selection count (~6,500), not
        the 19,647 pairs in the two splits (plan §4.2).
        """
        for pair in _iter_selected_pairs(self._hf_revision):
            yield DatasetRow(
                id=pair["pair_id"],
                x=pair["premise"],
                y=pair["label"],
                meta={
                    # Fixed reference side of the pair: never rewritten by Stage 2,
                    # so it stays byte-identical across the variant triple (D3).
                    "hypothesis": pair["hypothesis"],
                    "label_name": pair["label_name"],
                    "prompt_id": pair["prompt_id"],
                    "genre": pair["genre"],
                },
            )


@register_dataset("mnli_hypothesis")
class MnliHypothesisLoader:
    """RQ3 §4 position-variation variant: rewrites the hypothesis, not the premise.

    Selects the exact same pairs/ids as `MnliLoader` (`_iter_selected_pairs` is
    shared) so RQ3's Δ_idi is directly comparable, row-for-row, to Ch. 1's
    premise-position result. Only the x/meta role assignment is swapped: `x`
    (the field Stage 2 rewrites) is now the hypothesis, and the premise moves
    to `meta["premise"]` — a fixed reference `augmentation.prompts.loader
    .render_context` renders read-only into the augmenter/judge prompts,
    mirroring how `MnliLoader` fixes the hypothesis.
    """

    name = "mnli_hypothesis"
    hf_dataset_id = _HF_DATASET_ID

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        del normalize

    def load(self) -> Iterable[DatasetRow]:
        """Yield one row per pair, `x` = hypothesis, `meta["premise"]` = the fixed premise."""
        for pair in _iter_selected_pairs(self._hf_revision):
            yield DatasetRow(
                id=pair["pair_id"],
                x=pair["hypothesis"],
                y=pair["label"],
                meta={
                    "premise": pair["premise"],
                    "label_name": pair["label_name"],
                    "prompt_id": pair["prompt_id"],
                    "genre": pair["genre"],
                },
            )


@register_dataset("mnli_wide")
class MnliWideLoader:
    """Candidate pool for idiom insertion: both validation halves (same ids as `mnli`) plus the
    first 20,000 premises of `train`, so every existing row and its GPU results stay in the pool.
    Until 2026-10-18 it took 6,000; the train premises are a prefix, so that pool is a subset.

    One pair per premise, like `MnliLoader`; the training split has the same genres as
    `validation_matched`.
    """

    name = "mnli_wide"
    hf_dataset_id = _HF_DATASET_ID

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        del normalize

    def load(self) -> Iterable[DatasetRow]:
        validation = list(_iter_selected_pairs(self._hf_revision))
        taken = {p["pair_id"] for p in validation} | {p["premise"] for p in validation}
        # A few training premises also occur in validation (same pairID); skip them.
        extra = (
            p
            for p in _iter_selected_pairs(self._hf_revision, ("train",))
            if p["pair_id"] not in taken and p["premise"] not in taken
        )
        for pair in chain(validation, islice(extra, _WIDE_TRAIN_PAIRS)):
            yield DatasetRow(
                id=pair["pair_id"],
                x=pair["premise"],
                y=pair["label"],
                meta={
                    "hypothesis": pair["hypothesis"],
                    "label_name": pair["label_name"],
                    "prompt_id": pair["prompt_id"],
                    "genre": pair["genre"],
                },
            )
