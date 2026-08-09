"""MNLI (MultiNLI) raw dataset loader (MNLI_DATASET_PLAN §5.1)."""

from collections.abc import Iterable
from typing import Any, cast

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


@register_dataset("mnli")
class MnliLoader:
    """Loads `nyu-mll/multi_nli`, merging both HF validation splits (D2)."""

    name = "mnli"
    hf_dataset_id = "nyu-mll/multi_nli"

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        # MNLI ids are the source `pairID` and don't depend on normalization.
        del normalize

    def load(self) -> Iterable[DatasetRow]:
        """Yield one row per premise — the first pair seen for each `promptID` (R5).

        MNLI pairs ~3 hypotheses with every premise. Emitting all of them would put
        the same premise in the dataset repeatedly, breaking the independence
        assumption of Stage 4's paired significance test and tripling Stage 2's
        API spend for no extra premise coverage. The selection is deliberately
        hardcoded rather than configurable, so Stage 1 stays dataset-agnostic
        (`cleaning/pipeline.py`); changing it means editing this loader.

        Selection is deterministic under the pinned `hf_revision`: splits are read
        in `_SPLITS` order and rows in HF row order. Consequence: the Stage 1
        sidecar's `row_counts.raw_loaded` reports the post-selection count
        (~6,500), not the 19,647 pairs in the two splits (plan §4.2).
        """
        ds = load_dataset(self.hf_dataset_id, revision=self._hf_revision)
        seen_prompt_ids: set[int] = set()
        for split_name in _SPLITS:
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
                yield DatasetRow(
                    id=cast(str, row["pairID"]),
                    x=cast(str, row["premise"]),
                    y=label,
                    meta={
                        # Fixed reference side of the pair: never rewritten by Stage 2,
                        # so it stays byte-identical across the variant triple (D3).
                        "hypothesis": row["hypothesis"],
                        "label_name": _LABEL_NAMES[label],
                        "prompt_id": prompt_id,
                        "genre": row["genre"],
                    },
                )
