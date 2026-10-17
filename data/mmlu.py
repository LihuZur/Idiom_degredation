"""MMLU raw dataset loader (STAGE1_PLAN §3.1)."""

from collections.abc import Iterable
from typing import Any, cast

from datasets import load_dataset

from data.base import DatasetRow
from data.registry import register_dataset

_HF_DATASET_ID = "cais/mmlu"

# Letters MMLU's fixed 4-way choice list is rendered under, both here (Stage 2's
# rewritten `x` for `mmlu_choices`) and in `augmentation/prompts/loader.py`'s
# `render_context` (the *fixed* choices block rendered for plain `mmlu`) — kept
# in sync deliberately so a rewritten choices block and a rendered read-only
# choices block always look the same to a human or a judge.
_CHOICE_LETTERS = "ABCD"


def _iter_mmlu_rows(hf_revision: str) -> Iterable[dict[str, Any]]:
    """Shared row enumeration for `MmluLoader` and `MmluChoicesLoader`.

    Both loaders read `cais/mmlu` config `"all"`, HF `validation` split, in HF
    row order, so they select the exact same items under the same ids
    (`{subject}/{row_index}`) — required for RQ3 §4's stem-vs-choices Δ_idi
    comparison (H3c). Only the x/meta role assignment differs downstream,
    mirroring `data/mnli.py`'s `_iter_selected_pairs`.
    """
    ds = load_dataset(_HF_DATASET_ID, "all", revision=hf_revision)
    validation = ds["validation"]
    for i, raw_row in enumerate(validation):
        row = cast(dict[str, Any], raw_row)
        row["_row_index"] = i
        yield row


def encode_choices(choices: list[str]) -> str:
    """Join MMLU's 4 answer choices into the single string Stage 2 rewrites.

    Same `"A) ...\\nB) ..."` shape as `render_context`'s fixed choices block,
    so the rewritten text a human reads looks identical to the read-only
    rendering used everywhere else.
    """
    return "\n".join(
        f"{letter}) {choice}" for letter, choice in zip(_CHOICE_LETTERS, choices, strict=True)
    )


def decode_choices(text: str) -> list[str]:
    """Inverse of `encode_choices`.

    Raises `ValueError` if `text` is not exactly 4 lines in the
    `"A) ...\\nB) ...\\nC) ...\\nD) ..."` shape — fails loud at eval time
    rather than silently mis-assigning a rewritten choice to the wrong letter.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) != len(_CHOICE_LETTERS):
        raise ValueError(
            f"expected {len(_CHOICE_LETTERS)} choice lines, got {len(lines)}: {text!r}"
        )
    choices: list[str] = []
    for letter, line in zip(_CHOICE_LETTERS, lines, strict=True):
        prefix = f"{letter}) "
        if not line.startswith(prefix):
            raise ValueError(f"expected line to start with {prefix!r}, got {line!r}")
        choices.append(line[len(prefix) :])
    return choices


@register_dataset("mmlu")
class MmluLoader:
    """Loads `cais/mmlu` config `"all"`, HF `validation` split only (Q3)."""

    name = "mmlu"
    hf_dataset_id = _HF_DATASET_ID

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        # MMLU ids are `{subject}/{row_index}` and don't depend on normalization.
        del normalize

    def load(self) -> Iterable[DatasetRow]:
        """Yield MMLU rows from HF `validation`, covering all 57 subjects."""
        for row in _iter_mmlu_rows(self._hf_revision):
            meta = {
                "subject": row["subject"],
                "choices": row["choices"],
                "answer_index": row["answer"],
            }
            yield DatasetRow(
                id=f"{row['subject']}/{row['_row_index']}",
                x=row["question"],
                y=row["answer"],
                meta=meta,
            )


@register_dataset("mmlu_choices")
class MmluChoicesLoader:
    """RQ3 §4 position-variation variant: rewrites the answer choices, not the
    question stem.

    Selects the exact same rows/ids as `MmluLoader` (`_iter_mmlu_rows` is
    shared) so RQ3's Δ_idi is directly comparable, row-for-row, to Ch. 1's
    question-stem-position result (H3c). Only the x/meta role assignment is
    swapped: `x` (the field Stage 2 rewrites) is now the four choices joined
    into one string via `encode_choices`, and the question moves to
    `meta["question"]` — a fixed reference `augmentation.prompts.loader
    .render_context` renders read-only into the augmenter/judge prompts,
    mirroring how `MmluLoader` fixes the choices for the plain `mmlu` variant.
    """

    name = "mmlu_choices"
    hf_dataset_id = _HF_DATASET_ID

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        del normalize

    def load(self) -> Iterable[DatasetRow]:
        """Yield one row per item, `x` = encoded choices, `meta["question"]` = the fixed stem."""
        for row in _iter_mmlu_rows(self._hf_revision):
            meta = {
                "question": row["question"],
                "subject": row["subject"],
                "answer_index": row["answer"],
            }
            yield DatasetRow(
                id=f"{row['subject']}/{row['_row_index']}",
                x=encode_choices(cast(list[str], row["choices"])),
                y=row["answer"],
                meta=meta,
            )
