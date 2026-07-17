"""MMLU raw dataset loader (STAGE1_PLAN §3.1)."""

from collections.abc import Iterable
from typing import Any, cast

from datasets import load_dataset

from data.base import DatasetRow
from data.registry import register_dataset


@register_dataset("mmlu")
class MmluLoader:
    """Loads `cais/mmlu` config `"all"`, HF `validation` split only (Q3)."""

    name = "mmlu"
    hf_dataset_id = "cais/mmlu"

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        # MMLU ids are `{subject}/{row_index}` and don't depend on normalization.
        del normalize

    def load(self) -> Iterable[DatasetRow]:
        """Yield MMLU rows from HF `validation`, covering all 57 subjects."""
        ds = load_dataset(self.hf_dataset_id, "all", revision=self._hf_revision)
        validation = ds["validation"]
        for i, raw_row in enumerate(validation):
            row = cast(dict[str, Any], raw_row)
            meta = {
                "subject": row["subject"],
                "choices": row["choices"],
                "answer_index": row["answer"],
            }
            yield DatasetRow(
                id=f"{row['subject']}/{i}",
                x=row["question"],
                y=row["answer"],
                meta=meta,
            )
