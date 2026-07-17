"""SST-2 raw dataset loader (STAGE1_PLAN §3.1)."""

import hashlib
from collections.abc import Iterable
from typing import Any, cast

from datasets import load_dataset

from cleaning.normalize import apply as apply_normalizers
from data.base import DatasetRow
from data.registry import register_dataset


@register_dataset("sst2")
class Sst2Loader:
    """Loads `stanfordnlp/sst2`, merging HF `train` + `validation` (Q1/Q2)."""

    name = "sst2"
    hf_dataset_id = "stanfordnlp/sst2"

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        self._normalize = normalize

    def load(self) -> Iterable[DatasetRow]:
        """Yield SST-2 rows from HF `train` + `validation` (HF `test` is skipped)."""
        ds = load_dataset(self.hf_dataset_id, revision=self._hf_revision)
        for split_name in ("train", "validation"):
            for raw_row in ds[split_name]:
                row = cast(dict[str, Any], raw_row)
                raw_x = cast(str, row["sentence"])
                # Q5: id = sha256(normalized_x)[:16]. The loader applies the same
                # normalizer chain the pipeline will apply, so ids stay a stable
                # function of the normalized text (STAGE1_PLAN §7 gotcha).
                normalized_x = apply_normalizers(self._normalize, raw_x)
                example_id = hashlib.sha256(normalized_x.encode("utf-8")).hexdigest()[:16]
                yield DatasetRow(id=example_id, x=raw_x, y=row["label"], meta={})
