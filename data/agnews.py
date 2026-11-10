"""AG News raw dataset loader (THESIS_PLAN_v2 §1): topic classification.

`fancyzhx/ag_news` (Zhang, Zhao & LeCun 2015): news snippets (title + description) in four
balanced classes. The standard 7,600-row `test` split is used; `agnews_wide` adds training rows as
a larger candidate pool for the v2 idiom generator.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from typing import Any, cast

from datasets import load_dataset

from cleaning.normalize import apply as apply_normalizers
from data.base import DatasetRow
from data.registry import register_dataset

LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")
_HF_DATASET_ID = "fancyzhx/ag_news"
_WIDE_TRAIN_ROWS = 20000


def _iter_rows(
    hf_revision: str, normalize: list[str], split: str, extra_meta: dict[str, Any] | None = None
) -> Iterable[DatasetRow]:
    """Rows of one split; ids are a hash of the normalized text (stable across pools)."""
    ds = load_dataset(_HF_DATASET_ID, revision=hf_revision, split=split)
    for raw_row in ds:
        row = cast(dict[str, Any], raw_row)
        raw_x = cast(str, row["text"])
        normalized_x = apply_normalizers(normalize, raw_x)
        example_id = hashlib.sha256(normalized_x.encode("utf-8")).hexdigest()[:16]
        label = int(row["label"])
        yield DatasetRow(
            id=example_id,
            x=raw_x,
            y=label,
            meta={"label_name": LABEL_NAMES[label], **(extra_meta or {})},
        )


@register_dataset("agnews")
class AgNewsLoader:
    """Loads `fancyzhx/ag_news` `test` (7,600 rows, 1,900 per class)."""

    name = "agnews"
    hf_dataset_id = _HF_DATASET_ID

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        self._normalize = normalize

    def load(self) -> Iterable[DatasetRow]:
        return _iter_rows(self._hf_revision, self._normalize, "test")


@register_dataset("agnews_wide")
class AgNewsWideLoader:
    """Candidate pool for idiom insertion: the whole `test` split (same ids and rows as `agnews`)
    plus the first 20,000 `train` rows in dataset order, minus any whose text repeats a test row.
    `train` contains some snippets twice with different labels (e.g. Business and Sci/Tech); those
    are dropped entirely, and of exact repeats only the first copy is kept.

    Train rows carry `meta.split = "train"`; test rows keep exactly the `agnews` meta, so every
    existing row of the old wide pool stays byte-identical.
    """

    name = "agnews_wide"
    hf_dataset_id = _HF_DATASET_ID

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        self._hf_revision = hf_revision
        self._normalize = normalize

    def load(self) -> Iterable[DatasetRow]:
        seen: set[str] = set()
        for row in _iter_rows(self._hf_revision, self._normalize, "test"):
            seen.add(row.id)
            yield row
        train = list(_iter_rows(self._hf_revision, self._normalize, "train", {"split": "train"}))
        labels: dict[str, set[int]] = defaultdict(set)
        for r in train:
            labels[r.id].add(r.y)
        ambiguous = {i for i, ys in labels.items() if len(ys) > 1}

        def keep(rows: Iterable[DatasetRow]) -> Iterable[DatasetRow]:
            for r in rows:
                if r.id not in seen and r.id not in ambiguous:
                    seen.add(r.id)
                    yield r

        yield from islice(keep(train), _WIDE_TRAIN_ROWS)
