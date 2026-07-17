"""Cleaner Protocol (ARCHITECTURE §2.2)."""

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from data.base import DatasetRow


@runtime_checkable
class Cleaner(Protocol):
    """Filter, normalize, deduplicate and cap raw examples for one dataset."""

    dataset: str

    def clean(self, examples: Iterable[DatasetRow]) -> Iterable[DatasetRow]:
        """Yield the retained + normalized examples.

        Implementations are also responsible for writing
        `datasets_out/{dataset}/original.csv`.
        """
        ...
