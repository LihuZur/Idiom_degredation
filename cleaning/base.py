"""Cleaner Protocol (ARCHITECTURE §2.2)."""

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from data.base import Example


@runtime_checkable
class Cleaner(Protocol):
    """Filter, normalize, deduplicate and cap raw examples for one dataset."""

    dataset: str

    def clean(self, examples: Iterable[Example]) -> Iterable[Example]:
        """Yield the retained + normalized examples.

        Implementations are also responsible for writing
        `datasets_out/{dataset}/original.csv`.
        """
        ...
