"""Types and Protocol for dataset loaders (ARCHITECTURE §2.1)."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class Example:
    """A single raw task example, before Stage 1 cleaning.

    Attributes:
        id: Stable example identifier (source id when available, else a
            content hash assigned by Stage 1).
        x: Input text.
        y: Label. Type is dataset-specific; kept as `Any` so the in-memory
            schema is uniform across datasets.
        split: Source split.
        meta: Dataset-specific fields (e.g. MMLU choices / subject).
    """

    id: str
    x: str
    y: Any
    split: Split
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


@runtime_checkable
class DatasetLoader(Protocol):
    """A raw HF dataset loader (ARCHITECTURE §2.1)."""

    name: str

    def load(self, split: Split) -> Iterable[Example]:
        """Yield raw examples for the given split.

        No filtering / cleaning happens here — that is Stage 1's job.
        """
        ...
