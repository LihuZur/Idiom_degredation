"""Types and Protocol for dataset loaders (ARCHITECTURE §2.1)."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """A single raw task example, before Stage 1 cleaning.

    Attributes:
        id: Stable example identifier (source id when available, else a
            content hash assigned by Stage 1).
        x: Input text.
        y: Label. Type is dataset-specific; kept as `Any` so the in-memory
            schema is uniform across datasets.
        meta: Dataset-specific fields (e.g. MMLU choices / subject).
    """

    id: str
    x: str
    y: Any
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


@runtime_checkable
class DatasetLoader(Protocol):
    """A raw HF dataset loader (ARCHITECTURE §2.1)."""

    name: str
    hf_dataset_id: str

    def __init__(self, *, hf_revision: str, normalize: list[str]) -> None:
        """Construct a loader pinned to `hf_revision`, given the Stage 1
        normalizer chain (needed by some loaders to assign stable ids).
        """
        ...

    def load(self) -> Iterable[DatasetRow]:
        """Yield raw examples for this dataset.

        No filtering / cleaning happens here — that is Stage 1's job.
        """
        ...
