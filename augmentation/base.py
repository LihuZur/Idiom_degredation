"""Augmenter / Validator types and Protocols (ARCHITECTURE §2.3)."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from data.base import DatasetRow

Variant = Literal["original", "paraphrase", "idiomatic"]


@dataclass(frozen=True, slots=True)
class AugmentedRow:
    """One row of an augmented variant. Aligned to its `original` row by `id`."""

    id: str
    variant: Variant
    x: str
    y: Any
    augmenter_model: str
    prompt_hash: str
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Per-validator outcome for a single `AugmentedRow`."""

    name: str
    passed: bool
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict[str, Any])


@runtime_checkable
class Augmenter(Protocol):
    """Reword `x` into the assigned variant. `id` and `y` are preserved."""

    variant: Variant
    augmenter_model: str

    def augment(self, ex: DatasetRow) -> AugmentedRow: ...


@runtime_checkable
class Validator(Protocol):
    """A single validator gate for augmented rows."""

    name: str

    def validate(self, ex: AugmentedRow) -> ValidationResult: ...
