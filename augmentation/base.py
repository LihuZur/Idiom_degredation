"""Augmenter / Validator types and Protocols (ARCHITECTURE §2.3)."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from augmentation.providers.base import LLMClient
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

    def __init__(
        self,
        *,
        variant: Variant,
        prompt_hash: str,
        client: LLMClient,
        prompt_template: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        """Construct an augmenter assigned to `variant`.

        Tags emitted rows with `prompt_hash` (the resolved template's hash) and
        rewrites `x` through the shared `client`, rendering `prompt_template`
        with the configured decoding params.
        """
        ...

    def augment(self, ex: DatasetRow) -> AugmentedRow: ...


@runtime_checkable
class Validator(Protocol):
    """A single validator gate for augmented rows.

    `validate` receives the augmented row and its `original` source row so
    label-preservation judges can compare both texts against the gold label.
    """

    name: str

    def validate(self, ex: AugmentedRow, original: DatasetRow) -> ValidationResult: ...
