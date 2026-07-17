"""Evaluator Protocol and RunResult (ARCHITECTURE §2.6)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from augmentation.base import AugmentedExample, Variant
from models.base import FormattedInput, Model, Prediction


@dataclass(frozen=True, slots=True)
class RunResult:
    """Per-variant aggregate + per-example predictions for one Stage 3 run."""

    variant: Variant
    metrics: dict[str, float]
    predictions: list[Prediction]
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


@runtime_checkable
class Evaluator(Protocol):
    """A dataset-specific Stage 3 evaluator (ARCHITECTURE §2.6)."""

    dataset: str
    metric: Callable[[list[Prediction], list[Any]], dict[str, float]]

    def format(self, ex: AugmentedExample, variant: Variant) -> FormattedInput: ...

    def run_variant(self, model: Model, variant_csv: Path) -> RunResult: ...
