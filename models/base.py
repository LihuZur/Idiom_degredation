"""Types and Protocol for evaluation-time models (ARCHITECTURE §2.5)."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch

Precision = Literal["fp32", "fp16", "bf16"]
ModelKind = Literal["decoder"]  # current phase (README §4)


@dataclass(frozen=True, slots=True)
class FormattedInput:
    """A dataset-specific prompt-ready input for a single example.

    The prompt string is produced by the dataset's `Evaluator.format`.
    """

    id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True, slots=True)
class Prediction:
    """Model output for one `FormattedInput`."""

    id: str
    raw: str
    parsed: Any
    meta: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static metadata for a registered model (recorded in each result file)."""

    name: str
    hf_repo: str
    revision: str | None
    kind: ModelKind
    default_precision: Precision


@runtime_checkable
class Model(Protocol):
    """Uniform runtime interface consumed by Stage 3 (`eval/`)."""

    id: str
    device: torch.device

    def predict(self, batch: list[FormattedInput]) -> list[Prediction]: ...
