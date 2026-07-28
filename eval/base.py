"""Evaluator Protocol and RunResult (ARCHITECTURE §2.6)."""

import abc
import csv
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from tqdm import tqdm

from augmentation.base import AugmentedRow, Variant
from eval.config import EvalConfig
from models.base import FormattedInput, Model, Prediction


def _parse_meta(meta_str: str | None) -> dict[str, Any]:
    if not meta_str:
        return {}
    try:
        val = json.loads(meta_str)
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}


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

    def format(self, ex: AugmentedRow) -> FormattedInput: ...

    def run_variant(
        self, model: Model, variant_csv: Path, limit: int | None = None
    ) -> RunResult: ...


class BaseEvaluator(abc.ABC):
    """Abstract base class for all dataset-specific evaluators (STAGE3_PLAN §3.2)."""

    dataset: ClassVar[str]

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    @abc.abstractmethod
    def format(self, ex: AugmentedRow) -> FormattedInput:
        """Format an augmented example into a prompt-ready FormattedInput."""
        ...

    @abc.abstractmethod
    def parse(self, raw: str, ex: AugmentedRow) -> tuple[str | None, Literal["ok", "unparseable"]]:
        """Parse raw model output into a task label (or None if unparseable)."""
        ...

    def score(self, parsed: str | None, y: Any) -> bool:
        """Score the parsed output against the canonical label."""
        return parsed is not None and str(parsed) == str(y)

    def _load_rows(self, variant_csv: Path, limit: int | None) -> list[AugmentedRow]:
        """Load augmented rows from CSV."""
        with variant_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                AugmentedRow(
                    id=row["id"],
                    variant=row["variant"],  # type: ignore
                    x=row["x"],
                    y=row["y"],
                    augmenter_model=row.get("augmenter_model", ""),
                    prompt_hash=row.get("prompt_hash", ""),
                    meta=_parse_meta(row.get("meta")),
                )
                for row in reader
            ]

        if limit is not None:
            rows = rows[:limit]

        if not rows:
            raise ValueError(f"No examples loaded from {variant_csv}")

        return rows

    def _run_inference(
        self, model: Model, examples: list[AugmentedRow]
    ) -> tuple[list[Prediction], float]:
        """Run batch inference on the examples and return predictions and wall time."""
        formatted_inputs = [self.format(ex) for ex in examples]
        batch_size = self.cfg.batch.size
        predictions: list[Prediction] = []

        variant = examples[0].variant if examples else "?"

        start_time = time.perf_counter()
        with tqdm(
            total=len(formatted_inputs),
            desc=f"{self.dataset}/{variant}",
            unit="ex",
        ) as pbar:
            for i in range(0, len(formatted_inputs), batch_size):
                batch = formatted_inputs[i : i + batch_size]
                batch_predictions = model.predict(batch)
                predictions.extend(batch_predictions)
                pbar.update(len(batch))
        wall_time = time.perf_counter() - start_time

        return predictions, wall_time

    def _process_results(
        self,
        predictions: list[Prediction],
        examples: list[AugmentedRow],
        wall_time: float,
    ) -> RunResult:
        """Parse predictions, score them, and build the RunResult."""
        variant = examples[0].variant
        scored_predictions: list[Prediction] = []
        unparseable_ids: list[str] = []
        n_unparseable = 0
        n_correct = 0

        for pred, ex in zip(predictions, examples, strict=True):
            parsed, parse_status = self.parse(pred.raw, ex)
            correct = self.score(parsed, ex.y)

            if parse_status == "unparseable":
                unparseable_ids.append(ex.id)
                n_unparseable += 1
            if correct:
                n_correct += 1

            new_meta = dict(pred.meta)
            new_meta.update(
                {
                    "parse_status": parse_status,
                    "correct": correct,
                }
            )
            scored_predictions.append(replace(pred, parsed=parsed, meta=new_meta))

        n = len(examples)
        accuracy = n_correct / n if n > 0 else 0.0
        unparseable_rate = n_unparseable / n if n > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "unparseable_rate": unparseable_rate,
            "n": float(n),
            "n_unparseable": float(n_unparseable),
        }

        meta = {
            "unparseable_ids": unparseable_ids,
            "wall_time_seconds": wall_time,
        }

        return RunResult(
            variant=variant,
            metrics=metrics,
            predictions=scored_predictions,
            meta=meta,
        )

    def run_variant(self, model: Model, variant_csv: Path, limit: int | None = None) -> RunResult:
        """Evaluate a model on a variant CSV file."""
        examples = self._load_rows(variant_csv, limit)
        predictions, wall_time = self._run_inference(model, examples)
        return self._process_results(predictions, examples, wall_time)
