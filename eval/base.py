"""Evaluator Protocol and RunResult (ARCHITECTURE §2.6)."""

import abc
import csv
import json
import time
from collections.abc import Callable
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


def strip_reasoning_trace(raw: str) -> str:
    """Drop a leading <think>...</think> block so parsers see only the final answer.

    Reasoning-distill models (e.g. DeepSeek-R1 distills) always emit a chain-of-thought
    block that mentions every answer option before concluding, regardless of system-prompt
    instructions asking for a bare answer. Evaluator.parse() implementations must call this
    first, or naive first-occurrence matching will pick up letters/words from the reasoning
    instead of the model's actual final answer.
    """
    if "</think>" in raw:
        return raw.split("</think>", 1)[1]
    return raw


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

    def _score_batch(
        self, predictions: list[Prediction], examples: list[AugmentedRow]
    ) -> list[Prediction]:
        """Parse and score one batch of raw predictions against their examples."""
        scored: list[Prediction] = []
        for pred, ex in zip(predictions, examples, strict=True):
            parsed, parse_status = self.parse(pred.raw, ex)
            correct = self.score(parsed, ex.y)
            new_meta = dict(pred.meta)
            new_meta.update({"parse_status": parse_status, "correct": correct})
            scored.append(replace(pred, parsed=parsed, meta=new_meta))
        return scored

    def _run_inference(
        self,
        model: Model,
        examples: list[AugmentedRow],
        on_new_predictions: Callable[[list[Prediction]], None] | None = None,
    ) -> tuple[list[Prediction], float]:
        """Run batch inference, scoring each batch immediately so progress can be checkpointed.

        `on_new_predictions`, if given, is called with each freshly-scored batch as soon
        as it's ready, so a caller can persist progress to disk before the next batch runs.
        """
        formatted_inputs = [self.format(ex) for ex in examples]
        batch_size = self.cfg.batch.size
        scored: list[Prediction] = []

        variant = examples[0].variant if examples else "?"

        start_time = time.perf_counter()
        total = len(formatted_inputs)
        print(f"Starting inference: {self.dataset}/{variant} ({total} examples)...", flush=True)
        with tqdm(
            total=total,
            desc=f"{self.dataset}/{variant}",
            unit="ex",
        ) as pbar:
            for i in range(0, len(formatted_inputs), batch_size):
                batch = formatted_inputs[i : i + batch_size]
                batch_examples = examples[i : i + batch_size]
                batch_predictions = model.predict(batch)
                batch_scored = self._score_batch(batch_predictions, batch_examples)
                scored.extend(batch_scored)
                if on_new_predictions is not None:
                    on_new_predictions(batch_scored)
                pbar.update(len(batch_scored))
                # Plain newline-based heartbeat as a fallback for environments (e.g.
                # Colab `!` cells) where tqdm's `\r`-based redraws don't render live.
                print(f"  {self.dataset}/{variant}: {len(scored)}/{total} done", flush=True)
        wall_time = time.perf_counter() - start_time

        return scored, wall_time

    def _aggregate(
        self, variant: Variant, scored_predictions: list[Prediction], wall_time: float
    ) -> RunResult:
        """Build the RunResult (metrics + meta) from already-scored predictions."""
        n = len(scored_predictions)
        n_correct = sum(1 for p in scored_predictions if p.meta.get("correct"))
        unparseable_ids = [
            p.id for p in scored_predictions if p.meta.get("parse_status") == "unparseable"
        ]
        n_unparseable = len(unparseable_ids)
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

    def run_variant(
        self,
        model: Model,
        variant_csv: Path,
        limit: int | None = None,
        already_done: dict[str, Prediction] | None = None,
        prior_wall_time: float = 0.0,
        on_new_predictions: Callable[[list[Prediction]], None] | None = None,
    ) -> RunResult:
        """Evaluate a model on a variant CSV file.

        `already_done` (example id -> already-scored Prediction) lets a resumed run
        skip examples a prior session already completed; only the remaining rows go
        through inference. `on_new_predictions` is invoked with each freshly-scored
        batch so a caller can checkpoint progress to disk mid-run.
        """
        examples = self._load_rows(variant_csv, limit)
        already_done = already_done or {}
        remaining = [ex for ex in examples if ex.id not in already_done]

        if remaining:
            new_scored, new_wall_time = self._run_inference(model, remaining, on_new_predictions)
        else:
            new_scored, new_wall_time = [], 0.0

        by_id = {**already_done, **{p.id: p for p in new_scored}}
        merged = [by_id[ex.id] for ex in examples if ex.id in by_id]

        return self._aggregate(examples[0].variant, merged, prior_wall_time + new_wall_time)
