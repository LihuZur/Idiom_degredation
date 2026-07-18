"""Typed loaders for Stage 3 result files (ARCHITECTURE §2.7).

Parses ``results/{dataset}/{model}.json`` into pydantic models and enforces
the "complete-triple" invariant: every loaded run must have run and recorded
all three variants (original, paraphrase, idiomatic) for every task.
"""

import json
from pathlib import Path

from pydantic import BaseModel

VARIANTS: tuple[str, str, str] = ("original", "paraphrase", "idiomatic")


class PerTaskVariant(BaseModel):
    """A single variant's model output and correctness for one task.

    `parsed` is `None` when the model output could not be parsed into a label
    (`parse_status != "ok"`); analysis derives numbers from `correct`, so an
    unparseable output is simply an incorrect one.
    """

    raw: str
    parsed: str | None
    parse_status: str
    correct: bool


class PerTask(BaseModel):
    """One task's gold label and per-variant outputs."""

    id: str
    y: str
    variants: dict[str, PerTaskVariant]


class ResultFile(BaseModel):
    """The subset of a Stage 3 result JSON relevant to analysis."""

    dataset: str
    model_id: str
    model_revision: str
    config_hash: str
    prompt_hash: str
    variants_run: list[str]
    per_task: list[PerTask]


def load_result(path: Path) -> ResultFile:
    """Load and validate a single Stage 3 result JSON file.

    Args:
        path: Path to a ``results/{dataset}/{model}.json`` file.

    Returns:
        The parsed and validated `ResultFile`.

    Raises:
        ValueError: If the complete-triple invariant is violated: any of
            `VARIANTS` is missing from `variants_run`, or any `per_task`
            entry is missing one of the variant sub-objects.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    variants_run: list[str] = raw["variants_run"]
    if not set(VARIANTS).issubset(variants_run):
        missing = set(VARIANTS) - set(variants_run)
        raise ValueError(
            f"{path}: incomplete variant triple — variants_run is missing {sorted(missing)}"
        )

    per_task: list[PerTask] = []
    for i, entry in enumerate(raw["per_task"]):
        missing_keys = [v for v in VARIANTS if v not in entry]
        if missing_keys:
            raise ValueError(
                f"{path}: per_task[{i}] (id={entry.get('id', '?')!r}) is missing "
                f"variant(s) {missing_keys}"
            )
        per_task.append(
            PerTask(
                id=entry["id"],
                y=entry["y"],
                variants={v: PerTaskVariant(**entry[v]) for v in VARIANTS},
            )
        )

    return ResultFile(
        dataset=raw["dataset"],
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        config_hash=raw["config_hash"],
        prompt_hash=raw["prompt_hash"],
        variants_run=variants_run,
        per_task=per_task,
    )


def discover_results(results_dir: Path) -> list[Path]:
    """Find all Stage 3 result files under `results_dir`.

    Args:
        results_dir: Root directory containing ``{dataset}/{model}.json`` files.

    Returns:
        A sorted list of matching paths, for deterministic ordering.
    """
    return sorted(results_dir.glob("*/*.json"))


def correct_by_variant(result: ResultFile) -> dict[str, list[bool]]:
    """Extract paired per-task correctness flags for each variant.

    Args:
        result: A loaded, validated `ResultFile`.

    Returns:
        A mapping from each of `VARIANTS` to the ordered list of `.correct`
        booleans across `result.per_task`. All three lists are paired: same
        length, same task order.
    """
    return {v: [task.variants[v].correct for task in result.per_task] for v in VARIANTS}
