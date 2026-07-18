"""Shared test helper for Stage 4 (Visualize) tests: builds toy result JSON files.

Not collected as a test module (no ``test_*`` functions), only imported by
``tests/test_analysis_*.py`` and ``tests/test_smoke.py``.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from analysis.results import VARIANTS


def write_result_file(
    results_dir: Path,
    *,
    dataset: str,
    model_id: str,
    correct_by_variant: Mapping[str, Sequence[bool]],
    model_revision: str = "rev0",
    config_hash: str = "cfg0",
    prompt_hash: str = "ph0",
    variants_run: Sequence[str] | None = None,
) -> Path:
    """Write a toy Stage 3 result JSON with known per-variant correctness.

    Args:
        results_dir: Root directory the file is written under, as
            ``results_dir/{dataset}/{model_id}.json``.
        dataset: Dataset name.
        model_id: Model id.
        correct_by_variant: Mapping from variant name to an ordered sequence
            of `.correct` booleans. Keys present here are the only variants
            included in each `per_task` entry — omit a key (while passing a
            complete `variants_run`) to build a per_task-level "missing
            variant" fixture. All sequences must share the same length.
        model_revision: Value for the top-level `model_revision` field.
        config_hash: Value for the top-level `config_hash` field.
        prompt_hash: Value for the top-level `prompt_hash` field.
        variants_run: Value for the top-level `variants_run` field. Defaults
            to the full `VARIANTS` triple; pass a subset to build a
            variants_run-level "missing variant" fixture.

    Returns:
        The path the file was written to.
    """
    lengths = {len(flags) for flags in correct_by_variant.values()}
    if len(lengths) > 1:
        raise ValueError(f"correct_by_variant sequences must share a length, got {lengths}")
    n = next(iter(lengths), 0)

    if variants_run is None:
        variants_run = list(VARIANTS)

    per_task: list[dict[str, object]] = []
    for i in range(n):
        task: dict[str, object] = {"id": f"task-{i}", "y": "0"}
        for variant, flags in correct_by_variant.items():
            correct = flags[i]
            task[variant] = {
                "raw": "positive" if correct else "negative",
                "parsed": "0",
                "parse_status": "ok",
                "correct": correct,
            }
        per_task.append(task)

    data = {
        "dataset": dataset,
        "model_id": model_id,
        "model_revision": model_revision,
        "config_hash": config_hash,
        "prompt_hash": prompt_hash,
        "variants_run": list(variants_run),
        "per_task": per_task,
        # Extra top-level fields present in real Stage 3 output but unused
        # by the Stage 4 loader — included for schema realism only.
        "metrics": {},
        "tool_versions": {},
    }

    path = results_dir / dataset / f"{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
