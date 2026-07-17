"""IO helpers for writing and loading evaluation results (STAGE3_PLAN §2)."""

import json
from pathlib import Path
from typing import Any


def write_result(path: Path, result: dict[str, Any]) -> None:
    """Write the evaluation result JSON (sort_keys=True, indent=2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first, then rename to ensure atomic write
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_result(path: Path) -> dict[str, Any]:
    """Load an evaluation result JSON file."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in result file {path}")
    return data
