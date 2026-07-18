"""Atomic writers and provenance metadata for Stage 4 outputs (ARCHITECTURE §2.7).

Mirrors the tmp-file-then-replace atomic write pattern used in `eval/io.py`.
"""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly

from analysis.results import ResultFile


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a DataFrame to CSV.

    Args:
        df: The table to write.
        path: Destination `.csv` path; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a DataFrame as a Markdown table.

    Args:
        df: The table to write.
        path: Destination `.md` path; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # tabulate is a runtime-only dependency backing `to_markdown`; no stubs
    # are shipped for it, so its return type is partially unknown to pyright.
    text = df.to_markdown(index=False)  # pyright: ignore[reportUnknownMemberType]
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_sidecar(meta: dict[str, Any], path: Path) -> None:
    """Atomically write a JSON sidecar file.

    Args:
        meta: JSON-serializable metadata to write.
        path: Destination `.json` path; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_provenance(
    results: list[ResultFile],
    *,
    n_resamples: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    """Build a provenance dict describing how a Stage 4 output was produced.

    Args:
        results: The result files that fed into the aggregation.
        n_resamples: Number of bootstrap resamples used.
        ci: Confidence level used for bootstrap intervals.
        seed: RNG seed used for bootstrap resampling.

    Returns:
        A JSON-serializable dict with tool versions, a UTC timestamp,
        bootstrap parameters, and per-run source hashes/metadata. Sources are
        keyed by ``"{dataset}/{model_id}"`` so runs that share a model id
        across datasets are each recorded (a bare model id is not unique).
    """
    return {
        "tool_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "plotly": plotly.__version__,
        },
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "bootstrap": {
            "n_resamples": n_resamples,
            "ci": ci,
            "seed": seed,
        },
        "sources": {
            f"{result.dataset}/{result.model_id}": {
                "config_hash": result.config_hash,
                "model_revision": result.model_revision,
                "prompt_hash": result.prompt_hash,
                "dataset": result.dataset,
                "model_id": result.model_id,
            }
            for result in results
        },
    }
