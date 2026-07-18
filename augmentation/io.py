"""CSV / sidecar writers for Stage 2 output (STAGE2_CONTRACT IO section)."""

import csv
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from augmentation.base import AugmentedRow, ValidationResult

_COLUMNS = [
    "id",
    "variant",
    "x",
    "y",
    "meta",
    "augmenter_model",
    "prompt_hash",
    "validators",
]


class AugmentToolVersions(BaseModel, extra="forbid"):
    """Tool versions recorded in the Stage 2 sidecar."""

    python: str
    pandas: str


class AugmentRowCounts(BaseModel, extra="forbid"):
    """Row counts through the Stage 2 augmentation step."""

    input_rows: int
    augmented: int
    validators_passed_by_name: dict[str, int]
    validators_failed_by_name: dict[str, int]
    written: int


class CacheStats(BaseModel, extra="forbid"):
    """Response cache hit/miss counters for one variant run."""

    hits: int
    misses: int


class AugmentSidecar(BaseModel, extra="forbid"):
    """Stage 2 sidecar written to `{variant}.meta.json`."""

    stage: Literal["augment"] = "augment"
    dataset: str
    variant: str
    config_path: str
    config_hash: str
    resolved_config: dict[str, Any]
    augmenter_model: str
    prompt_file: str
    prompt_hash: str
    tool_versions: AugmentToolVersions
    row_counts: AugmentRowCounts
    cache_stats: CacheStats
    timestamp_utc: str


def write_variant_csv(path: Path, rows: list[tuple[AugmentedRow, list[ValidationResult]]]) -> None:
    """Write augmented rows + validator results as a variant CSV (STAGE2_CONTRACT IO section)."""
    records = [
        {
            "id": ex.id,
            "variant": ex.variant,
            "x": ex.x,
            "y": str(ex.y),
            "meta": json.dumps(ex.meta, sort_keys=True) if ex.meta else "{}",
            "augmenter_model": ex.augmenter_model,
            "prompt_hash": ex.prompt_hash,
            "validators": json.dumps(
                {
                    vr.name: {
                        "passed": vr.passed,
                        "score": vr.score,
                        "details": vr.details,
                    }
                    for vr in results
                },
                sort_keys=True,
            ),
        }
        for ex, results in rows
    ]
    df = pd.DataFrame.from_records(records, columns=_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_sidecar(path: Path, sidecar: AugmentSidecar) -> None:
    """Write the Stage 2 sidecar JSON (`{variant}.meta.json`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
