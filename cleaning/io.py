"""CSV / sidecar writers for Stage 1 output (STAGE1_PLAN §3.4)."""

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from data.base import DatasetRow

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


class ToolVersions(BaseModel, extra="forbid"):
    """Tool versions recorded in the Stage 1 sidecar (STAGE1_PLAN §1)."""

    python: str
    datasets: str
    pandas: str


class RowCounts(BaseModel, extra="forbid"):
    """Row counts through each Stage 1 step (STAGE1_PLAN §1)."""

    raw_loaded: int
    after_normalize: int
    after_length_filter: int
    after_dedupe: int
    after_shuffle_and_cap: int
    written: int


class CleanSidecar(BaseModel, extra="forbid"):
    """Stage 1 sidecar written to `original.meta.json` (STAGE1_PLAN §1)."""

    stage: Literal["clean"] = "clean"
    dataset: str
    config_path: str
    config_hash: str
    resolved_config: dict[str, Any]
    hf_dataset_id: str
    hf_revision: str
    tool_versions: ToolVersions
    row_counts: RowCounts
    timestamp_utc: str


def write_original_csv(path: Path, rows: Iterable[DatasetRow]) -> None:
    """Write Stage 1 `DatasetRow`s as `original.csv` (STAGE1_PLAN §1 output contract)."""
    records = [
        {
            "id": ex.id,
            "variant": "original",
            "x": ex.x,
            "y": str(ex.y),
            "meta": json.dumps(ex.meta, sort_keys=True) if ex.meta else "{}",
            "augmenter_model": "",
            "prompt_hash": "",
            "validators": "",
        }
        for ex in rows
    ]
    df = pd.DataFrame.from_records(records, columns=_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_sidecar(path: Path, sidecar: CleanSidecar) -> None:
    """Write the Stage 1 sidecar JSON (`original.meta.json`, STAGE1_PLAN §1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar.model_dump(), indent=2, sort_keys=True))
