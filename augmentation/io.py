"""CSV / sidecar writers for Stage 2 output (STAGE2_CONTRACT IO section)."""

import csv
import json
from dataclasses import dataclass, field
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
    skipped: int = 0
    """Rows already present in the output CSV from an earlier run and resumed
    (not re-augmented) this run. 0 for a fresh full run."""


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


def _row_record(ex: AugmentedRow, results: list[ValidationResult]) -> dict[str, str]:
    """Serialize one augmented row + its validator results to a CSV record."""
    return {
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


def write_variant_csv(path: Path, rows: list[tuple[AugmentedRow, list[ValidationResult]]]) -> None:
    """Write augmented rows + validator results as a variant CSV (STAGE2_CONTRACT IO section)."""
    records = [_row_record(ex, results) for ex, results in rows]
    df = pd.DataFrame.from_records(records, columns=_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


@dataclass(slots=True)
class VariantProgress:
    """What an existing (possibly partial) variant CSV already contains.

    Used to resume a run: `ids` are the rows already written (in file order),
    and `augmenter_model` / `prompt_hash` identify which run produced them so a
    changed model or prompt can be detected and the stale CSV discarded.
    """

    ids: list[str] = field(default_factory=list[str])
    augmenter_model: str | None = None
    prompt_hash: str | None = None
    passed_by_name: dict[str, int] = field(default_factory=dict[str, int])
    failed_by_name: dict[str, int] = field(default_factory=dict[str, int])


def heal_variant_csv(path: Path) -> None:
    """Drop a torn final line left by an interrupted incremental write.

    A row is flushed as a whole line ending in ``\\n``; a process killed
    mid-write can leave a trailing partial line with no newline. Truncating
    back to the last newline yields a clean CSV that later rows append to
    safely (the dropped row is simply re-augmented on resume)."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        data = f.read()
        if data.endswith(b"\n"):
            return
        last_newline = data.rfind(b"\n")
        f.seek(last_newline + 1 if last_newline >= 0 else 0)
        f.truncate()


def read_variant_progress(path: Path) -> VariantProgress:
    """Stream an existing variant CSV into a `VariantProgress` (empty if absent).

    Call `heal_variant_csv` first so a torn tail line is already removed. Reads
    one row at a time (no full-file materialization) to keep resume memory-light
    on large datasets."""
    progress = VariantProgress()
    if not path.exists() or path.stat().st_size == 0:
        return progress
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row.get("id")
            if not row_id:
                continue
            progress.ids.append(row_id)
            if progress.augmenter_model is None:
                progress.augmenter_model = row.get("augmenter_model") or None
                progress.prompt_hash = row.get("prompt_hash") or None
            _tally_validators(row.get("validators"), progress)
    return progress


def _tally_validators(validators_json: str | None, progress: VariantProgress) -> None:
    """Fold one row's `validators` JSON column into the pass/fail tallies."""
    if not validators_json:
        return
    try:
        parsed: Any = json.loads(validators_json)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed, dict):
        return
    for name, verdict in parsed.items():
        passed = isinstance(verdict, dict) and bool(verdict.get("passed"))
        tally = progress.passed_by_name if passed else progress.failed_by_name
        tally[name] = tally.get(name, 0) + 1


class VariantCsvAppender:
    """Append-only, per-row-flushed writer for a variant CSV (resumable output).

    Writes the header only when starting a fresh (missing/empty) file and
    flushes after every row so an interrupted run leaves a durable partial CSV
    that a later run resumes from. Use as a context manager so the handle is
    always closed (and flushed) even when augmentation aborts mid-run."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=_COLUMNS, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
        )
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def append(self, ex: AugmentedRow, results: list[ValidationResult]) -> None:
        """Write one augmented row and flush it to disk."""
        self._writer.writerow(_row_record(ex, results))
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._fh.close()

    def __enter__(self) -> "VariantCsvAppender":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def write_sidecar(path: Path, sidecar: AugmentSidecar) -> None:
    """Write the Stage 2 sidecar JSON (`{variant}.meta.json`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
