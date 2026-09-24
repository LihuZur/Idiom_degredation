"""Unit tests for the RQ1 accepted-only filter in scripts/variant_confusion.py.

Pure-function tests only (`aligned_correctness` filtering, `load_accepted_ids`)
— the full CLI pipeline has no existing test coverage to extend.
"""

import csv
from pathlib import Path
from typing import Any

import click
import pytest

from scripts.variant_confusion import aligned_correctness, load_accepted_ids

_FULL_ENTRY: dict[str, Any] = {
    "original": {"correct": True},
    "paraphrase": {"correct": False},
    "idiomatic": {"correct": True},
}


def _entry(task_id: str) -> dict[str, Any]:
    return {"id": task_id, **_FULL_ENTRY}


def test_aligned_correctness_no_filter_keeps_all() -> None:
    per_task = [_entry("a"), _entry("b")]
    assert set(aligned_correctness(per_task)) == {"a", "b"}


def test_aligned_correctness_filters_by_accepted_ids() -> None:
    per_task = [_entry("a"), _entry("b"), _entry("c")]
    aligned = aligned_correctness(per_task, accepted_ids=frozenset({"a", "c"}))
    assert set(aligned) == {"a", "c"}
    assert aligned["a"] == (True, False, True)


def test_aligned_correctness_still_skips_missing_variants() -> None:
    per_task = [{"id": "a", "original": {"correct": True}}]  # missing paraphrase/idiomatic
    assert aligned_correctness(per_task, accepted_ids=frozenset({"a"})) == {}


def _write_accept_csv(path: Path, rows: list[tuple[str, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "accepted_pair"])
        writer.writeheader()
        for row_id, accepted in rows:
            writer.writerow({"id": row_id, "accepted_pair": str(accepted)})


def test_load_accepted_ids_reads_true_rows(tmp_path: Path) -> None:
    path = tmp_path / "meaning_accept.csv"
    _write_accept_csv(path, [("a", True), ("b", False), ("c", True)])
    assert load_accepted_ids(path) == frozenset({"a", "c"})


def test_load_accepted_ids_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="No accept-rule file found"):
        load_accepted_ids(tmp_path / "nope.csv")
