"""Tests for augmentation/io.py (STAGE2_CONTRACT IO section)."""

import json
from pathlib import Path

import pandas as pd

from augmentation.base import AugmentedRow, ValidationResult
from augmentation.io import (
    AugmentRowCounts,
    AugmentSidecar,
    AugmentToolVersions,
    CacheStats,
    write_sidecar,
    write_variant_csv,
)


def test_write_variant_csv_round_trip_has_8_columns_in_order(tmp_path: Path) -> None:
    ex = AugmentedRow(
        id="1",
        variant="paraphrase",
        x="hello world",
        y=1,
        augmenter_model="identity",
        prompt_hash="abc123",
        meta={"a": 1},
    )
    results = [
        ValidationResult(
            name="semantic_similarity", passed=True, score=1.0, details={"stub": True}
        ),
        ValidationResult(
            name="label_preservation", passed=True, score=None, details={"stub": True}
        ),
    ]
    out_path = tmp_path / "paraphrase.csv"
    write_variant_csv(out_path, [(ex, results)])

    df = pd.read_csv(out_path, keep_default_na=False, dtype={"id": str, "y": str})
    assert list(df.columns) == [
        "id",
        "variant",
        "x",
        "y",
        "meta",
        "augmenter_model",
        "prompt_hash",
        "validators",
    ]

    row = df.iloc[0]
    assert row["id"] == "1"
    assert row["variant"] == "paraphrase"
    assert row["x"] == "hello world"
    assert row["y"] == "1"
    assert json.loads(row["meta"]) == {"a": 1}
    assert row["augmenter_model"] == "identity"
    assert row["prompt_hash"] == "abc123"


def test_validators_column_is_valid_json_with_expected_structure(tmp_path: Path) -> None:
    ex = AugmentedRow(
        id="1",
        variant="idiomatic",
        x="text",
        y=0,
        augmenter_model="identity",
        prompt_hash="ph",
        meta={},
    )
    results = [
        ValidationResult(
            name="semantic_similarity", passed=True, score=1.0, details={"stub": True}
        ),
        ValidationResult(
            name="idiom_presence",
            passed=True,
            score=None,
            details={"stub": True, "note": "no real idiom detection yet"},
        ),
    ]
    out_path = tmp_path / "idiomatic.csv"
    write_variant_csv(out_path, [(ex, results)])

    df = pd.read_csv(out_path, keep_default_na=False, dtype={"id": str})
    validators = json.loads(df.iloc[0]["validators"])
    assert validators == {
        "semantic_similarity": {"passed": True, "score": 1.0, "details": {"stub": True}},
        "idiom_presence": {
            "passed": True,
            "score": None,
            "details": {"stub": True, "note": "no real idiom detection yet"},
        },
    }


def test_y_is_stringified(tmp_path: Path) -> None:
    ex = AugmentedRow(
        id="1", variant="paraphrase", x="x", y=42, augmenter_model="identity", prompt_hash="ph"
    )
    out_path = tmp_path / "paraphrase.csv"
    write_variant_csv(out_path, [(ex, [])])
    df = pd.read_csv(out_path, keep_default_na=False, dtype={"id": str, "y": str})
    assert df.iloc[0]["y"] == "42"


def test_empty_meta_and_validators_serialize_as_empty_json_object(tmp_path: Path) -> None:
    ex = AugmentedRow(
        id="1",
        variant="idiomatic",
        x="x",
        y=0,
        augmenter_model="identity",
        prompt_hash="ph",
        meta={},
    )
    out_path = tmp_path / "idiomatic.csv"
    write_variant_csv(out_path, [(ex, [])])
    df = pd.read_csv(out_path, keep_default_na=False, dtype={"id": str})
    assert df.iloc[0]["meta"] == "{}"
    assert df.iloc[0]["validators"] == "{}"


def test_meta_round_trips(tmp_path: Path) -> None:
    ex = AugmentedRow(
        id="1",
        variant="paraphrase",
        x="x",
        y=0,
        augmenter_model="identity",
        prompt_hash="ph",
        meta={"nested": {"a": 1}, "list": [1, 2, 3]},
    )
    out_path = tmp_path / "paraphrase.csv"
    write_variant_csv(out_path, [(ex, [])])
    df = pd.read_csv(out_path, keep_default_na=False, dtype={"id": str})
    assert json.loads(df.iloc[0]["meta"]) == {"nested": {"a": 1}, "list": [1, 2, 3]}


def test_augment_sidecar_round_trips_through_model_dump_and_validate() -> None:
    sidecar = AugmentSidecar(
        dataset="sst2",
        variant="paraphrase",
        config_path="/tmp/config.yaml",
        config_hash="deadbeefcafebabe",
        resolved_config={"dataset": "sst2"},
        augmenter_model="identity",
        prompt_file="paraphrase_v1.txt",
        prompt_hash="abc123",
        tool_versions=AugmentToolVersions(python="3.12.0", pandas="2.2.0"),
        row_counts=AugmentRowCounts(
            input_rows=2,
            augmented=2,
            validators_passed_by_name={"semantic_similarity": 2},
            validators_failed_by_name={},
            written=2,
        ),
        cache_stats=CacheStats(hits=1, misses=1),
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    dumped = sidecar.model_dump()
    restored = AugmentSidecar.model_validate(dumped)
    assert restored == sidecar


def test_write_sidecar_writes_readable_json_matching_model(tmp_path: Path) -> None:
    sidecar = AugmentSidecar(
        dataset="mmlu",
        variant="idiomatic",
        config_path="/tmp/config.yaml",
        config_hash="cafebabedeadbeef",
        resolved_config={"dataset": "mmlu"},
        augmenter_model="identity",
        prompt_file="idiomatic_v1.txt",
        prompt_hash="def456",
        tool_versions=AugmentToolVersions(python="3.12.0", pandas="2.2.0"),
        row_counts=AugmentRowCounts(
            input_rows=1,
            augmented=1,
            validators_passed_by_name={},
            validators_failed_by_name={},
            written=1,
        ),
        cache_stats=CacheStats(hits=0, misses=1),
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    out_path = tmp_path / "idiomatic.meta.json"
    write_sidecar(out_path, sidecar)
    loaded = json.loads(out_path.read_text())
    assert AugmentSidecar.model_validate(loaded) == sidecar
