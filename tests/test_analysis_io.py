"""Tests for analysis/io.py: atomic writers and provenance metadata."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from analysis.io import build_provenance, write_markdown, write_sidecar, write_table
from analysis.results import load_result
from tests._stage4_helpers import write_result_file


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["sst2", "mmlu"],
            "model_id": ["modelA", "modelB"],
            "acc_original": [0.5, 0.75],
        }
    )


def test_write_table_round_trips_and_is_atomic(tmp_path: Path) -> None:
    df = _toy_df()
    path = tmp_path / "table.csv"

    write_table(df, path)

    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
    reloaded = pd.read_csv(path)
    assert reloaded.equals(df)


def test_write_markdown_contains_pipe_table_and_columns(tmp_path: Path) -> None:
    df = _toy_df()
    path = tmp_path / "table.md"

    write_markdown(df, path)

    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert "|" in text
    # A Markdown table header-separator row: "|---|---|..." (possibly with
    # alignment colons), on the second non-empty line.
    assert any(set(line.replace("|", "").strip()) <= set("-: ") for line in lines if line.strip())
    for column in df.columns:
        assert column in text


def test_write_sidecar_round_trips_and_keys_are_sorted(tmp_path: Path) -> None:
    meta = {"zeta": 1, "alpha": {"nested_b": 2, "nested_a": 1}, "beta": [1, 2, 3]}
    path = tmp_path / "sidecar.json"

    write_sidecar(meta, path)

    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == meta

    # Top-level and nested keys appear in sorted order in the raw file text.
    raw_text = path.read_text(encoding="utf-8")
    assert list(json.loads(raw_text).keys()) == sorted(meta.keys())
    top_level_positions = [raw_text.index(f'"{k}"') for k in sorted(meta.keys())]
    assert top_level_positions == sorted(top_level_positions)


def test_build_provenance_shape(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    path_a = write_result_file(
        results_dir,
        dataset="sst2",
        model_id="modelA",
        correct_by_variant={
            "original": [True, False],
            "paraphrase": [True, True],
            "idiomatic": [False, True],
        },
        model_revision="rev-a",
        config_hash="cfg-a",
        prompt_hash="ph-a",
    )
    path_b = write_result_file(
        results_dir,
        dataset="mmlu",
        model_id="modelB",
        correct_by_variant={
            "original": [True, True],
            "paraphrase": [False, True],
            "idiomatic": [True, False],
        },
        model_revision="rev-b",
        config_hash="cfg-b",
        prompt_hash="ph-b",
    )
    results = [load_result(path_a), load_result(path_b)]

    provenance = build_provenance(results, n_resamples=123, ci=0.9, seed=7)

    assert set(provenance["tool_versions"].keys()) == {"python", "pandas", "numpy", "plotly"}
    for value in provenance["tool_versions"].values():
        assert isinstance(value, str) and value

    # Should be parseable as an ISO-8601 UTC timestamp.
    ts = provenance["timestamp_utc"]
    datetime.fromisoformat(ts.replace("Z", "+00:00"))

    assert provenance["bootstrap"] == {"n_resamples": 123, "ci": 0.9, "seed": 7}

    # Sources are keyed by "{dataset}/{model_id}" (asserted below)


def test_build_provenance_records_every_dataset_for_shared_model(tmp_path: Path) -> None:
    """Same model id across two datasets must yield two source entries, not one."""
    results_dir = tmp_path / "results"
    triple = {"original": [True], "paraphrase": [True], "idiomatic": [True]}
    paths = [
        write_result_file(results_dir, dataset="sst2", model_id="shared", correct_by_variant=triple),
        write_result_file(results_dir, dataset="mmlu", model_id="shared", correct_by_variant=triple),
    ]
    provenance = build_provenance(
        [load_result(p) for p in paths], n_resamples=10, ci=0.95, seed=0
    )
    assert set(provenance["sources"].keys()) == {"sst2/shared", "mmlu/shared"}


def test_build_provenance_source_keys(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    path_a = write_result_file(
        results_dir,
        dataset="sst2",
        model_id="modelA",
        correct_by_variant={
            "original": [True, False],
            "paraphrase": [True, True],
            "idiomatic": [False, True],
        },
        model_revision="rev-a",
        config_hash="cfg-a",
        prompt_hash="ph-a",
    )
    path_b = write_result_file(
        results_dir,
        dataset="mmlu",
        model_id="modelB",
        correct_by_variant={
            "original": [True, True],
            "paraphrase": [False, True],
            "idiomatic": [True, False],
        },
        model_revision="rev-b",
        config_hash="cfg-b",
        prompt_hash="ph-b",
    )
    provenance = build_provenance(
        [load_result(path_a), load_result(path_b)], n_resamples=10, ci=0.95, seed=0
    )

    # Sources are keyed by "{dataset}/{model_id}" so runs sharing a model id
    # across datasets are each recorded rather than colliding.
    assert provenance["sources"] == {
        "sst2/modelA": {
            "config_hash": "cfg-a",
            "model_revision": "rev-a",
            "prompt_hash": "ph-a",
            "dataset": "sst2",
            "model_id": "modelA",
        },
        "mmlu/modelB": {
            "config_hash": "cfg-b",
            "model_revision": "rev-b",
            "prompt_hash": "ph-b",
            "dataset": "mmlu",
            "model_id": "modelB",
        },
    }
