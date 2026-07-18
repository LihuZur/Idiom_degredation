"""Tests for the complete-triple invariant: any missing variant must raise loudly."""

from pathlib import Path

import pytest

from analysis.aggregate import aggregate, bootstrap_by_run
from analysis.results import load_result
from tests._stage4_helpers import write_result_file

_COMPLETE = {
    "original": [True, False],
    "paraphrase": [True, True],
    "idiomatic": [False, True],
}

_N_RESAMPLES = 200
_CI = 0.95
_SEED = 0


def test_load_result_raises_when_variants_run_missing_a_variant(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    path = write_result_file(
        results_dir,
        dataset="sst2",
        model_id="modelA",
        correct_by_variant=_COMPLETE,
        variants_run=["original", "paraphrase"],  # missing "idiomatic"
    )
    with pytest.raises(ValueError, match="variants_run"):
        load_result(path)


def test_load_result_raises_when_per_task_missing_a_variant_key(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    path = write_result_file(
        results_dir,
        dataset="sst2",
        model_id="modelA",
        # per_task entries omit "idiomatic" entirely...
        correct_by_variant={"original": [True, False], "paraphrase": [True, True]},
        # ...even though variants_run (falsely) claims all three ran.
        variants_run=["original", "paraphrase", "idiomatic"],
    )
    with pytest.raises(ValueError, match="missing"):
        load_result(path)


def test_bootstrap_by_run_raises_loudly_on_incomplete_file_in_dir(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    write_result_file(
        results_dir, dataset="sst2", model_id="good_model", correct_by_variant=_COMPLETE
    )
    write_result_file(
        results_dir,
        dataset="sst2",
        model_id="bad_model",
        correct_by_variant=_COMPLETE,
        variants_run=["original", "paraphrase"],
    )

    with pytest.raises(ValueError):
        bootstrap_by_run(results_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)


def test_aggregate_raises_loudly_on_incomplete_file_in_dir(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    write_result_file(
        results_dir, dataset="sst2", model_id="good_model", correct_by_variant=_COMPLETE
    )
    write_result_file(
        results_dir,
        dataset="mmlu",
        model_id="bad_model",
        correct_by_variant={"original": [True], "paraphrase": [True]},
        variants_run=["original", "paraphrase", "idiomatic"],
    )

    with pytest.raises(ValueError):
        aggregate(results_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)
