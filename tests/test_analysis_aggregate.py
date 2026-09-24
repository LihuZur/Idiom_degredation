"""Tests for analysis/aggregate.py: the one-row-per-(dataset, model) summary table."""

import csv
from pathlib import Path

import pytest

from analysis.aggregate import aggregate
from analysis.results import filter_to_accepted, load_accepted_ids, load_result
from tests._stage4_helpers import write_result_file

_COLUMNS = [
    "dataset",
    "model_id",
    "model_revision",
    "config_hash",
    "prompt_hash",
    "n",
    "acc_original",
    "acc_paraphrase",
    "acc_idiomatic",
    "delta_paraphrase",
    "delta_paraphrase_ci_low",
    "delta_paraphrase_ci_high",
    "delta_paraphrase_p",
    "delta_idiom",
    "delta_idiom_ci_low",
    "delta_idiom_ci_high",
    "delta_idiom_p",
]

_N_RESAMPLES = 200
_CI = 0.95
_SEED = 0


def _build_results(results_dir: Path) -> None:
    # sst2 / modelA: original 2/4, paraphrase 3/4, idiomatic 1/4
    write_result_file(
        results_dir,
        dataset="sst2",
        model_id="modelA",
        correct_by_variant={
            "original": [True, True, False, False],
            "paraphrase": [True, True, True, False],
            "idiomatic": [True, False, False, False],
        },
    )
    # sst2 / modelB: all variants perfect, delta 0
    write_result_file(
        results_dir,
        dataset="sst2",
        model_id="modelB",
        correct_by_variant={
            "original": [True, True, True, True],
            "paraphrase": [True, True, True, True],
            "idiomatic": [True, True, True, True],
        },
    )
    # mmlu / modelA
    write_result_file(
        results_dir,
        dataset="mmlu",
        model_id="modelA",
        correct_by_variant={
            "original": [True, False, True, False],
            "paraphrase": [True, True, True, False],
            "idiomatic": [False, False, False, False],
        },
    )
    # mmlu / modelB
    write_result_file(
        results_dir,
        dataset="mmlu",
        model_id="modelB",
        correct_by_variant={
            "original": [False, False, False, False],
            "paraphrase": [True, True, False, False],
            "idiomatic": [True, True, True, True],
        },
    )


def test_aggregate_columns_and_shape(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_results(results_dir)

    df = aggregate(results_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)

    assert list(df.columns) == _COLUMNS
    assert df.shape == (4, len(_COLUMNS))

    pairs = list(zip(df["dataset"], df["model_id"], strict=True))
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == {
        ("sst2", "modelA"),
        ("sst2", "modelB"),
        ("mmlu", "modelA"),
        ("mmlu", "modelB"),
    }


def test_aggregate_computed_values_match_hand_computation(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_results(results_dir)

    df = aggregate(results_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)

    row = df[(df["dataset"] == "sst2") & (df["model_id"] == "modelA")].iloc[0]
    assert row["n"] == 4
    assert row["acc_original"] == pytest.approx(0.5)
    assert row["acc_paraphrase"] == pytest.approx(0.75)
    assert row["acc_idiomatic"] == pytest.approx(0.25)
    assert row["delta_paraphrase"] == pytest.approx(0.25)
    assert row["delta_idiom"] == pytest.approx(-0.5)

    identity_row = df[(df["dataset"] == "sst2") & (df["model_id"] == "modelB")].iloc[0]
    assert identity_row["delta_paraphrase"] == pytest.approx(0.0)
    assert identity_row["delta_idiom"] == pytest.approx(0.0)


def _write_meaning_accept_csv(path: Path, accepted_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "accepted_pair"])
        writer.writeheader()
        for task_id in ("task-0", "task-1", "task-2", "task-3"):
            writer.writerow({"id": task_id, "accepted_pair": str(task_id in accepted_ids)})


def test_aggregate_accepted_only_restricts_to_accepted_ids(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    datasets_dir = tmp_path / "datasets_out"
    _build_results(results_dir)
    # sst2/modelA: original [T,T,F,F], paraphrase [T,T,T,F], idiomatic [T,F,F,F].
    # Keep only task-0 and task-2 accepted.
    _write_meaning_accept_csv(datasets_dir / "sst2" / "meaning_accept.csv", {"task-0", "task-2"})
    _write_meaning_accept_csv(
        datasets_dir / "mmlu" / "meaning_accept.csv", {"task-0", "task-1", "task-2", "task-3"}
    )

    df = aggregate(
        results_dir,
        n_resamples=_N_RESAMPLES,
        ci=_CI,
        seed=_SEED,
        accepted_only=True,
        datasets_dir=datasets_dir,
    )

    row = df[(df["dataset"] == "sst2") & (df["model_id"] == "modelA")].iloc[0]
    assert row["n"] == 2
    # task-0: orig T, par T, idi T. task-2: orig F, par T, idi F.
    assert row["acc_original"] == pytest.approx(0.5)
    assert row["acc_paraphrase"] == pytest.approx(1.0)
    assert row["acc_idiomatic"] == pytest.approx(0.5)


def test_aggregate_accepted_only_missing_meaning_accept_raises(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    datasets_dir = tmp_path / "datasets_out"
    _build_results(results_dir)

    with pytest.raises(FileNotFoundError, match="No accept-rule file found"):
        aggregate(
            results_dir,
            n_resamples=_N_RESAMPLES,
            ci=_CI,
            seed=_SEED,
            accepted_only=True,
            datasets_dir=datasets_dir,
        )


def test_load_accepted_ids_reads_true_rows(tmp_path: Path) -> None:
    path = tmp_path / "meaning_accept.csv"
    _write_meaning_accept_csv(path, {"task-0", "task-3"})
    assert load_accepted_ids(path) == frozenset({"task-0", "task-3"})


def test_filter_to_accepted_keeps_only_matching_ids(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    write_result_file(
        results_dir,
        dataset="sst2",
        model_id="modelA",
        correct_by_variant={
            "original": [True, True],
            "paraphrase": [True, True],
            "idiomatic": [True, True],
        },
    )
    result = load_result(results_dir / "sst2" / "modelA.json")
    filtered = filter_to_accepted(result, frozenset({"task-0"}))
    assert [t.id for t in filtered.per_task] == ["task-0"]
    assert result.dataset == filtered.dataset
