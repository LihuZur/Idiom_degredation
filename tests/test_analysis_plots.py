"""Tests for analysis/plots.py: HTML-only Plotly figures + companion tables (D9)."""

from pathlib import Path

import pandas as pd
import pytest

from analysis.aggregate import bootstrap_by_run
from analysis.plots import (
    _has_resolved_delta,
    plot_cross_dataset_summary,
    plot_dataset_cross_model,
)
from tests._stage4_helpers import write_result_file

_N_RESAMPLES = 200
_CI = 0.95
_SEED = 0


def _build_results(results_dir: Path) -> None:
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


def _build_mixed_results(results_dir: Path) -> None:
    """Two models on one dataset: one with a resolved delta, one without.

    `degraded` is correct on every original example and wrong on every
    idiomatic one, so Δ_idiom is pinned at -1.0 and its bootstrap CI cannot
    reach zero. `flat` is identical across variants, so both its deltas are
    exactly 0.0 with a degenerate CI that does not clear zero. That makes the
    `_significant` cut exactly {degraded}.
    """
    n = 40
    write_result_file(
        results_dir,
        dataset="sst2",
        model_id="degraded",
        correct_by_variant={
            "original": [True] * n,
            "paraphrase": [True] * n,
            "idiomatic": [False] * n,
        },
    )
    write_result_file(
        results_dir,
        dataset="sst2",
        model_id="flat",
        correct_by_variant={
            "original": [True, False] * (n // 2),
            "paraphrase": [True, False] * (n // 2),
            "idiomatic": [True, False] * (n // 2),
        },
    )


def _assert_no_png_anywhere(root: Path) -> None:
    assert list(root.rglob("*.png")) == []


def test_plot_dataset_cross_model_writes_html_and_tables_no_png(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_results(results_dir)
    out_dir = tmp_path / "out"

    html_path = plot_dataset_cross_model(
        "sst2", results_dir, out_dir=out_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )

    assert html_path.name == "sst2_cross_model.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "plotly" in content

    tables_dir = out_dir / "tables"
    assert (tables_dir / "sst2_cross_model.csv").exists()
    assert (tables_dir / "sst2_cross_model.md").exists()
    assert (tables_dir / "sst2_cross_model.meta.json").exists()

    _assert_no_png_anywhere(out_dir)


def test_plot_dataset_cross_model_raises_for_unknown_dataset(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_results(results_dir)
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="No result files"):
        plot_dataset_cross_model(
            "does_not_exist",
            results_dir,
            out_dir=out_dir,
            n_resamples=_N_RESAMPLES,
            ci=_CI,
            seed=_SEED,
        )


def test_plot_cross_dataset_summary_writes_html_and_tables_no_png(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_results(results_dir)
    out_dir = tmp_path / "out"

    html_path = plot_cross_dataset_summary(
        results_dir, out_dir=out_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )

    assert html_path.name == "cross_dataset_summary.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "plotly" in content

    tables_dir = out_dir / "tables"
    assert (tables_dir / "cross_dataset_summary.csv").exists()
    assert (tables_dir / "cross_dataset_summary.md").exists()
    assert (tables_dir / "cross_dataset_summary.meta.json").exists()

    _assert_no_png_anywhere(out_dir)


def test_has_resolved_delta_matches_ci_crossing_zero(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_mixed_results(results_dir)

    runs = {
        run.result.model_id: run
        for run in bootstrap_by_run(results_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)
    }

    # Δ_idiom is pinned at -1.0, so its whole interval sits below zero.
    degraded = runs["degraded"].boot.delta_idiom
    assert degraded.ci_high < 0.0
    assert _has_resolved_delta(runs["degraded"])

    # Both deltas are exactly zero, so neither interval clears the axis.
    assert runs["flat"].boot.delta_idiom.ci_low == runs["flat"].boot.delta_idiom.ci_high == 0.0
    assert not _has_resolved_delta(runs["flat"])


def test_dataset_significant_cut_keeps_only_resolved_models(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_mixed_results(results_dir)
    out_dir = tmp_path / "out"

    plot_dataset_cross_model(
        "sst2", results_dir, out_dir=out_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )

    assert (out_dir / "figures" / "sst2_cross_model_significant.html").exists()

    tables_dir = out_dir / "tables"
    full = pd.read_csv(tables_dir / "sst2_cross_model.csv")
    significant = pd.read_csv(tables_dir / "sst2_cross_model_significant.csv")

    assert sorted(full["model_id"]) == ["degraded", "flat"]
    assert list(significant["model_id"]) == ["degraded"]
    assert (tables_dir / "sst2_cross_model_significant.md").exists()
    assert (tables_dir / "sst2_cross_model_significant.meta.json").exists()

    _assert_no_png_anywhere(out_dir)


def test_summary_significant_cut_keeps_only_resolved_runs(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _build_mixed_results(results_dir)
    out_dir = tmp_path / "out"

    plot_cross_dataset_summary(
        results_dir, out_dir=out_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )

    assert (out_dir / "figures" / "cross_dataset_summary_significant.html").exists()

    tables_dir = out_dir / "tables"
    significant = pd.read_csv(tables_dir / "cross_dataset_summary_significant.csv")
    assert list(significant["model_id"]) == ["degraded"]

    # Every surviving row must really have an interval clear of zero.
    for _, row in significant.iterrows():
        assert (
            row["delta_paraphrase_ci_low"] > 0
            or row["delta_paraphrase_ci_high"] < 0
            or row["delta_idiom_ci_low"] > 0
            or row["delta_idiom_ci_high"] < 0
        )

    _assert_no_png_anywhere(out_dir)


def test_no_significant_files_when_nothing_resolves(tmp_path: Path) -> None:
    """A dataset where every CI straddles zero writes no `_significant` artifacts."""
    results_dir = tmp_path / "results"
    write_result_file(
        results_dir,
        dataset="sst2",
        model_id="flat",
        correct_by_variant={
            "original": [True, False] * 10,
            "paraphrase": [True, False] * 10,
            "idiomatic": [True, False] * 10,
        },
    )
    out_dir = tmp_path / "out"

    plot_dataset_cross_model(
        "sst2", results_dir, out_dir=out_dir, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )

    assert (out_dir / "figures" / "sst2_cross_model.html").exists()
    assert not (out_dir / "figures" / "sst2_cross_model_significant.html").exists()
    assert not (out_dir / "tables" / "sst2_cross_model_significant.csv").exists()
