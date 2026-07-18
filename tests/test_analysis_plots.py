"""Tests for analysis/plots.py: HTML-only Plotly figures + companion tables (D9)."""

from pathlib import Path

import pytest

from analysis.plots import plot_cross_dataset_summary, plot_dataset_cross_model
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
