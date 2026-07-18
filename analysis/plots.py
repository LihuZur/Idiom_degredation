"""Plotly figures for Stage 4 (Visualize): cross-model and cross-dataset views.

Two entry points (ARCHITECTURE §2.7):

- `plot_dataset_cross_model`: for one dataset, a grouped bar chart of
  per-variant accuracy across every model that has results for it.
- `plot_cross_dataset_summary`: across all datasets, a grouped bar chart of
  the paraphrase/idiom accuracy deltas by model.

Each figure ships with a companion table (CSV + Markdown + JSON provenance
sidecar) built from the same `RunStats` used to draw it, so the numbers on
disk always match the numbers in the plot. Figures are written as
standalone HTML only — no PNG/kaleido anywhere in this module.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from analysis.aggregate import RunStats, bootstrap_by_run
from analysis.io import build_provenance, write_markdown, write_sidecar, write_table
from analysis.results import VARIANTS

_DATASET_TABLE_COLUMNS = [
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
    "acc_original_ci_low",
    "acc_original_ci_high",
    "acc_paraphrase_ci_low",
    "acc_paraphrase_ci_high",
    "acc_idiomatic_ci_low",
    "acc_idiomatic_ci_high",
]

_SUMMARY_TABLE_COLUMNS = [
    "dataset",
    "model_id",
    "delta_paraphrase",
    "delta_paraphrase_ci_low",
    "delta_paraphrase_ci_high",
    "delta_paraphrase_p",
    "delta_idiom",
    "delta_idiom_ci_low",
    "delta_idiom_ci_high",
    "delta_idiom_p",
]

# Plotly's default qualitative palette — used to keep one stable color per
# dataset in the cross-dataset summary figure without pulling in
# plotly.express just for a color cycle.
_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def _error_y(points: list[float], ci_lows: list[float], ci_highs: list[float]) -> dict[str, object]:
    """Build a Plotly `error_y` spec of offsets from each point's own CI (D12)."""
    return {
        "type": "data",
        "symmetric": False,
        "array": [high - point for high, point in zip(ci_highs, points, strict=True)],
        "arrayminus": [point - low for point, low in zip(points, ci_lows, strict=True)],
    }


def _dataset_table(dataset_runs: list[RunStats]) -> pd.DataFrame:
    """Build the companion table for `plot_dataset_cross_model` from `RunStats`."""
    rows: list[dict[str, str | int | float]] = []
    for run in dataset_runs:
        result = run.result
        boot = run.boot
        rows.append(
            {
                "dataset": result.dataset,
                "model_id": result.model_id,
                "model_revision": result.model_revision,
                "config_hash": result.config_hash,
                "prompt_hash": result.prompt_hash,
                "n": boot.n,
                "acc_original": boot.acc["original"].point,
                "acc_paraphrase": boot.acc["paraphrase"].point,
                "acc_idiomatic": boot.acc["idiomatic"].point,
                "delta_paraphrase": boot.delta_paraphrase.point,
                "delta_paraphrase_ci_low": boot.delta_paraphrase.ci_low,
                "delta_paraphrase_ci_high": boot.delta_paraphrase.ci_high,
                "delta_paraphrase_p": boot.delta_paraphrase.p_value,
                "delta_idiom": boot.delta_idiom.point,
                "delta_idiom_ci_low": boot.delta_idiom.ci_low,
                "delta_idiom_ci_high": boot.delta_idiom.ci_high,
                "delta_idiom_p": boot.delta_idiom.p_value,
                "acc_original_ci_low": boot.acc["original"].ci_low,
                "acc_original_ci_high": boot.acc["original"].ci_high,
                "acc_paraphrase_ci_low": boot.acc["paraphrase"].ci_low,
                "acc_paraphrase_ci_high": boot.acc["paraphrase"].ci_high,
                "acc_idiomatic_ci_low": boot.acc["idiomatic"].ci_low,
                "acc_idiomatic_ci_high": boot.acc["idiomatic"].ci_high,
            }
        )
    return pd.DataFrame(rows, columns=_DATASET_TABLE_COLUMNS)


def _summary_table(runs: list[RunStats]) -> pd.DataFrame:
    """Build the companion (model x dataset) delta table for `plot_cross_dataset_summary`."""
    rows: list[dict[str, str | float]] = []
    for run in runs:
        result = run.result
        boot = run.boot
        rows.append(
            {
                "dataset": result.dataset,
                "model_id": result.model_id,
                "delta_paraphrase": boot.delta_paraphrase.point,
                "delta_paraphrase_ci_low": boot.delta_paraphrase.ci_low,
                "delta_paraphrase_ci_high": boot.delta_paraphrase.ci_high,
                "delta_paraphrase_p": boot.delta_paraphrase.p_value,
                "delta_idiom": boot.delta_idiom.point,
                "delta_idiom_ci_low": boot.delta_idiom.ci_low,
                "delta_idiom_ci_high": boot.delta_idiom.ci_high,
                "delta_idiom_p": boot.delta_idiom.p_value,
            }
        )
    return pd.DataFrame(rows, columns=_SUMMARY_TABLE_COLUMNS)


def plot_dataset_cross_model(
    dataset: str,
    results_dir: Path,
    *,
    out_dir: Path,
    n_resamples: int,
    ci: float,
    seed: int,
) -> Path:
    """Build the per-dataset, cross-model accuracy-by-variant figure and table.

    Draws one grouped bar chart per model_id x variant (original / paraphrase
    / idiomatic), with each bar's error bar taken from that bar's own
    bootstrap percentile CI, and writes a companion table with the same
    numbers (CSV + Markdown + provenance sidecar).

    Args:
        dataset: The dataset name to filter results to (matches
            `ResultFile.dataset`).
        results_dir: Root directory containing ``{dataset}/{model}.json`` files.
        out_dir: Root output directory; figures go to `out_dir/figures`,
            tables to `out_dir/tables`.
        n_resamples: Number of bootstrap resamples to draw per run.
        ci: Confidence level in (0, 1) for the bootstrap intervals.
        seed: RNG seed for bootstrap resampling (shared across runs).

    Returns:
        The path to the written HTML figure.

    Raises:
        ValueError: If no result files are found for `dataset`.
    """
    runs = bootstrap_by_run(results_dir, n_resamples=n_resamples, ci=ci, seed=seed)
    dataset_runs = [run for run in runs if run.result.dataset == dataset]
    if not dataset_runs:
        raise ValueError(f"No result files found for dataset {dataset!r} under {results_dir}")

    model_ids = [run.result.model_id for run in dataset_runs]

    fig = go.Figure()
    for variant in VARIANTS:
        points = [run.boot.acc[variant].point for run in dataset_runs]
        ci_lows = [run.boot.acc[variant].ci_low for run in dataset_runs]
        ci_highs = [run.boot.acc[variant].ci_high for run in dataset_runs]
        customdata = [
            (
                run.result.model_revision,
                run.result.config_hash,
                run.boot.delta_paraphrase.point,
                run.boot.delta_paraphrase.ci_low,
                run.boot.delta_paraphrase.ci_high,
                run.boot.delta_paraphrase.p_value,
                run.boot.delta_idiom.point,
                run.boot.delta_idiom.ci_low,
                run.boot.delta_idiom.ci_high,
                run.boot.delta_idiom.p_value,
            )
            for run in dataset_runs
        ]
        fig.add_trace(
            go.Bar(
                name=variant,
                x=model_ids,
                y=points,
                error_y=_error_y(points, ci_lows, ci_highs),
                customdata=customdata,
                hovertemplate=(
                    "model=%{x}<br>"
                    f"variant={variant}<br>"
                    "accuracy=%{y:.3f}<br>"
                    "model_revision=%{customdata[0]}<br>"
                    "config_hash=%{customdata[1]}<br>"
                    "delta_paraphrase=%{customdata[2]:.3f} "
                    "CI=[%{customdata[3]:.3f}, %{customdata[4]:.3f}] "
                    "p=%{customdata[5]:.3f}<br>"
                    "delta_idiom=%{customdata[6]:.3f} "
                    "CI=[%{customdata[7]:.3f}, %{customdata[8]:.3f}] "
                    "p=%{customdata[9]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    for run in dataset_runs:
        fig.add_annotation(
            x=run.result.model_id,
            xref="x",
            y=1.0,
            yref="paper",
            yshift=20,
            text=(
                f"Δpara={run.boot.delta_paraphrase.point:+.3f} "
                f"(p={run.boot.delta_paraphrase.p_value:.3f})<br>"
                f"Δidiom={run.boot.delta_idiom.point:+.3f} "
                f"(p={run.boot.delta_idiom.p_value:.3f})"
            ),
            showarrow=False,
            font={"size": 10},
        )

    fig.update_layout(
        title=f"{dataset}: accuracy by model across variants",
        yaxis_title="accuracy",
        yaxis_range=[0, 1],
        xaxis_title="model",
        barmode="group",
        margin={"t": 100},
    )

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{dataset}_cross_model.html"
    fig.write_html(html_path)

    tables_dir = out_dir / "tables"
    table = _dataset_table(dataset_runs)
    write_table(table, tables_dir / f"{dataset}_cross_model.csv")
    write_markdown(table, tables_dir / f"{dataset}_cross_model.md")
    provenance = build_provenance(
        [run.result for run in dataset_runs], n_resamples=n_resamples, ci=ci, seed=seed
    )
    write_sidecar(provenance, tables_dir / f"{dataset}_cross_model.meta.json")

    return html_path


def plot_cross_dataset_summary(
    results_dir: Path,
    *,
    out_dir: Path,
    n_resamples: int,
    ci: float,
    seed: int,
) -> Path:
    """Build the cross-dataset paraphrase/idiom delta summary figure and table.

    Draws one grouped bar chart of Δ_paraphrase and Δ_idiom by model, with
    bars colored by dataset and patterned by delta kind, and writes a
    companion (model x dataset) delta table (CSV + Markdown + provenance
    sidecar).

    Args:
        results_dir: Root directory containing ``{dataset}/{model}.json`` files.
        out_dir: Root output directory; figures go to `out_dir/figures`,
            tables to `out_dir/tables`.
        n_resamples: Number of bootstrap resamples to draw per run.
        ci: Confidence level in (0, 1) for the bootstrap intervals.
        seed: RNG seed for bootstrap resampling (shared across runs).

    Returns:
        The path to the written HTML figure.

    Raises:
        ValueError: If no result files are found under `results_dir`.
    """
    runs = bootstrap_by_run(results_dir, n_resamples=n_resamples, ci=ci, seed=seed)
    if not runs:
        raise ValueError(f"No result files found under {results_dir}")

    datasets = sorted({run.result.dataset for run in runs})
    color_by_dataset = {d: _PALETTE[i % len(_PALETTE)] for i, d in enumerate(datasets)}

    fig = go.Figure()
    for dataset in datasets:
        dataset_runs = [run for run in runs if run.result.dataset == dataset]
        model_ids = [run.result.model_id for run in dataset_runs]
        for delta_kind, pattern_shape in (("paraphrase", ""), ("idiom", "/")):
            deltas = [
                run.boot.delta_paraphrase if delta_kind == "paraphrase" else run.boot.delta_idiom
                for run in dataset_runs
            ]
            points = [delta.point for delta in deltas]
            ci_lows = [delta.ci_low for delta in deltas]
            ci_highs = [delta.ci_high for delta in deltas]
            customdata = [
                (run.result.config_hash, delta.p_value, delta.ci_low, delta.ci_high)
                for run, delta in zip(dataset_runs, deltas, strict=True)
            ]
            fig.add_trace(
                go.Bar(
                    name=f"{dataset} · Δ{delta_kind}",
                    legendgroup=dataset,
                    x=model_ids,
                    y=points,
                    marker={
                        "color": color_by_dataset[dataset],
                        "pattern": {"shape": pattern_shape},
                    },
                    error_y=_error_y(points, ci_lows, ci_highs),
                    customdata=customdata,
                    hovertemplate=(
                        "model=%{x}<br>"
                        f"dataset={dataset}<br>"
                        f"delta={delta_kind}<br>"
                        "value=%{y:.3f}<br>"
                        "CI=[%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>"
                        "p=%{customdata[1]:.3f}<br>"
                        "config_hash=%{customdata[0]}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="Cross-dataset paraphrase/idiom deltas by model",
        yaxis_title="delta (accuracy)",
        xaxis_title="model",
        barmode="group",
    )

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / "cross_dataset_summary.html"
    fig.write_html(html_path)

    tables_dir = out_dir / "tables"
    table = _summary_table(runs)
    write_table(table, tables_dir / "cross_dataset_summary.csv")
    write_markdown(table, tables_dir / "cross_dataset_summary.md")
    provenance = build_provenance(
        [run.result for run in runs], n_resamples=n_resamples, ci=ci, seed=seed
    )
    write_sidecar(provenance, tables_dir / "cross_dataset_summary.meta.json")

    return html_path
