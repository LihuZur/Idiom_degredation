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
from plotly.subplots import make_subplots

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

# One stable hue per variant, so identity survives across both panels of the
# per-dataset figure: the Δ_paraphrase bar wears the same color as the
# `paraphrase` accuracy bar it was derived from. Validated as a categorical
# triple (lightness band, chroma floor, CVD separation) against the light
# surface below; the sub-3:1 contrast of the green is discharged by the
# companion CSV/Markdown table that ships with every figure.
_VARIANT_COLORS = {
    "original": "#636EFA",
    "paraphrase": "#EF553B",
    "idiomatic": "#00CC96",
}

# Recessive chart chrome. These figures are written as standalone HTML for a
# light-background report, so the light surface is committed to explicitly
# rather than inherited from whatever template Plotly defaults to.
_SURFACE = "#FCFCFB"
_GRID = "#E7E7E2"
_AXIS_INK = "#575650"
_MUTED_INK = "#76756D"

# Alpha for a delta bar whose CI clears zero vs. one whose CI straddles it, so
# resolved effects read as solid and unresolved ones recede. The same
# distinction is *also* carried by a "*" direct label and by the CI and
# p-value in the hover box, so this is never a color-alone encoding.
_ALPHA_RESOLVED = 0.95
_ALPHA_UNRESOLVED = 0.28

# Vertical room per model row in the per-dataset figure. Three grouped
# horizontal bars per model need enough height that the bars stay thin and the
# tick label clears its neighbours.
_ROW_HEIGHT_PX = 46
_CHROME_HEIGHT_PX = 210

# Anchored to the figure container rather than the plot area: these figures
# grow with the model count, so a paper-relative legend drifts down into the
# subplot titles on short figures. Container coordinates keep it beside the
# main title at every height.
_LEGEND: dict[str, object] = {
    "orientation": "h",
    "yref": "container",
    "yanchor": "top",
    "y": 0.985,
    "xanchor": "right",
    "x": 1,
    "title": {"text": ""},
    "bgcolor": "rgba(0,0,0,0)",
}


def _error_x(points: list[float], ci_lows: list[float], ci_highs: list[float]) -> dict[str, object]:
    """Build a Plotly `error_x` spec of offsets from each point's own CI.

    Offsets rather than absolute bounds, because Plotly's error bars are
    expressed relative to each point.
    """
    return {
        "type": "data",
        "symmetric": False,
        "array": [high - point for high, point in zip(ci_highs, points, strict=True)],
        "arrayminus": [point - low for point, low in zip(points, ci_lows, strict=True)],
        "thickness": 1,
        "width": 3,
        "color": _MUTED_INK,
    }


def _has_resolved_delta(run: RunStats) -> bool:
    """Whether either delta's bootstrap CI lies wholly on one side of zero.

    This is the filter behind every `*_significant` figure/table: a run is kept
    when Δ_paraphrase **or** Δ_idiom has a confidence interval that does not
    cross the zero axis, i.e. the sign of the effect is resolved by the data
    rather than being consistent with "no change". Runs where both intervals
    straddle zero are dropped.
    """
    return any(
        delta.ci_low > 0.0 or delta.ci_high < 0.0
        for delta in (run.boot.delta_paraphrase, run.boot.delta_idiom)
    )


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


def _build_dataset_figure(
    dataset: str,
    dataset_runs: list[RunStats],
    *,
    ci: float,
    significant_only: bool = False,
) -> go.Figure:
    """Draw the two-panel per-dataset figure for `dataset_runs`.

    Left panel is per-variant accuracy, right panel is the same runs' deltas
    against `original`, sharing one model axis so a row reads straight across.
    Bars are horizontal because model ids are long enough that vertical tick
    labels collide well before 17 of them fit.

    Args:
        dataset: Dataset name, used in the title.
        dataset_runs: The runs to draw, one grouped row each.
        ci: Confidence level the error bars represent, for the subtitle.
        significant_only: Whether `dataset_runs` has already been filtered to
            runs with a delta CI clear of zero; only changes the subtitle.

    Returns:
        The assembled Plotly figure.
    """
    # Draw the highest-accuracy model at the top. Plotly stacks categorical y
    # values bottom-up, so ascending order here renders as descending on
    # screen. Only the figure is reordered — the caller's list keeps its load
    # order so the companion table stays row-aligned with `summary.csv`.
    plot_runs = sorted(dataset_runs, key=lambda run: run.boot.acc["original"].point)
    model_ids = [run.result.model_id for run in plot_runs]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.56, 0.44],
        horizontal_spacing=0.06,
        subplot_titles=("Accuracy by variant", "Δ vs. original"),
    )

    # Panel 1 — per-variant accuracy, one grouped horizontal bar per model.
    for variant in VARIANTS:
        points = [run.boot.acc[variant].point for run in plot_runs]
        ci_lows = [run.boot.acc[variant].ci_low for run in plot_runs]
        ci_highs = [run.boot.acc[variant].ci_high for run in plot_runs]
        fig.add_trace(
            go.Bar(
                name=variant,
                legendgroup=variant,
                orientation="h",
                x=points,
                y=model_ids,
                offsetgroup=variant,
                marker={"color": _VARIANT_COLORS[variant], "line": {"width": 0}},
                error_x=_error_x(points, ci_lows, ci_highs),
                customdata=[
                    (run.result.model_revision, run.result.config_hash, run.boot.n)
                    for run in plot_runs
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"variant={variant}<br>"
                    "accuracy=%{x:.3f}<br>"
                    "n=%{customdata[2]}<br>"
                    "model_revision=%{customdata[0]}<br>"
                    "config_hash=%{customdata[1]}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Panel 2 has one fewer bar than panel 1 (there is no Δ of `original`
    # against itself), which would shift its group off the row that panel 1
    # draws. An empty bar holding the `original` offset slot keeps both panels
    # on the same grid, so a model's row reads straight across.
    fig.add_trace(
        go.Bar(
            name="original",
            legendgroup="original",
            showlegend=False,
            orientation="h",
            x=[0.0] * len(model_ids),
            y=model_ids,
            offsetgroup="original",
            marker={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Panel 2 — deltas against `original`. Each delta keeps the hue of the
    # variant it came from, so identity carries across the two panels. A bar
    # whose CI clears zero is drawn solid and marked "*"; one that straddles
    # zero recedes — the reader can see at a glance which effects are resolved.
    for delta_kind, variant in (("paraphrase", "paraphrase"), ("idiom", "idiomatic")):
        deltas = [
            run.boot.delta_paraphrase if delta_kind == "paraphrase" else run.boot.delta_idiom
            for run in plot_runs
        ]
        points = [delta.point for delta in deltas]
        resolved = [delta.ci_low > 0.0 or delta.ci_high < 0.0 for delta in deltas]
        fig.add_trace(
            go.Bar(
                name=f"Δ{delta_kind}",
                legendgroup=variant,
                showlegend=False,
                orientation="h",
                x=points,
                y=model_ids,
                offsetgroup=variant,
                marker={
                    "color": _VARIANT_COLORS[variant],
                    "line": {"width": 0},
                    "opacity": [
                        _ALPHA_RESOLVED if is_resolved else _ALPHA_UNRESOLVED
                        for is_resolved in resolved
                    ],
                },
                error_x=_error_x(
                    points,
                    [delta.ci_low for delta in deltas],
                    [delta.ci_high for delta in deltas],
                ),
                text=["*" if is_resolved else "" for is_resolved in resolved],
                textposition="outside",
                textfont={"size": 14, "color": _AXIS_INK},
                cliponaxis=False,
                customdata=[(delta.ci_low, delta.ci_high, delta.p_value) for delta in deltas],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Δ{delta_kind} vs. original<br>"
                    "delta=%{x:+.3f}<br>"
                    f"CI=[%{{customdata[0]:+.3f}}, %{{customdata[1]:+.3f}}]<br>"
                    "p=%{customdata[2]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    scope = (
        "models with at least one delta CI clear of zero"
        if significant_only
        else "all models with results"
    )
    fig.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"<b>{dataset}: accuracy by model across variants</b>"
                f"<br><span style='font-size:12px;color:{_MUTED_INK}'>"
                f"{len(plot_runs)} {'model' if len(plot_runs) == 1 else 'models'} ({scope})"
                f" · n={plot_runs[0].boot.n} paired examples"
                f" · error bars are {round(ci * 100)}% bootstrap CIs;"
                " “*” = CI does not cross zero</span>"
            ),
            "x": 0,
            "xanchor": "left",
            "font": {"size": 17},
        },
        barmode="group",
        bargap=0.30,
        bargroupgap=0.12,
        height=_CHROME_HEIGHT_PX + _ROW_HEIGHT_PX * len(plot_runs),
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font={"color": _AXIS_INK, "size": 12},
        legend=_LEGEND,
        margin={"t": 135, "l": 195, "r": 45, "b": 60},
    )
    fig.update_xaxes(
        title_text="accuracy", range=[0, 1], gridcolor=_GRID, zeroline=False, row=1, col=1
    )
    # The zero rule is the "no degradation" reference the delta bars are read
    # against, so it is drawn as the axis's own zeroline rather than an
    # overlaid shape.
    fig.update_xaxes(
        title_text="Δ accuracy",
        gridcolor=_GRID,
        zeroline=True,
        zerolinecolor=_AXIS_INK,
        zerolinewidth=1,
        row=1,
        col=2,
    )
    fig.update_yaxes(gridcolor=_GRID, ticksuffix="  ")
    return fig


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

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"

    fig = _build_dataset_figure(dataset, dataset_runs, ci=ci)
    html_path = figures_dir / f"{dataset}_cross_model.html"
    fig.write_html(html_path)

    table = _dataset_table(dataset_runs)
    write_table(table, tables_dir / f"{dataset}_cross_model.csv")
    write_markdown(table, tables_dir / f"{dataset}_cross_model.md")
    provenance = build_provenance(
        [run.result for run in dataset_runs], n_resamples=n_resamples, ci=ci, seed=seed
    )
    write_sidecar(provenance, tables_dir / f"{dataset}_cross_model.meta.json")

    # Companion cut: only the models whose effect is resolved by the data —
    # at least one delta CI clear of zero. Written even when empty is
    # impossible (we skip instead), so a missing `_significant` file means
    # "no model on this dataset showed a resolved effect".
    significant_runs = [run for run in dataset_runs if _has_resolved_delta(run)]
    if significant_runs:
        sig_fig = _build_dataset_figure(dataset, significant_runs, ci=ci, significant_only=True)
        sig_fig.write_html(figures_dir / f"{dataset}_cross_model_significant.html")

        sig_table = _dataset_table(significant_runs)
        write_table(sig_table, tables_dir / f"{dataset}_cross_model_significant.csv")
        write_markdown(sig_table, tables_dir / f"{dataset}_cross_model_significant.md")
        write_sidecar(
            build_provenance(
                [run.result for run in significant_runs],
                n_resamples=n_resamples,
                ci=ci,
                seed=seed,
            ),
            tables_dir / f"{dataset}_cross_model_significant.meta.json",
        )

    return html_path


def _build_summary_figure(
    runs: list[RunStats],
    *,
    ci: float,
    significant_only: bool = False,
) -> go.Figure:
    """Draw the cross-dataset delta figure as small multiples, one facet per dataset.

    Faceting by dataset rather than crowding every dataset into one grouped row
    keeps the bars readable at 17 models, and frees color to carry the delta
    kind — the same hues the per-dataset figure uses for those variants.

    Args:
        runs: The runs to draw, across all datasets.
        ci: Confidence level the error bars represent, for the subtitle.
        significant_only: Whether `runs` has already been filtered to runs with
            a delta CI clear of zero; only changes the subtitle.

    Returns:
        The assembled Plotly figure.
    """
    datasets = sorted({run.result.dataset for run in runs})

    # One shared model order across every facet, so a row reads straight
    # across. Most-degraded (most negative mean Δ_idiom) sorts last, which
    # Plotly's bottom-up categorical axis renders at the top.
    idioms_by_model: dict[str, list[float]] = {}
    for run in runs:
        idioms_by_model.setdefault(run.result.model_id, []).append(run.boot.delta_idiom.point)
    model_order = sorted(
        idioms_by_model,
        key=lambda m: sum(idioms_by_model[m]) / len(idioms_by_model[m]),
        reverse=True,
    )

    fig = make_subplots(
        rows=1,
        cols=len(datasets),
        shared_yaxes=True,
        horizontal_spacing=0.045,
        subplot_titles=tuple(datasets),
    )

    for col, dataset in enumerate(datasets, start=1):
        by_model = {run.result.model_id: run for run in runs if run.result.dataset == dataset}
        for delta_kind, variant in (("paraphrase", "paraphrase"), ("idiom", "idiomatic")):
            models = [m for m in model_order if m in by_model]
            deltas = [
                by_model[m].boot.delta_paraphrase
                if delta_kind == "paraphrase"
                else by_model[m].boot.delta_idiom
                for m in models
            ]
            if not deltas:
                continue
            points = [delta.point for delta in deltas]
            resolved = [delta.ci_low > 0.0 or delta.ci_high < 0.0 for delta in deltas]
            fig.add_trace(
                go.Bar(
                    name=f"Δ{delta_kind}",
                    legendgroup=variant,
                    showlegend=col == 1,
                    orientation="h",
                    x=points,
                    y=models,
                    marker={
                        "color": _VARIANT_COLORS[variant],
                        "line": {"width": 0},
                        "opacity": [
                            _ALPHA_RESOLVED if is_resolved else _ALPHA_UNRESOLVED
                            for is_resolved in resolved
                        ],
                    },
                    error_x=_error_x(
                        points,
                        [delta.ci_low for delta in deltas],
                        [delta.ci_high for delta in deltas],
                    ),
                    customdata=[
                        (by_model[m].result.config_hash, delta.ci_low, delta.ci_high, delta.p_value)
                        for m, delta in zip(models, deltas, strict=True)
                    ],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"dataset={dataset}<br>"
                        f"Δ{delta_kind} vs. original<br>"
                        "delta=%{x:+.3f}<br>"
                        "CI=[%{customdata[1]:+.3f}, %{customdata[2]:+.3f}]<br>"
                        "p=%{customdata[3]:.3f}<br>"
                        "config_hash=%{customdata[0]}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    scope = "runs with at least one delta CI clear of zero" if significant_only else "all runs"
    fig.update_layout(
        template="plotly_white",
        title={
            "text": (
                "<b>Cross-dataset paraphrase/idiom deltas by model</b>"
                f"<br><span style='font-size:12px;color:{_MUTED_INK}'>"
                f"{len(runs)} {'run' if len(runs) == 1 else 'runs'} ({scope}) across "
                f"{len(datasets)} {'dataset' if len(datasets) == 1 else 'datasets'} · "
                f"error bars are {round(ci * 100)}% bootstrap CIs; "
                "faded bars have a CI crossing zero</span>"
            ),
            "x": 0,
            "xanchor": "left",
            "font": {"size": 17},
        },
        barmode="group",
        bargap=0.30,
        bargroupgap=0.12,
        height=_CHROME_HEIGHT_PX + _ROW_HEIGHT_PX * len(model_order),
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font={"color": _AXIS_INK, "size": 12},
        legend=_LEGEND,
        margin={"t": 135, "l": 195, "r": 45, "b": 60},
    )
    # The zero rule is the "no degradation" reference the delta bars are read
    # against, so it is drawn as the axis's own zeroline rather than an
    # overlaid shape.
    # One x-range for every facet. Small multiples are only comparable if a
    # given bar length means the same delta in each panel; per-facet
    # autoscaling would silently make a small mmlu effect look like a large
    # sst2 one. Padded so the widest CI whisker still lands inside the axis.
    bounds = [
        bound
        for run in runs
        for delta in (run.boot.delta_paraphrase, run.boot.delta_idiom)
        for bound in (delta.ci_low, delta.ci_high)
    ]
    span = max(max(bounds) - min(bounds), 1e-6)
    fig.update_xaxes(
        title_text="Δ accuracy",
        range=[min(bounds) - 0.06 * span, max(bounds) + 0.06 * span],
        gridcolor=_GRID,
        zeroline=True,
        zerolinecolor=_AXIS_INK,
        zerolinewidth=1,
    )
    fig.update_yaxes(gridcolor=_GRID, ticksuffix="  ")
    return fig


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

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"

    fig = _build_summary_figure(runs, ci=ci)
    html_path = figures_dir / "cross_dataset_summary.html"
    fig.write_html(html_path)

    table = _summary_table(runs)
    write_table(table, tables_dir / "cross_dataset_summary.csv")
    write_markdown(table, tables_dir / "cross_dataset_summary.md")
    provenance = build_provenance(
        [run.result for run in runs], n_resamples=n_resamples, ci=ci, seed=seed
    )
    write_sidecar(provenance, tables_dir / "cross_dataset_summary.meta.json")

    # Companion cut: only the (model, dataset) runs whose effect is resolved —
    # at least one delta CI clear of zero. A model that resolves on one dataset
    # but not another keeps its row and simply has no bar in the other facet.
    significant_runs = [run for run in runs if _has_resolved_delta(run)]
    if significant_runs:
        sig_fig = _build_summary_figure(significant_runs, ci=ci, significant_only=True)
        sig_fig.write_html(figures_dir / "cross_dataset_summary_significant.html")

        sig_table = _summary_table(significant_runs)
        write_table(sig_table, tables_dir / "cross_dataset_summary_significant.csv")
        write_markdown(sig_table, tables_dir / "cross_dataset_summary_significant.md")
        write_sidecar(
            build_provenance(
                [run.result for run in significant_runs],
                n_resamples=n_resamples,
                ci=ci,
                seed=seed,
            ),
            tables_dir / "cross_dataset_summary_significant.meta.json",
        )

    return html_path
