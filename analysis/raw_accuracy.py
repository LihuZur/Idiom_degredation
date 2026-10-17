"""Raw per-variant accuracy table: original / paraphrase / idiomatic, per model and dataset."""

import pandas as pd

from analysis.aggregate import RunStats
from analysis.results import VARIANTS

_COLUMNS = ["dataset", "model_id", "n", "variant", "correct", "acc", "ci_low", "ci_high"]


def raw_accuracy_table(runs: list[RunStats], datasets: tuple[str, ...]) -> pd.DataFrame:
    """Long-format raw accuracies with bootstrap CIs, plus an unweighted mean over models.

    Args:
        runs: Bootstrapped runs from `analysis.aggregate.bootstrap_by_run`.
        datasets: Datasets to keep, in output order.

    Returns:
        One row per (dataset, model, variant); `model_id == "MEAN"` rows hold the
        unweighted mean over models (point estimate only, CI columns are NaN).
    """
    rows: list[dict[str, str | int | float]] = []
    for dataset in datasets:
        ds_runs = [r for r in runs if r.result.dataset == dataset]
        for run in sorted(ds_runs, key=lambda r: r.result.model_id):
            for variant in VARIANTS:
                est = run.boot.acc[variant]
                rows.append(
                    {
                        "dataset": dataset,
                        "model_id": run.result.model_id,
                        "n": run.boot.n,
                        "variant": variant,
                        "correct": round(est.point * run.boot.n),
                        "acc": est.point,
                        "ci_low": est.ci_low,
                        "ci_high": est.ci_high,
                    }
                )
        for variant in VARIANTS:
            points = [r.boot.acc[variant].point for r in ds_runs]
            if points:
                rows.append(
                    {
                        "dataset": dataset,
                        "model_id": "MEAN",
                        "n": ds_runs[0].boot.n,
                        "variant": variant,
                        "correct": -1,
                        "acc": sum(points) / len(points),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                    }
                )
    return pd.DataFrame(rows, columns=_COLUMNS)


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model); accuracy columns in percent, `acc [lo, hi]` strings."""
    rows: dict[tuple[str, str], dict[str, str | int]] = {}
    for rec in long.to_dict("records"):
        key = (str(rec["dataset"]), str(rec["model_id"]))
        row = rows.setdefault(
            key,
            {"dataset": key[0], "model_id": key[1], "n": int(rec["n"])},  # pyright: ignore[reportUnknownArgumentType]
        )
        cell = f"{100 * float(rec['acc']):.2f}"  # pyright: ignore[reportUnknownArgumentType]
        if pd.notna(rec["ci_low"]):
            cell += f" [{100 * float(rec['ci_low']):.1f}, {100 * float(rec['ci_high']):.1f}]"  # pyright: ignore[reportUnknownArgumentType]
        row[str(rec["variant"])] = cell
    return pd.DataFrame(list(rows.values()))
