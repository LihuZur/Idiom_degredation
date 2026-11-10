"""Raw per-variant accuracy table per model and dataset.

Pilot datasets report original / paraphrase / idiomatic; v2 datasets report
original / span_literal / idiomatic (`analysis.results.variants_for`).
"""

import pandas as pd
from scipy.stats import chi2

from analysis.aggregate import RunStats
from analysis.results import correct_by_variant, variants_for

_COLUMNS = ["dataset", "model_id", "n", "variant", "correct", "acc", "ci_low", "ci_high"]
_SOLE_COLUMNS = ["dataset", "model_id", "n_orig_correct", "b", "c", "diff", "chi2", "p"]


def _mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar chi-square with continuity correction; `(0.0, 1.0)` without discordant rows."""
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return stat, float(chi2.sf(stat, 1))


def sole_cause_table(runs: list[RunStats], datasets: tuple[str, ...]) -> pd.DataFrame:
    """Per-model sole-cause counts, matching the definitions in `idiom-variant-confusion`.

    Among rows the model got right on the original: `b` = idiomatic wrong while
    the control (paraphrase, or span_literal for v2 datasets) right, `c` = control
    wrong while idiomatic right; `diff = b - c`.
    A `POOLED` row sums b and c over models (indicative only: a row that is hard
    for many models is counted many times).
    """
    rows: list[dict[str, str | int | float]] = []
    for dataset in datasets:
        ds_runs = sorted(
            (r for r in runs if r.result.dataset == dataset), key=lambda r: r.result.model_id
        )
        tot_b = tot_c = tot_orig = 0
        for run in ds_runs:
            flags = correct_by_variant(run.result)
            base, control, treat = variants_for(dataset)
            trio = list(zip(flags[base], flags[control], flags[treat], strict=True))
            b = sum(o and p and not i for o, p, i in trio)
            c = sum(o and not p and i for o, p, i in trio)
            n_orig = sum(o for o, _, _ in trio)
            stat, p_value = _mcnemar(b, c)
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": run.result.model_id,
                    "n_orig_correct": n_orig,
                    "b": b,
                    "c": c,
                    "diff": b - c,
                    "chi2": stat,
                    "p": p_value,
                }
            )
            tot_b, tot_c, tot_orig = tot_b + b, tot_c + c, tot_orig + n_orig
        if ds_runs:
            stat, p_value = _mcnemar(tot_b, tot_c)
            rows.append(
                {
                    "dataset": dataset,
                    "model_id": "POOLED",
                    "n_orig_correct": tot_orig,
                    "b": tot_b,
                    "c": tot_c,
                    "diff": tot_b - tot_c,
                    "chi2": stat,
                    "p": p_value,
                }
            )
    return pd.DataFrame(rows, columns=_SOLE_COLUMNS)


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
            for variant in variants_for(dataset):
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
        for variant in variants_for(dataset):
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


_P_FLOOR = 1e-4


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < _P_FLOOR else f"{p:.4f}"


def to_wide(long: pd.DataFrame, sole: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (dataset, model); accuracy columns in percent, `acc [lo, hi]` strings.

    With `sole`, appends `b / c` (idiomatic-only-wrong / paraphrase-only-wrong) and the
    McNemar p; the `MEAN` rows have no b/c and the `POOLED` rows have no accuracy.
    """
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
    if sole is not None:
        for rec in sole.to_dict("records"):
            key = (str(rec["dataset"]), str(rec["model_id"]))
            row = rows.setdefault(
                key,
                {"dataset": key[0], "model_id": key[1], "n": int(rec["n_orig_correct"])},  # pyright: ignore[reportUnknownArgumentType]
            )
            row["b / c"] = f"{int(rec['b'])} / {int(rec['c'])}"  # pyright: ignore[reportUnknownArgumentType]
            row["McNemar p"] = _fmt_p(float(rec["p"]))  # pyright: ignore[reportUnknownArgumentType]
    return pd.DataFrame(list(rows.values())).fillna("")
