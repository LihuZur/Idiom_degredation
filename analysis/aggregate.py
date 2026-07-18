"""Build the cross-run summary DataFrame consumed by Stage 4 tables/plots (ARCHITECTURE §2.7)."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analysis.results import ResultFile, correct_by_variant, discover_results, load_result
from analysis.stats import BootstrapResult, paired_bootstrap

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


@dataclass(frozen=True, slots=True)
class RunStats:
    """A loaded result file paired with its bootstrap statistics."""

    result: ResultFile
    boot: BootstrapResult


def bootstrap_by_run(
    results_dir: Path,
    *,
    n_resamples: int,
    ci: float,
    seed: int,
) -> list[RunStats]:
    """Load every discovered result file and compute its paired bootstrap.

    Args:
        results_dir: Root directory containing ``{dataset}/{model}.json`` files.
        n_resamples: Number of bootstrap resamples to draw per run.
        ci: Confidence level in (0, 1) for the bootstrap intervals.
        seed: RNG seed for bootstrap resampling (shared across runs).

    Returns:
        One `RunStats` per discovered result file, in discovery order.
    """
    stats: list[RunStats] = []
    for path in discover_results(results_dir):
        result = load_result(path)
        boot = paired_bootstrap(
            correct_by_variant(result),
            n_resamples=n_resamples,
            ci=ci,
            seed=seed,
        )
        stats.append(RunStats(result=result, boot=boot))
    return stats


def aggregate(
    results_dir: Path,
    *,
    n_resamples: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    """Build a one-row-per-(dataset, model) summary table of bootstrap statistics.

    Args:
        results_dir: Root directory containing ``{dataset}/{model}.json`` files.
        n_resamples: Number of bootstrap resamples to draw per run.
        ci: Confidence level in (0, 1) for the bootstrap intervals.
        seed: RNG seed for bootstrap resampling (shared across runs).

    Returns:
        A DataFrame with columns `dataset, model_id, model_revision,
        config_hash, prompt_hash, n, acc_original, acc_paraphrase,
        acc_idiomatic, delta_paraphrase, delta_paraphrase_ci_low,
        delta_paraphrase_ci_high, delta_paraphrase_p, delta_idiom,
        delta_idiom_ci_low, delta_idiom_ci_high, delta_idiom_p` — one row
        per result file, in discovery order.
    """
    rows: list[dict[str, str | int | float]] = []
    for run in bootstrap_by_run(results_dir, n_resamples=n_resamples, ci=ci, seed=seed):
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
            }
        )
    return pd.DataFrame(rows, columns=_COLUMNS)
