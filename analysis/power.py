"""Power simulation for the primary comparison (THESIS_PLAN_v2 §0c, step 5).

Item-level approximation of the primary test: for each item the difference d = mean over models of
(correct on the control - correct on the idiomatic variant), so positive d means the idiom hurts.
The pilot's item-to-item spread of d is resampled with a planted mean effect; the test is a two-sided
one-sample t-test on d across items. Items are the unit because models see the same items, which is
what the mixed-effects model's item random effect accounts for; this is a simplification, so treat the
numbers as planning figures.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm, t

from analysis.results import ResultFile

_COLUMNS = [
    "dataset",
    "n_pilot_items",
    "pilot_mean_diff_pp",
    "item_sd",
    "n",
    "effect_pp",
    "power",
    "mde_pp_at_target_power",
]


def item_differences(
    results: Sequence[ResultFile], dataset: str, control: str, treat: str
) -> NDArray[np.float64]:
    """Per-item mean over models of (control correct - treat correct), on ids every model has."""
    ds = [r for r in results if r.dataset == dataset]
    if not ds:
        raise ValueError(f"no results for dataset {dataset!r}")
    per_model = [
        {
            task.id: float(task.variants[control].correct) - float(task.variants[treat].correct)
            for task in r.per_task
        }
        for r in ds
    ]
    shared = set.intersection(*(set(m) for m in per_model))
    return np.array([np.mean([m[i] for m in per_model]) for i in sorted(shared)], dtype=np.float64)


def simulate_power(
    d: NDArray[np.float64],
    n: int,
    effect: float,
    *,
    alpha: float,
    reps: int,
    rng: np.random.Generator,
) -> float:
    """Share of simulated studies with n items that reject H0 (mean difference = 0)."""
    centred = d - d.mean()
    idx = rng.integers(0, len(d), size=(reps, n))
    sample = centred[idx] + effect
    mean = sample.mean(axis=1)
    sd = sample.std(axis=1, ddof=1)
    stat = mean / (sd / np.sqrt(n))
    p = 2 * t.sf(np.abs(stat), df=n - 1)
    return float(np.mean(p < alpha))


def minimum_detectable_effect(sd: float, n: int, *, alpha: float, power: float) -> float:
    """Analytic effect (as a fraction) detectable with the given power."""
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / np.sqrt(n))


def power_table(
    results: Sequence[ResultFile],
    datasets: Sequence[str],
    *,
    control: str,
    treat: str,
    n_grid: Sequence[int],
    effects_pp: Sequence[float],
    alpha: float,
    target_power: float,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int | float]] = []
    for dataset in datasets:
        d = item_differences(results, dataset, control, treat)
        sd = float(d.std(ddof=1))
        for n in n_grid:
            mde = 100 * minimum_detectable_effect(sd, n, alpha=alpha, power=target_power)
            for effect in effects_pp:
                rows.append(
                    {
                        "dataset": dataset,
                        "n_pilot_items": len(d),
                        "pilot_mean_diff_pp": 100 * float(d.mean()),
                        "item_sd": sd,
                        "n": n,
                        "effect_pp": effect,
                        "power": simulate_power(
                            d, n, effect / 100, alpha=alpha, reps=reps, rng=rng
                        ),
                        "mde_pp_at_target_power": mde,
                    }
                )
    return pd.DataFrame(rows, columns=_COLUMNS)
