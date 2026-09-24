"""Paired significance stats for per-variant accuracy and deltas (ARCHITECTURE §2.7).

Confidence intervals come from a paired percentile bootstrap
(`scipy.stats.bootstrap` with ``paired=True``, so one shared set of resampled
indices is applied to all variants). The delta **p-values** use McNemar's exact
test via `scipy.stats.binomtest`, the standard significance test for paired
binary (correct/incorrect) outcomes on the same items.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, bootstrap


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a two-sided confidence interval."""

    point: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class DeltaEstimate:
    """A difference-of-accuracy estimate.

    ``point`` and the CI come from the paired bootstrap; ``p_value`` is
    McNemar's exact two-sided p-value for the paired binary comparison.
    """

    point: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Bootstrap accuracy and delta estimates for one evaluation run."""

    n: int
    acc: dict[str, Estimate]
    delta_paraphrase: DeltaEstimate
    delta_idiom: DeltaEstimate


def accuracy_from_correct(flags: Sequence[bool]) -> float:
    """Compute the mean of a sequence of correctness flags.

    Args:
        flags: Per-task correctness booleans.

    Returns:
        The mean as a float; `0.0` for an empty sequence.
    """
    if not flags:
        return 0.0
    return float(np.mean(np.asarray(flags, dtype=np.float64)))


def mcnemar_exact_p(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """McNemar's exact two-sided p-value for two paired binary variants.

    Tests marginal homogeneity (H0: ``acc(a) == acc(b)``) using only the
    discordant pairs, via an exact binomial test (`scipy.stats.binomtest`)
    with success probability 0.5.

    Args:
        a: First variant's correctness flags (0.0/1.0), paired with ``b``.
        b: Second variant's correctness flags (0.0/1.0), paired with ``a``.

    Returns:
        The exact two-sided p-value in ``[0, 1]``; ``1.0`` when there are no
        discordant pairs (no evidence of any difference).
    """
    a_correct = a.astype(bool)
    b_correct = b.astype(bool)
    n_a_right_b_wrong = int(np.sum(a_correct & ~b_correct))
    n_a_wrong_b_right = int(np.sum(~a_correct & b_correct))
    n_discordant = n_a_right_b_wrong + n_a_wrong_b_right
    if n_discordant == 0:
        return 1.0
    successes = min(n_a_right_b_wrong, n_a_wrong_b_right)
    return float(binomtest(successes, n_discordant, 0.5, alternative="two-sided").pvalue)


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two raters' paired binary judgments (RQ1 H1d/H1e).

    Args:
        a: First rater's binary judgments.
        b: Second rater's judgments, paired 1:1 with ``a`` (same item order).

    Returns:
        Kappa in ``[-1, 1]``; `1.0` when raters agree perfectly and would
        otherwise divide by zero (no disagreement possible by chance either,
        e.g. both raters always say the same single class).

    Raises:
        ValueError: If ``a`` and ``b`` have different lengths, or either is empty.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must have equal length; got {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("cohen_kappa requires at least one paired judgment")
    a_arr = np.asarray(a, dtype=bool)
    b_arr = np.asarray(b, dtype=bool)
    po = float(np.mean(a_arr == b_arr))
    p_a_true = float(np.mean(a_arr))
    p_b_true = float(np.mean(b_arr))
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def gwet_ac1(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Gwet's AC1 for two raters' paired binary judgments.

    Unlike Cohen's kappa, AC1's chance-agreement term does not blow up when
    the pooled prevalence is skewed near 0 or 1 (the "kappa paradox": high
    raw agreement but near-zero kappa under extreme marginals) — prefer this
    over `cohen_kappa` when one or both raters say "true" on the vast
    majority of items.

    Args:
        a: First rater's binary judgments.
        b: Second rater's judgments, paired 1:1 with ``a`` (same item order).

    Returns:
        AC1 in ``[-1, 1]``; `1.0` when raters agree perfectly and would
        otherwise divide by zero (no disagreement possible by chance either).

    Raises:
        ValueError: If ``a`` and ``b`` have different lengths, or either is empty.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must have equal length; got {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("gwet_ac1 requires at least one paired judgment")
    a_arr = np.asarray(a, dtype=bool)
    b_arr = np.asarray(b, dtype=bool)
    po = float(np.mean(a_arr == b_arr))
    pi_true = (float(np.mean(a_arr)) + float(np.mean(b_arr))) / 2
    pe = 2 * pi_true * (1 - pi_true)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def paired_bootstrap(
    correct_by_variant: Mapping[str, Sequence[bool]],
    *,
    n_resamples: int,
    ci: float,
    seed: int,
) -> BootstrapResult:
    """Run a paired percentile bootstrap over per-variant correctness flags.

    All variants are resampled with the SAME per-iteration index draws (paired
    resampling), since they score the same underlying tasks. Point estimates
    are the plain-mean accuracies; confidence intervals come from
    `scipy.stats.bootstrap` (paired, percentile method) over a single vector
    statistic so per-variant and delta CIs share the same resamples. Delta
    p-values use McNemar's exact test (`mcnemar_exact_p`).

    Args:
        correct_by_variant: Mapping from variant name to its ordered
            correctness flags. All sequences must share the same length.
        n_resamples: Number of bootstrap resamples to draw.
        ci: Confidence level in (0, 1), e.g. `0.95`.
        seed: Seed for the `numpy.random.default_rng` used for resampling.

    Returns:
        A `BootstrapResult` with per-variant accuracy estimates and the
        paraphrase-vs-original and idiomatic-vs-paraphrase delta estimates.

    Raises:
        ValueError: If the variant flag sequences have unequal lengths.
    """
    lengths = {len(flags) for flags in correct_by_variant.values()}
    if len(lengths) != 1:
        raise ValueError(
            "paired_bootstrap requires all variants to have equal-length, paired "
            f"flag sequences; got lengths {sorted(lengths)}"
        )
    n = next(iter(lengths))

    orig, para, idiom = (
        np.asarray(correct_by_variant[variant], dtype=np.float64)
        for variant in ("original", "paraphrase", "idiomatic")
    )
    # Point estimates are plain-mean accuracies (independent of the resampling).
    acc_point = {"original": float(orig.mean()), "paraphrase": float(para.mean())}
    acc_point["idiomatic"] = float(idiom.mean())

    def _stat(
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        axis: int = -1,
    ) -> NDArray[np.float64]:
        """Per-resample vector: [acc_orig, acc_para, acc_idiom, Δ_para, Δ_idiom]."""
        acc_a = a.mean(axis=axis)
        acc_b = b.mean(axis=axis)
        acc_c = c.mean(axis=axis)
        return np.stack([acc_a, acc_b, acc_c, acc_b - acc_a, acc_c - acc_b])

    # `paired=True` resamples one index array per iteration and applies it to all
    # three variants, so the per-variant accuracy CIs and the delta CIs are drawn
    # from the SAME resamples. Percentile method, seeded for determinism.
    interval = bootstrap(
        (orig, para, idiom),
        _stat,
        paired=True,
        vectorized=True,
        n_resamples=n_resamples,
        confidence_level=ci,
        method="percentile",
        rng=np.random.default_rng(seed),
    ).confidence_interval
    lo = np.asarray(interval.low, dtype=np.float64)
    hi = np.asarray(interval.high, dtype=np.float64)

    acc = {
        "original": Estimate(acc_point["original"], float(lo[0]), float(hi[0])),
        "paraphrase": Estimate(acc_point["paraphrase"], float(lo[1]), float(hi[1])),
        "idiomatic": Estimate(acc_point["idiomatic"], float(lo[2]), float(hi[2])),
    }
    return BootstrapResult(
        n=n,
        acc=acc,
        delta_paraphrase=DeltaEstimate(
            point=acc_point["paraphrase"] - acc_point["original"],
            ci_low=float(lo[3]),
            ci_high=float(hi[3]),
            p_value=mcnemar_exact_p(orig, para),
        ),
        delta_idiom=DeltaEstimate(
            point=acc_point["idiomatic"] - acc_point["paraphrase"],
            ci_low=float(lo[4]),
            ci_high=float(hi[4]),
            p_value=mcnemar_exact_p(para, idiom),
        ),
    )
