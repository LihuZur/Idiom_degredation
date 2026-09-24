"""Tests for analysis/stats.py: accuracy, bootstrap CIs, and McNemar p-values."""

import numpy as np
import pytest
from scipy.stats import binomtest

from analysis.stats import (
    BootstrapResult,
    accuracy_from_correct,
    cohen_kappa,
    gwet_ac1,
    mcnemar_exact_p,
    paired_bootstrap,
)

_N_RESAMPLES = 500
_CI = 0.95
_SEED = 0


def test_accuracy_from_correct_toy_flags() -> None:
    assert accuracy_from_correct([True, True, False, False]) == pytest.approx(0.5)


def test_accuracy_from_correct_empty_is_zero() -> None:
    assert accuracy_from_correct([]) == 0.0


def _known_correct_by_variant() -> dict[str, list[bool]]:
    # original: 5/10 = 0.5, paraphrase: 7/10 = 0.7, idiomatic: 4/10 = 0.4
    return {
        "original": [True] * 5 + [False] * 5,
        "paraphrase": [True] * 7 + [False] * 3,
        "idiomatic": [True] * 4 + [False] * 6,
    }


def test_delta_math_known_case() -> None:
    result = paired_bootstrap(
        _known_correct_by_variant(), n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )
    assert result.acc["original"].point == pytest.approx(0.5)
    assert result.acc["paraphrase"].point == pytest.approx(0.7)
    assert result.acc["idiomatic"].point == pytest.approx(0.4)
    assert result.delta_paraphrase.point == pytest.approx(0.2)
    assert result.delta_idiom.point == pytest.approx(-0.3)


def test_bootstrap_determinism_same_seed() -> None:
    data = _known_correct_by_variant()
    first = paired_bootstrap(data, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)
    second = paired_bootstrap(data, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)
    assert first == second


def test_bootstrap_different_seed_runs_and_keeps_same_point_estimates() -> None:
    data = _known_correct_by_variant()
    baseline = paired_bootstrap(data, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)
    other = paired_bootstrap(data, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED + 1)
    # Point estimates are plain means, independent of resampling seed.
    for variant in ("original", "paraphrase", "idiomatic"):
        assert other.acc[variant].point == pytest.approx(baseline.acc[variant].point)
    assert other.delta_paraphrase.point == pytest.approx(baseline.delta_paraphrase.point)
    assert other.delta_idiom.point == pytest.approx(baseline.delta_idiom.point)


def test_ci_brackets_point_estimate() -> None:
    result = paired_bootstrap(
        _known_correct_by_variant(), n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )
    for variant in ("original", "paraphrase", "idiomatic"):
        est = result.acc[variant]
        assert est.ci_low <= est.point <= est.ci_high
    for delta in (result.delta_paraphrase, result.delta_idiom):
        assert delta.ci_low <= delta.point <= delta.ci_high


def test_p_value_in_unit_interval() -> None:
    result = paired_bootstrap(
        _known_correct_by_variant(), n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED
    )
    assert 0.0 <= result.delta_paraphrase.p_value <= 1.0
    assert 0.0 <= result.delta_idiom.p_value <= 1.0


def test_identity_edge_case_all_variants_equal() -> None:
    flags = [True, True, False, False, True, False, True, True, False, False]
    correct_by_variant = {"original": flags, "paraphrase": flags, "idiomatic": flags}
    result = paired_bootstrap(correct_by_variant, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)
    assert isinstance(result, BootstrapResult)
    for delta in (result.delta_paraphrase, result.delta_idiom):
        assert delta.point == 0.0
        assert delta.ci_low == 0.0
        assert delta.ci_high == 0.0
        assert delta.p_value == 1.0


def test_mcnemar_matches_scipy_binomtest_on_known_discordant_case() -> None:
    n = 10
    # original all wrong, paraphrase all right -> all n pairs discordant one way;
    # idiomatic == paraphrase -> zero discordant pairs.
    data = {
        "original": [False] * n,
        "paraphrase": [True] * n,
        "idiomatic": [True] * n,
    }
    result = paired_bootstrap(data, n_resamples=_N_RESAMPLES, ci=_CI, seed=_SEED)
    expected = float(binomtest(0, n, 0.5, alternative="two-sided").pvalue)
    assert result.delta_paraphrase.p_value == pytest.approx(expected)
    # No discordant pairs -> McNemar yields exactly 1.0.
    assert result.delta_idiom.p_value == 1.0


def test_mcnemar_exact_p_no_discordant_pairs_is_one() -> None:
    a = np.array([1.0, 0.0, 1.0, 0.0])
    assert mcnemar_exact_p(a, a) == 1.0


def test_paired_bootstrap_rejects_unequal_length_variants() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap(
            {"original": [True, False], "paraphrase": [True], "idiomatic": [True, False]},
            n_resamples=_N_RESAMPLES,
            ci=_CI,
            seed=_SEED,
        )


def test_cohen_kappa_perfect_agreement_is_one() -> None:
    a = [True, False, True, True, False, False]
    assert cohen_kappa(a, a) == pytest.approx(1.0)


def test_cohen_kappa_known_value() -> None:
    # Standard textbook example: 2x2 confusion counts
    # agree-true=20, a-true/b-false=5, a-false/b-true=10, agree-false=15 (n=50).
    a = [True] * 25 + [False] * 25
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    po = (20 + 15) / 50
    pe = (25 / 50) * (30 / 50) + (25 / 50) * (20 / 50)
    expected = (po - pe) / (1 - pe)
    assert cohen_kappa(a, b) == pytest.approx(expected)


def test_cohen_kappa_rejects_unequal_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        cohen_kappa([True, False], [True])


def test_cohen_kappa_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        cohen_kappa([], [])


def test_cohen_kappa_all_same_class_both_raters_is_one() -> None:
    # No possible disagreement and no chance-agreement headroom -> defined as 1.0.
    assert cohen_kappa([True, True, True], [True, True, True]) == 1.0


def test_gwet_ac1_perfect_agreement_is_one() -> None:
    a = [True, False, True, True, False, False]
    assert gwet_ac1(a, a) == pytest.approx(1.0)


def test_gwet_ac1_rejects_unequal_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        gwet_ac1([True, False], [True])


def test_gwet_ac1_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        gwet_ac1([], [])


def test_gwet_ac1_stays_high_under_skewed_prevalence_where_kappa_collapses() -> None:
    # 18/20 raw agreement with near-uniform "True" prevalence: kappa's chance-agreement
    # term inflates toward 1 under this skew and collapses kappa well below raw agreement,
    # while AC1 is designed to track raw agreement more closely in this regime.
    a = [True] * 19 + [False]
    b = [True] * 17 + [False] * 2 + [True]
    kappa = cohen_kappa(a, b)
    ac1 = gwet_ac1(a, b)
    raw_agreement = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    assert kappa < ac1
    assert ac1 == pytest.approx(raw_agreement, abs=0.15)
