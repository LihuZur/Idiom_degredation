"""Unit test for `analysis.mixed_effects` — synthetic data, no real judge output needed."""

import numpy as np
import pandas as pd
import pytest

from analysis.mixed_effects import (
    MixedEffectsResult,
    OpacityFrequencyResult,
    fit_arm_effect,
    fit_opacity_frequency_effect,
)


def _synthetic_df(seed: int = 0, n_rows: int = 80, n_models: int = 6) -> pd.DataFrame:
    """Rows where `idi` is markedly less accurate than `original` (known-sign effect)."""
    rng = np.random.default_rng(seed)
    records = []
    for row_id in range(n_rows):
        row_bump = rng.normal(0.0, 0.05)
        for model_id in range(n_models):
            for arm, p_correct in (("original", 0.7), ("idi", 0.45)):
                p = min(max(p_correct + row_bump, 0.01), 0.99)
                records.append(
                    {
                        "row_id": f"r{row_id}",
                        "model_id": f"m{model_id}",
                        "arm": arm,
                        "is_correct": int(rng.random() < p),
                    }
                )
    return pd.DataFrame(records)


def test_fit_arm_effect_recovers_known_sign() -> None:
    result = fit_arm_effect(_synthetic_df())
    assert isinstance(result, MixedEffectsResult)
    # "idi" < "original" in the synthetic data; alphabetically "idi" is the
    # reference level, so the fixed effect on "original" should be positive.
    assert result.coef > 0
    assert result.ci_low < result.coef < result.ci_high
    assert 0.0 <= result.p_value <= 1.0


def test_fit_arm_effect_rejects_missing_columns() -> None:
    df = pd.DataFrame({"is_correct": [1, 0], "arm": ["original", "idi"]})
    with pytest.raises(ValueError, match="missing required columns"):
        fit_arm_effect(df)


def _synthetic_rq2_df(seed: int = 0, n_rows: int = 100, n_models: int = 5) -> pd.DataFrame:
    """Rows where higher opacity lowers `correct_idi` (known-sign effect)."""
    rng = np.random.default_rng(seed)
    records = []
    for row_id in range(n_rows):
        opacity = rng.normal(0.0, 1.0)
        log_freq = rng.normal(5.0, 1.0)
        row_bump = rng.normal(0.0, 0.05)
        for model_id in range(n_models):
            log_params = rng.choice([np.log(0.5), np.log(1.5), np.log(7.0)])
            correct_par = int(rng.random() < 0.8)
            logit = 1.0 - 0.8 * opacity + 0.1 * log_freq + 0.5 * correct_par + row_bump
            p = 1.0 / (1.0 + np.exp(-logit))
            records.append(
                {
                    "row_id": f"r{row_id}",
                    "model_id": f"m{model_id}",
                    "opacity": opacity,
                    "log_freq": log_freq,
                    "log_params": log_params,
                    "correct_par": correct_par,
                    "correct_idi": int(rng.random() < p),
                }
            )
    return pd.DataFrame(records)


def test_fit_opacity_frequency_effect_recovers_known_sign() -> None:
    result = fit_opacity_frequency_effect(_synthetic_rq2_df())
    assert isinstance(result, OpacityFrequencyResult)
    assert result.opacity_coef < 0
    assert result.opacity_ci[0] < result.opacity_coef < result.opacity_ci[1]
    assert 0.0 <= result.opacity_p <= 1.0
    assert result.n_obs == 500


def test_fit_opacity_frequency_effect_rejects_missing_columns() -> None:
    df = pd.DataFrame({"correct_idi": [1, 0], "opacity": [0.1, 0.2]})
    with pytest.raises(ValueError, match="missing required columns"):
        fit_opacity_frequency_effect(df)
