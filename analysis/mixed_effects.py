"""RQ1/RQ2 mixed-effects re-analysis.

RQ1 (paper_preparations/RQ1_meaning_preservation.md §5 step 8): fits
`is_correct ~ arm + (1 | row_id) + (1 | model_id)` via statsmodels
`MixedLM` on the per-row Bernoulli outcome, per dataset, on the
meaning-validated (accepted) subset. This is a linear approximation to the
binary outcome, matching the RQ1 plan's stated method; a Firth-corrected
logistic fallback for non-convergent fits is noted as a TODO (not yet needed
— no real data has been fit through this module yet).

RQ2 (paper_preparations/RQ2_idiom_properties.md §5 step 5): fits
`correct_idi ~ opacity + log_freq + opacity:log_params + correct_par
+ (1 | row_id) + (1 | model_id)` to test H2a (opacity), H2b (frequency),
and H2c (opacity x scale interaction).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf

from analysis.results import ResultFile, filter_to_accepted, load_result

_REQUIRED_COLUMNS = {"is_correct", "arm", "row_id", "model_id"}

# Parameter counts (billions), from each model's public HF model card —
# RQ2 plan §5 step 5's `opacity : log(params)` interaction term (H2c).
# Keep in sync with `models/registry.py`'s registered names.
MODEL_PARAMS_B: dict[str, float] = {
    "qwen3.5-0.5b-instruct": 0.5,
    "qwen3.5-1.5b-instruct": 1.5,
    "qwen3.5-7b-instruct": 7.0,
    "falcon3-3b-instruct": 3.0,
    "falcon3-7b-instruct": 7.0,
    "gemma-4-2b-instruct": 2.0,
    "granite-3.1-2b-instruct": 2.0,
    "h2o-danube3-4b-chat": 4.0,
    "mistral-7b-instruct-v0.3": 7.0,
    "olmo-2-1b-instruct": 1.0,
    "olmo-2-7b-instruct": 7.0,
    "phi-4-mini-instruct": 3.8,
    "smollm2-1.7b-instruct": 1.7,
    "yi-1.5-6b-chat": 6.0,
}

_RQ2_REQUIRED_NORM_COLUMNS = {"id", "opacity", "log_freq"}


@dataclass(frozen=True, slots=True)
class MixedEffectsResult:
    """The fixed-effect coefficient on `arm` from a `fit_arm_effect` call."""

    coef: float
    ci_low: float
    ci_high: float
    p_value: float
    converged: bool


def fit_arm_effect(df: pd.DataFrame) -> MixedEffectsResult:
    """Fit `is_correct ~ arm + (1 | row_id) + (1 | model_id)` for one dataset.

    Args:
        df: Long-format rows with columns `is_correct` (0/1), `arm`
            (categorical, e.g. `"original"`/`"par"`/`"idi"`), `row_id`, and
            `model_id`. One row per (row_id, arm, model_id) observation.

    Returns:
        A `MixedEffectsResult` for the fixed effect of the non-reference
        `arm` level with the largest-magnitude coefficient (the two-arm
        case has exactly one non-reference term).

    Raises:
        ValueError: If any required column is missing, or no `arm`
            fixed-effect term is found after fitting (e.g. `arm` has only
            one level).
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    model = smf.mixedlm(
        "is_correct ~ arm",
        df,
        groups=df["row_id"],
        vc_formula={"model_id": "0 + C(model_id)"},
    )
    result = model.fit()

    arm_terms = [t for t in result.params.index if t.startswith("arm[")]
    if not arm_terms:
        raise ValueError("no `arm` fixed-effect term found after fitting")
    term = arm_terms[-1]
    ci_low, ci_high = result.conf_int().loc[term]
    return MixedEffectsResult(
        coef=float(result.params[term]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(result.pvalues[term]),
        converged=bool(result.converged),
    )


def build_arm_dataframe(
    dataset: str,
    *,
    results_dir: Path,
    accepted_ids: frozenset[str] | None = None,
    arms: tuple[str, str] = ("paraphrase", "idiomatic"),
) -> pd.DataFrame:
    """Merge every model's Stage 3 result file into long rows for `fit_arm_effect` (RQ1 plan §5 step 8).

    Args:
        dataset: Dataset name (e.g. `"sst2"`).
        results_dir: Root directory containing `{dataset}/{model_id}.json` files.
        accepted_ids: If given (RQ1 plan step 5's accept rule), restrict to
            these row ids — the meaning-validated subset the chapter re-analyses.
        arms: The two `VARIANTS` names to compare. `fit_arm_effect` assumes a
            two-arm design (see its docstring). Defaults to
            `("paraphrase", "idiomatic")` to match the course paper's
            established headline metric `Δ_idi = acc_idi - acc_par` (the
            "extra cost of idiomaticity" *over and above* a non-idiomatic
            rewrite of the same input — NOT vs the unmodified original,
            which also bundles in the cost of rewording alone, `Δ_par`).

    Returns:
        One row per (row_id, model_id, arm) with columns `row_id`,
        `model_id`, `arm`, `is_correct`.
    """
    records: list[dict[str, Any]] = []
    for result_path in sorted((results_dir / dataset).glob("*.json")):
        model_id = result_path.stem
        result: ResultFile = load_result(result_path)
        if accepted_ids is not None:
            result = filter_to_accepted(result, accepted_ids)

        for task in result.per_task:
            for arm in arms:
                records.append(
                    {
                        "row_id": task.id,
                        "model_id": model_id,
                        "arm": arm,
                        "is_correct": int(task.variants[arm].correct),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_rq2_dataframe(
    dataset: str,
    *,
    results_dir: Path,
    datasets_dir: Path,
    accepted_ids: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Merge `idiom_norms.csv` with every model's Stage 3 result file for one dataset.

    Args:
        dataset: Dataset name (e.g. `"sst2"`).
        results_dir: Root directory containing `{dataset}/{model_id}.json` files.
        datasets_dir: Root directory containing `{dataset}/idiom_norms.csv`.
        accepted_ids: If given (RQ1 meaning-preservation accept rule), also
            restrict to ids present in this set (plan §8 meaning-drift
            confound sensitivity check).

    Returns:
        One row per (row_id, model_id) with columns `row_id`, `model_id`,
        `opacity`, `log_freq`, `log_params`, `correct_par`, `correct_idi`.
        Rows whose idiom has no opacity/frequency linkage (`idiom_norms.csv`
        `opacity`/`log_freq` is `NaN`) are dropped.

    Raises:
        ValueError: If a discovered result file's `model_id` has no known
            entry in `MODEL_PARAMS_B`.
    """
    norms_path = datasets_dir / dataset / "idiom_norms.csv"
    norms = pd.read_csv(norms_path, dtype={"id": str})
    missing_norm_cols = _RQ2_REQUIRED_NORM_COLUMNS - set(norms.columns)
    if missing_norm_cols:
        raise ValueError(f"{norms_path}: missing required columns: {sorted(missing_norm_cols)}")
    norms = norms.dropna(subset=["opacity", "log_freq"])
    norms_by_id = norms.set_index("id")[["opacity", "log_freq"]]

    records: list[dict[str, Any]] = []
    for result_path in sorted((results_dir / dataset).glob("*.json")):
        model_id = result_path.stem
        params_b = MODEL_PARAMS_B.get(model_id)
        if params_b is None:
            raise ValueError(
                f"no known parameter count for model_id {model_id!r}; add it to MODEL_PARAMS_B"
            )
        log_params = math.log(params_b)

        result: ResultFile = load_result(result_path)
        if accepted_ids is not None:
            result = filter_to_accepted(result, accepted_ids)

        for task in result.per_task:
            if task.id not in norms_by_id.index:
                continue
            norm_row = norms_by_id.loc[task.id]
            records.append(
                {
                    "row_id": task.id,
                    "model_id": model_id,
                    "opacity": float(norm_row["opacity"]),
                    "log_freq": float(norm_row["log_freq"]),
                    "log_params": log_params,
                    "correct_par": int(task.variants["paraphrase"].correct),
                    "correct_idi": int(task.variants["idiomatic"].correct),
                }
            )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True, slots=True)
class OpacityFrequencyResult:
    """Fixed-effect coefficients from a `fit_opacity_frequency_effect` call (H2a/b/c)."""

    opacity_coef: float
    opacity_ci: tuple[float, float]
    opacity_p: float
    log_freq_coef: float
    log_freq_ci: tuple[float, float]
    log_freq_p: float
    log_params_coef: float
    log_params_ci: tuple[float, float]
    log_params_p: float
    interaction_coef: float
    interaction_ci: tuple[float, float]
    interaction_p: float
    correct_par_coef: float
    converged: bool
    n_obs: int


_RQ2_REQUIRED_COLUMNS = {
    "correct_idi",
    "opacity",
    "log_freq",
    "log_params",
    "correct_par",
    "row_id",
    "model_id",
}


def fit_opacity_frequency_effect(df: pd.DataFrame) -> OpacityFrequencyResult:
    """Fit H2a/H2b/H2c for one dataset.

    `correct_idi ~ opacity_c + log_freq + log_params_c + opacity_c:log_params_c
    + correct_par` with a row-id random intercept. `opacity_c`/`log_params_c`
    are mean-centered versions of `opacity`/`log_params`.

    RQ2 plan §5 step 5 originally specified `opacity:log_params` without a
    standalone `log_params` main effect. That is a marginality violation:
    since `opacity` is never 0 in practice (mean ~0.3 in all three datasets),
    omitting the main effect lets the interaction term silently absorb
    `log_params`'s own (large, highly significant) main effect — VIF on the
    raw interaction term is 6-7 in all three datasets, over plan §8's own
    orthogonalisation threshold ("if VIF > 5, orthogonalise via
    residualisation"). Centering both variables before forming the
    interaction (the standard fix for this) drops VIF to ~1 and is
    equivalent to including the main effect; discovered 2026-09-14 when a
    `log_params` main-effect robustness check made H2c's significance vanish
    in mmlu/mnli.

    A `model_id` variance component was tried too (matching `fit_arm_effect`'s
    crossed-effects pattern), but with exactly one observation per
    (row_id, model_id) cell that leaves the row-id intercept, the model_id
    variance, and the residual scale mutually unidentifiable: the row-id
    variance collapses to exactly 0 and the fit reports a non-positive-
    definite Hessian. Re-fit as the standard two-way-crossed variance
    components (`groups` fixed at a single level, `row_id` and `model_id`
    both as `vc_formula` terms) confirmed the model_id variance is ~0 once it
    *can* be separately estimated (0.00015 vs. row_id's 0.029) — i.e. there's
    no real between-model variance left after the `opacity:log_params` fixed
    effect already captures model-scale-dependent variation. Dropping the
    model_id term removes the non-identifiability, converges cleanly with no
    Hessian warning, and leaves the H2a/H2c estimates essentially unchanged.

    Args:
        df: Long-format rows as returned by `build_rq2_dataframe`.

    Returns:
        The fixed-effect coefficients on `opacity` (H2a), `log_freq` (H2b),
        `log_params` (main effect, control), `opacity:log_params` (H2c), and
        `correct_par` (control).

    Raises:
        ValueError: If any required column is missing.
    """
    missing = _RQ2_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = df.assign(
        opacity_c=df["opacity"] - df["opacity"].mean(),
        log_params_c=df["log_params"] - df["log_params"].mean(),
    )

    model = smf.mixedlm(
        "correct_idi ~ opacity_c + log_freq + log_params_c + opacity_c:log_params_c + correct_par",
        df,
        groups=df["row_id"],
    )
    result = model.fit()

    def _term(name: str) -> tuple[float, tuple[float, float], float]:
        ci_low, ci_high = result.conf_int().loc[name]
        return (
            float(result.params[name]),
            (float(ci_low), float(ci_high)),
            float(result.pvalues[name]),
        )

    opacity_coef, opacity_ci, opacity_p = _term("opacity_c")
    log_freq_coef, log_freq_ci, log_freq_p = _term("log_freq")
    log_params_coef, log_params_ci, log_params_p = _term("log_params_c")
    interaction_coef, interaction_ci, interaction_p = _term("opacity_c:log_params_c")

    return OpacityFrequencyResult(
        opacity_coef=opacity_coef,
        opacity_ci=opacity_ci,
        opacity_p=opacity_p,
        log_freq_coef=log_freq_coef,
        log_freq_ci=log_freq_ci,
        log_freq_p=log_freq_p,
        log_params_coef=log_params_coef,
        log_params_ci=log_params_ci,
        log_params_p=log_params_p,
        interaction_coef=interaction_coef,
        interaction_ci=interaction_ci,
        interaction_p=interaction_p,
        correct_par_coef=float(result.params["correct_par"]),
        converged=bool(result.converged),
        n_obs=len(df),
    )
