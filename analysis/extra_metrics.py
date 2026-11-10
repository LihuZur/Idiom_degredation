"""Extra result metrics (THESIS_PLAN_v2 §0b, §7): unrobustness U, parse failures, rank stability.

- U (FLUKE): mean |correct_variant - correct_original| per model and variant. It is symmetric, so
  improvements count as brittleness; the directional net change and the broken / fixed counts are
  reported next to it.
- Parse-failure rate per model and variant (answers that could not be mapped to a label), so
  accuracy gaps are not artefacts of answer extraction.
- Rank stability: Spearman correlation of the models' accuracy across two variants.
"""

from collections.abc import Sequence
from itertools import combinations

import pandas as pd
from scipy.stats import spearmanr

from analysis.results import ResultFile, variants_for

_U_COLUMNS = ["dataset", "model_id", "variant", "n", "U", "net_change", "n_broken", "n_fixed"]
_PARSE_COLUMNS = ["dataset", "model_id", "variant", "n", "n_unparseable", "rate"]
_RANK_COLUMNS = ["dataset", "variant_a", "variant_b", "n_models", "spearman_rho", "p_value"]


def _variants(result: ResultFile) -> list[str]:
    return list(result.per_task[0].variants) if result.per_task else []


def _correct(result: ResultFile, variant: str) -> list[bool]:
    return [t.variants[variant].correct for t in result.per_task]


def unrobustness_table(results: Sequence[ResultFile]) -> pd.DataFrame:
    """U, net change and broken/fixed counts of every non-baseline variant against the original."""
    rows: list[dict[str, str | int | float]] = []
    for r in sorted(results, key=lambda x: (x.dataset, x.model_id)):
        base = _correct(r, variants_for(r.dataset)[0])
        for variant in (v for v in _variants(r) if v != variants_for(r.dataset)[0]):
            flags = _correct(r, variant)
            n = len(flags)
            broken = sum(b and not v for b, v in zip(base, flags, strict=True))
            fixed = sum(v and not b for b, v in zip(base, flags, strict=True))
            rows.append(
                {
                    "dataset": r.dataset,
                    "model_id": r.model_id,
                    "variant": variant,
                    "n": n,
                    "U": (broken + fixed) / n if n else float("nan"),
                    "net_change": (fixed - broken) / n if n else float("nan"),
                    "n_broken": broken,
                    "n_fixed": fixed,
                }
            )
    return pd.DataFrame(rows, columns=_U_COLUMNS)


def parse_failure_table(results: Sequence[ResultFile]) -> pd.DataFrame:
    """Share of answers per model and variant whose output could not be parsed into a label."""
    rows: list[dict[str, str | int | float]] = []
    for r in sorted(results, key=lambda x: (x.dataset, x.model_id)):
        for variant in _variants(r):
            statuses = [t.variants[variant].parse_status for t in r.per_task]
            bad = sum(s != "ok" for s in statuses)
            rows.append(
                {
                    "dataset": r.dataset,
                    "model_id": r.model_id,
                    "variant": variant,
                    "n": len(statuses),
                    "n_unparseable": bad,
                    "rate": bad / len(statuses) if statuses else float("nan"),
                }
            )
    return pd.DataFrame(rows, columns=_PARSE_COLUMNS)


def rank_stability_table(results: Sequence[ResultFile]) -> pd.DataFrame:
    """Spearman correlation of model accuracies between each pair of variants, per dataset."""
    rows: list[dict[str, str | int | float]] = []
    for dataset in sorted({r.dataset for r in results}):
        ds = [r for r in results if r.dataset == dataset and r.per_task]
        variants = _variants(ds[0]) if ds else []
        for a, b in combinations(variants, 2):
            acc_a = [sum(_correct(r, a)) / len(r.per_task) for r in ds]
            acc_b = [sum(_correct(r, b)) / len(r.per_task) for r in ds]
            rho, p = spearmanr(acc_a, acc_b)
            rows.append(
                {
                    "dataset": dataset,
                    "variant_a": a,
                    "variant_b": b,
                    "n_models": len(ds),
                    "spearman_rho": float(rho),  # pyright: ignore[reportArgumentType]
                    "p_value": float(p),  # pyright: ignore[reportArgumentType]
                }
            )
    return pd.DataFrame(rows, columns=_RANK_COLUMNS)
