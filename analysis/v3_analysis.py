"""v3 analysis (THESIS_PLAN_v3 §5): main test per task, per-model check, parse rates.

Main test: mixed-effects logistic regression `correct ~ idiomatic + log_params + (1|item) + (1|model)` on the
idiomatic and literal answers of every model (pre-registered in plan v2 §0c), Holm over the tasks. Estimated with
statsmodels' `BinomialBayesMixedGLM` (variational Bayes; no R / lme4 on the author's machine). Which estimator the
final analysis uses is an open choice (plan O36); for the POC the posterior mean and sd give an approximate interval
and z-test. An idiom random effect can be added (`idiom_effect=True`, plan O27).

Per-model check: items a model answers correctly in only one version (b = wrong only on the idiomatic, c = wrong only
on the literal), exact McNemar (course-project test), and a bootstrap that resamples whole idioms so items sharing an
idiom stay together (Same Harm, Different Words App. B.6-B.7 make the same argument; plan O27).

Unparsable answers count as wrong, as in the course project; parse rates are reported per version.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.mixed_effects import MODEL_PARAMS_B
from analysis.stats import mcnemar_exact_p

_MIN_OUTCOMES = 2  # need both right and wrong answers to fit the model
VERSIONS = {"idiomatic": "idiomatic", "span_literal": "literal", "original": "original"}


def _idioms(data_dir: Path) -> dict[str, str]:
    with (data_dir / "idiomatic.csv").open(encoding="utf-8", newline="") as f:
        return {r["id"]: json.loads(r["meta"])["idiom"]["idiom"] for r in csv.DictReader(f)}


def load_long(
    results_dir: Path, dataset: str, data_root: Path = Path("datasets_out")
) -> pd.DataFrame:
    """One row per (model, item, version) for one dataset's result files."""
    idioms = _idioms(data_root / dataset)
    rows: list[dict[str, Any]] = []
    for path in sorted((results_dir / dataset).glob("*.json")):
        res = json.loads(path.read_text(encoding="utf-8"))
        model = str(res["model_id"])
        params = MODEL_PARAMS_B.get(model)
        if params is None:
            raise ValueError(f"no parameter count for {model!r}; add it to MODEL_PARAMS_B")
        for task in res["per_task"]:
            for key, version in VERSIONS.items():
                if key not in task:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "log_params": math.log(params),
                        "item": str(task["id"]),
                        "idiom": idioms.get(str(task["id"]), ""),
                        "version": version,
                        "correct": int(bool(task[key]["correct"])),
                        "parse_ok": task[key].get("parse_status") == "ok",
                    }
                )
    return pd.DataFrame(rows)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and parse rate per model and version (percent)."""
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(["dataset", "model", "version"]):
        ds, model, version = (str(k) for k in key) if isinstance(key, tuple) else (str(key), "", "")
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "version": version,
                "accuracy": round(float(np.mean(g["correct"].to_numpy(float))) * 100, 2),
                "parse_rate": round(float(np.mean(g["parse_ok"].to_numpy(float))) * 100, 2),
                "n": len(g),
            }
        )
    return pd.DataFrame(rows)


def _acc(d: pd.DataFrame, version: str) -> float:
    vals = d.loc[d["version"] == version, "correct"].to_numpy(float)
    return float(np.mean(vals)) * 100 if len(vals) else float("nan")


def fit_main(df: pd.DataFrame, *, idiom_effect: bool = False, seed: int = 0) -> dict[str, Any]:
    """Main test for one dataset: idiomatic vs literal, item and model random effects."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: PLC0415

    d = pd.DataFrame(df[df["version"].isin(["idiomatic", "literal"])]).copy()
    acc_lit, acc_idi = _acc(d, "literal"), _acc(d, "idiomatic")
    n_models = len(set(d["model"].tolist()))
    row: dict[str, Any] = {
        "dataset": str(d["dataset"].tolist()[0]),
        "n_rows": len(d),
        "n_items": len(set(d["item"].tolist())),
        "n_models": n_models,
        "acc_literal": round(acc_lit, 2),
        "acc_idiomatic": round(acc_idi, 2),
        "raw_gap_pp": round(acc_idi - acc_lit, 2),
        "idiom_effect": idiom_effect,
        "estimator": "statsmodels BinomialBayesMixedGLM.fit_vb (POC; O36)",
    }
    nan = float("nan")
    empty = {
        "log_odds_idiomatic": nan,
        "ci_low": nan,
        "ci_high": nan,
        "odds_ratio": nan,
        "p_approx": nan,
    }
    if len(set(d["correct"].tolist())) < _MIN_OUTCOMES:  # all right or all wrong: cannot be fitted
        return row | empty | {"note": "not estimable: all answers identical"}
    d["idiomatic"] = (d["version"] == "idiomatic").astype(float)
    d["log_params_c"] = d["log_params"] - float(np.mean(np.asarray(d["log_params"], dtype=float)))
    vc = {"item": "0 + C(item)"}
    if n_models > 1:
        vc["model"] = "0 + C(model)"
    if idiom_effect:
        vc["idiom"] = "0 + C(idiom)"
    formula = "correct ~ idiomatic + log_params_c" if n_models > 1 else "correct ~ idiomatic"
    np.random.seed(seed)
    fit: Any = BinomialBayesMixedGLM.from_formula(formula, vc, d).fit_vb()
    j = list(fit.model.exog_names).index("idiomatic")
    mean, sd = float(fit.fe_mean[j]), float(fit.fe_sd[j])
    z = mean / sd if sd > 0 else nan
    return row | {
        "log_odds_idiomatic": round(mean, 4),
        "ci_low": round(mean - 1.96 * sd, 4),
        "ci_high": round(mean + 1.96 * sd, 4),
        "odds_ratio": round(math.exp(mean), 4),
        "p_approx": float(2 * (1 - norm.cdf(abs(z)))),
        "note": "",
    }


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, same order as given; a missing (NaN) p-value stays NaN and is not counted."""
    valid = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(valid, key=lambda i: pvals[i])
    adj = [float("nan")] * len(pvals)
    running = 0.0
    m = len(valid)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def per_model(df: pd.DataFrame, *, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Per model: b, c, accuracy gap, exact McNemar, and an idiom-cluster bootstrap of the gap."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    pairs = df[df["version"].isin(["idiomatic", "literal"])]
    for key, g in pairs.groupby(["dataset", "model"]):
        ds, model = (str(k) for k in key) if isinstance(key, tuple) else (str(key), "")
        by: dict[str, dict[str, float]] = {}
        idiom_of: dict[str, str] = {}
        for item, idiom, version, correct in zip(
            g["item"].tolist(),
            g["idiom"].tolist(),
            g["version"].tolist(),
            g["correct"].tolist(),
            strict=True,
        ):
            by.setdefault(str(item), {})[str(version)] = float(correct)
            idiom_of[str(item)] = str(idiom)
        items = [k for k, v in by.items() if "idiomatic" in v and "literal" in v]
        if not items:
            continue
        idi = np.array([by[k]["idiomatic"] for k in items])
        lit = np.array([by[k]["literal"] for k in items])
        diffs = idi - lit
        labels = np.array([idiom_of[k] for k in items])
        groups = [np.flatnonzero(labels == lab) for lab in dict.fromkeys(labels.tolist())]
        boots = np.empty(n_boot)
        for n in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            boots[n] = float(np.mean(diffs[np.concatenate([groups[k] for k in pick])])) * 100
        p_boot = float(min(1.0, 2 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))))
        out.append(
            {
                "dataset": ds,
                "model": model,
                "n_items": len(items),
                "b_wrong_only_idiomatic": int(np.sum((idi == 0) & (lit == 1))),
                "c_wrong_only_literal": int(np.sum((idi == 1) & (lit == 0))),
                "gap_pp": round(float(np.mean(diffs)) * 100, 2),
                "boot_ci_low": round(float(np.percentile(boots, 2.5)), 2),
                "boot_ci_high": round(float(np.percentile(boots, 97.5)), 2),
                "p_boot": p_boot,
                "p_mcnemar": mcnemar_exact_p(idi, lit),
            }
        )
    return pd.DataFrame(out)
