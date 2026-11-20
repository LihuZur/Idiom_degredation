"""Interpretability test A (plan v3 O42, 2026-11-20): does the model read our inserted idiom figuratively, and does a
literal reading predict its errors?

Direction: mean-difference between figurative and literal activations at the idiom's last token, unit length
(IdiomSteer §4.1, "MD"), built per model and layer on MAGPIE sentences (`data/figurative.py`), in two contexts:
`bare` (the sentence alone, as IdiomSteer §4.3) and `template` (the sentence inside the model's chat template with a
neutral instruction, like our evaluation prompts).

Validity: idiom-disjoint cross-validation; how well the projection on a direction built from other idioms separates
held-out figurative from literal uses (AUC). Our choice; IdiomSteer reports cross-validated probe accuracy (App. D).

Score: projection of a POC item's activation at the idiom's last token, rescaled so that +1 is the average MAGPIE
figurative use and -1 the average literal use of the same model, layer and context.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analysis.interp import perm_diff_p


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (Mann-Whitney; ties count half)."""
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def md_direction(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit mean-difference direction per layer. acts [n, layers, hidden], labels 1 figurative / 0 literal ->
    [layers, hidden]."""
    pos = labels.astype(bool)
    diff = acts[pos].astype(np.float32).mean(axis=0) - acts[~pos].astype(np.float32).mean(axis=0)
    return diff / np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-12)


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """[n, layers, hidden] x [layers, hidden] -> [n, layers]."""
    return np.einsum("nlh,lh->nl", acts.astype(np.float32), direction)


def idiom_folds(idioms: Sequence[str], k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold number per sentence; all sentences of an idiom fall in the same fold."""
    types = sorted(set(idioms))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(types))
    fold_of = {types[i]: n % k for n, i in enumerate(order)}
    return np.array([fold_of[i] for i in idioms])


def cv_auc_by_layer(acts: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """[folds, layers] AUC of held-out projections, directions built on the other folds' idioms."""
    out = []
    for f in np.unique(folds):
        test = folds == f
        v = md_direction(acts[~test], labels[~test])
        proj = project(acts[test], v)
        out.append([auc(proj[:, layer], labels[test]) for layer in range(proj.shape[1])])
    return np.array(out)


def scale_reference(
    acts: np.ndarray, labels: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per layer: midpoint and half-gap of the figurative and literal mean projections (for the +1 / -1 scale)."""
    proj = project(acts, direction)
    pos = labels.astype(bool)
    fig, lit = proj[pos].mean(axis=0), proj[~pos].mean(axis=0)
    return (fig + lit) / 2, np.maximum((fig - lit) / 2, 1e-12)


def rescale(proj: np.ndarray, mid: np.ndarray, half: np.ndarray) -> np.ndarray:
    return (proj - mid) / half


def best_layers(validity: pd.DataFrame) -> pd.DataFrame:
    """Per model: the context and layer with the highest mean held-out AUC (chosen on MAGPIE only, never on POC)."""
    idx = validity.groupby("model")["auc_mean"].idxmax()
    return validity.loc[
        idx, ["model", "context", "layer", "depth", "auc_mean", "auc_min"]
    ].reset_index(drop=True)


def depth_table(validity: pd.DataFrame) -> pd.DataFrame:
    v = validity.copy()
    v["depth_bin"] = pd.cut(
        v["depth"],
        [0, 0.25, 0.5, 0.75, 1.0001],
        right=False,
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
    )
    table = v.pivot_table(
        index=["model", "context"],
        columns="depth_bin",
        values="auc_mean",
        aggfunc="max",
        observed=True,
    )
    return table.reset_index()


def at_best(scores: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """POC scores at each model's chosen context and layer."""
    keys = best[["model", "context", "layer"]]
    return scores.merge(keys, on=["model", "context", "layer"], how="inner")


def group_tests(chosen: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Idiom score of items wrong only with the idiom (and, as control, wrong only with the literal version) vs items
    right in both; labels permuted within model and task. Hypothesis: idiom-only errors have lower (more literal)
    scores; literal-only errors should not."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for flip in ("idiom_only_wrong", "literal_only_wrong"):
        for task in [*sorted(chosen["task"].unique()), "all"]:
            sel = chosen[chosen["group"].isin([flip, "both_right"])]
            if task != "all":
                sel = sel[sel["task"] == task]
            in_group = np.asarray(sel["group"] == flip, dtype=bool)
            if in_group.sum() == 0 or (~in_group).sum() == 0:
                continue
            vals = np.asarray(sel["score_idi"], dtype=float)
            cells = np.asarray(sel["model"].astype(str) + "|" + sel["task"].astype(str))
            out.append(
                {
                    "flip": flip,
                    "task": task,
                    "n_flip": int(in_group.sum()),
                    "n_both_right": int((~in_group).sum()),
                    "score_flip": vals[in_group].mean(),
                    "score_both_right": vals[~in_group].mean(),
                    "p_perm": perm_diff_p(vals, in_group, cells, n_perm, rng),
                }
            )
    return pd.DataFrame(out)


def score_vs_pgold(chosen: pd.DataFrame) -> pd.DataFrame:
    """Spearman of the idiom score with the p_gold drop (literal - idiomatic), per task and pooled, after centring
    both within model and task. Hypothesis: negative (more literal reading, bigger drop)."""
    d = chosen.dropna(subset=["pg_lit", "pg_idi", "score_idi"]).copy()
    d["drop"] = d["pg_lit"].astype(float) - d["pg_idi"].astype(float)
    for c in ("drop", "score_idi"):
        d[c + "_c"] = d[c] - d.groupby(["model", "task"])[c].transform("mean")
    rows: list[dict[str, Any]] = []
    for task in [*sorted(d["task"].unique()), "all"]:
        sel = cast(pd.DataFrame, d if task == "all" else d[d["task"] == task])
        x, y = cast(pd.Series, sel["score_idi_c"]), cast(pd.Series, sel["drop_c"])
        rho = float(x.corr(y, method="spearman"))
        rows.append({"task": task, "n": len(sel), "rho_score_vs_pgold_drop": rho})
    return pd.DataFrame(rows)
