"""Interpretability pilot, analysis 1 (plan v3 F1, 2026-11-20): how far apart are the model's internal states for the
idiomatic and the literal version of the same item, layer by layer, and is the gap larger on items the model gets
wrong only with the idiom?

Distances are cosine distances (1 - cosine similarity) between residual-stream vectors:
- `d_il`, `d_io`, `d_lo`: last prompt token (the state the answer is generated from), idiomatic vs literal,
  idiomatic vs original, literal vs original. `d_lo` is the rewording-only baseline: how much a literal rewording
  moves the state.
- `d_span`: mean over the idiom's tokens (idiomatic version) vs mean over the literal replacement's tokens (literal
  version): does the model represent the idiom like its literal meaning, and from which layer on?
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

GROUPS = {
    (1, 0): "idiom_only_wrong",  # literal right, idiomatic wrong
    (0, 1): "literal_only_wrong",
    (1, 1): "both_right",
    (0, 0): "both_wrong",
}
_MIN_FOR_RHO = 5


def cosine_distance_layers(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine similarity per layer; a and b are [layers, hidden]."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return 1.0 - num / np.maximum(den, 1e-12)


def layer_means(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Mean vector per layer over many [layers, hidden] arrays (all items and variants of one task and model)."""
    return np.mean(np.stack(list(vectors)), axis=0)


def item_rows(
    meta: Mapping[str, Any],
    last: dict[str, np.ndarray],
    span_idi: np.ndarray | None,
    span_lit: np.ndarray | None,
    center_last: np.ndarray | None = None,
    center_span: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """One row per layer for one item. `last` maps original/idiomatic/span_literal to [layers, hidden].

    Main columns (`d_il`, `d_io`, `d_lo`, `d_span`) are cosine distances after subtracting the per-layer mean vector
    (`center_*`, from `layer_means`): raw cosines between residual-stream states are dominated by a few directions
    shared by all inputs and are all close to 0 (seen in the 2026-11-20 smoke test). The raw values are kept as
    `*_raw`."""
    pairs = {
        "il": ("idiomatic", "span_literal"),
        "io": ("idiomatic", "original"),
        "lo": ("span_literal", "original"),
    }
    cl = center_last if center_last is not None else 0.0
    dist: dict[str, np.ndarray | None] = {}
    for name, (a, b) in pairs.items():
        dist[f"d_{name}_raw"] = cosine_distance_layers(last[a], last[b])
        dist[f"d_{name}"] = cosine_distance_layers(last[a] - cl, last[b] - cl)
    cs = center_span if center_span is not None else 0.0
    if span_idi is not None and span_lit is not None:
        dist["d_span_raw"] = cosine_distance_layers(span_idi, span_lit)
        dist["d_span"] = cosine_distance_layers(span_idi - cs, span_lit - cs)
    else:
        dist["d_span_raw"] = dist["d_span"] = None
    n_layers = len(last["idiomatic"])
    return [
        {
            **meta,
            "layer": k,
            "depth": k / max(n_layers - 1, 1),
            **{c: (float(v[k]) if v is not None else np.nan) for c, v in dist.items()},
        }
        for k in range(n_layers)
    ]


def group_of(lit_correct: int, idi_correct: int) -> str:
    return GROUPS[(int(lit_correct), int(idi_correct))]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per model, task and layer: mean distances, d_il on items wrong only with the idiom vs items right in both
    versions, and the Spearman correlation between d_il and the p_gold drop (literal - idiomatic) over items."""
    out: list[dict[str, object]] = []
    for keys, g in df.groupby(["model", "task", "layer"], sort=True):
        model, task, layer = cast(tuple[Any, Any, Any], keys)
        flip = g[g["group"] == "idiom_only_wrong"]["d_il"]
        back = g[g["group"] == "literal_only_wrong"]["d_il"]
        stable = g[g["group"] == "both_right"]["d_il"]
        drop = (g["pg_lit"] - g["pg_idi"]).astype(float)
        ok = drop.notna()
        enough = ok.sum() >= _MIN_FOR_RHO
        d_il = g.loc[ok, "d_il"].astype(float)
        rho = float(d_il.corr(drop[ok], method="spearman")) if enough else np.nan
        # a distance has no sign, so it can only predict the size of the p_gold change, not its direction
        rho_abs = float(d_il.corr(drop[ok].abs(), method="spearman")) if enough else np.nan
        out.append(
            {
                "model": model,
                "task": task,
                "layer": layer,
                "depth": float(g["depth"].iloc[0]),
                "items": len(g),
                "d_il": g["d_il"].mean(),
                "d_lo": g["d_lo"].mean(),
                "d_io": g["d_io"].mean(),
                "d_span": g["d_span"].mean(),
                "d_il_raw": g["d_il_raw"].mean(),
                "d_lo_raw": g["d_lo_raw"].mean(),
                "d_span_raw": g["d_span_raw"].mean(),
                "n_idiom_only_wrong": len(flip),
                "d_il_idiom_only_wrong": flip.mean() if len(flip) else np.nan,
                "d_il_both_right": stable.mean() if len(stable) else np.nan,
                "n_literal_only_wrong": len(back),
                "d_il_literal_only_wrong": back.mean() if len(back) else np.nan,
                "rho_d_il_vs_pgold_drop": rho,
                "rho_d_il_vs_abs_pgold_change": rho_abs,
            }
        )
    return pd.DataFrame(out)


def depth_bins(
    summary: pd.DataFrame, edges: Sequence[float] = (0, 0.25, 0.5, 0.75, 1.0001)
) -> pd.DataFrame:
    """Average the per-layer summary into relative-depth quarters, so models with different layer counts line up."""
    s = summary.copy()
    s["depth_bin"] = pd.cut(
        s["depth"], list(edges), right=False, labels=["0-25%", "25-50%", "50-75%", "75-100%"]
    )
    cols = [
        "d_il",
        "d_lo",
        "d_io",
        "d_span",
        "d_il_idiom_only_wrong",
        "d_il_both_right",
        "d_il_literal_only_wrong",
        "rho_d_il_vs_abs_pgold_change",
    ]
    means = cast(
        pd.DataFrame, s.groupby(["model", "task", "depth_bin"], observed=True)[cols].mean()
    )
    return means.reset_index()


DEPTH_LABELS = ("0-25%", "25-50%", "50-75%", "75-100%")


def item_depth_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per item: mean d_il within each relative-depth quarter, divided by the median over all items of the same model,
    task and quarter (models differ in their distance scale; 1.0 = a typical item of that model)."""
    d = df.copy()
    d["depth_bin"] = pd.cut(
        d["depth"], [0, 0.25, 0.5, 0.75, 1.0001], right=False, labels=list(DEPTH_LABELS)
    )
    keys = ["model", "task", "id", "group", "depth_bin"]
    items = cast(pd.DataFrame, d.groupby(keys, observed=True)["d_il"].mean()).reset_index()
    med = items.groupby(["model", "task", "depth_bin"], observed=True)["d_il"].transform("median")
    items["rel"] = items["d_il"] / med
    return items


def pooled_flip_test(df: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Items whose answer flipped vs items right in both versions, pooled over models: mean relative d_il of each
    group, and a permutation p-value (group labels shuffled within each model and task, so every model keeps its own
    number of flips). Run for both flip directions: if literal-only-wrong items are as far apart as idiom-only-wrong
    items, the distance only says that the answer changed, not that the idiom caused anything special."""
    rng = np.random.default_rng(seed)
    items = item_depth_means(df)
    out: list[dict[str, object]] = []
    for flip in ("idiom_only_wrong", "literal_only_wrong"):
        for task in [*sorted(items["task"].unique()), "all"]:
            for depth in DEPTH_LABELS:
                sel = items[
                    (items["depth_bin"] == depth) & items["group"].isin([flip, "both_right"])
                ]
                if task != "all":
                    sel = sel[sel["task"] == task]
                is_flip = np.asarray(sel["group"] == flip, dtype=bool)
                if is_flip.sum() == 0 or (~is_flip).sum() == 0:
                    continue
                rel = np.asarray(sel["rel"], dtype=float)
                cells = np.asarray(sel["model"].astype(str) + "|" + sel["task"].astype(str))
                observed = rel[is_flip].mean() - rel[~is_flip].mean()
                perm = np.empty(n_perm)
                groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
                for k in range(n_perm):
                    lab = is_flip.copy()
                    for idx in groups:
                        lab[idx] = rng.permutation(lab[idx])
                    perm[k] = rel[lab].mean() - rel[~lab].mean()
                out.append(
                    {
                        "flip": flip,
                        "task": task,
                        "depth_bin": depth,
                        "n_flip": int(is_flip.sum()),
                        "n_both_right": int((~is_flip).sum()),
                        "rel_flip": rel[is_flip].mean(),
                        "rel_both_right": rel[~is_flip].mean(),
                        "p_perm": float((np.abs(perm) >= abs(observed)).mean()),
                    }
                )
    return pd.DataFrame(out)
