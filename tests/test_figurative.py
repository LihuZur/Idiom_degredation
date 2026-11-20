import numpy as np
import pandas as pd
import pytest

from analysis.figurative import (
    at_best,
    auc,
    best_layers,
    cv_auc_by_layer,
    group_tests,
    idiom_folds,
    md_direction,
    project,
    rescale,
    scale_reference,
    score_vs_pgold,
)
from data.figurative import select_magpie


def _rec(i: int, idiom: str, label: str, conf: float = 1.0, variant: str = "identical"):
    text = f"They said {idiom} again ."
    lo = len("They said ")
    return {
        "id": i,
        "idiom": idiom,
        "label": label,
        "confidence": conf,
        "variant_type": variant,
        "context": ["", "", text, "", ""],
        "offsets": [[lo, lo + len(idiom)]],
    }


def test_select_magpie_filters_and_balances():
    recs = [_rec(n, "break the ice", "i") for n in range(8)] + [
        _rec(100 + n, "break the ice", "l") for n in range(6)
    ]
    recs += [_rec(200 + n, "spill the beans", "i") for n in range(9)]  # never literal: dropped
    recs += [_rec(300, "break the ice", "l", conf=0.67), _rec(301, "break the ice", "o")]
    recs += [
        _rec(400 + n, "call it a day", lab) for n in range(6) for lab in "il"
    ]  # excluded (POC idiom)
    rows = select_magpie(recs, {"call it a day"}, min_per_label=5, cap=5)
    assert {r["idiom"] for r in rows} == {"break the ice"}
    assert sum(r["label"] for r in rows) == 5 and len(rows) == 10
    r = rows[0]
    assert r["text"][r["span"][0] : r["span"][1]] == "break the ice"


def test_auc():
    assert auc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0])) == 1.0
    assert auc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1, 1, 0, 0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1, 0])) == 0.5


def _toy(
    n_types: int = 10, per: int = 6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    acts: list[np.ndarray] = []
    labels: list[int] = []
    idioms: list[str] = []
    for t in range(n_types):
        for y in (0, 1):
            for _ in range(per):
                a = rng.normal(size=(3, 8))
                a[2, 0] += 4.0 * y  # only layer 2 carries the distinction
                acts.append(a)
                labels.append(y)
                idioms.append(f"idiom{t}")
    return np.array(acts), np.array(labels), idioms


def test_cv_finds_the_informative_layer():
    acts, labels, idioms = _toy()
    folds = idiom_folds(idioms, k=5)
    for t in set(idioms):  # idiom-disjoint
        assert len({f for f, i in zip(folds, idioms, strict=True) if i == t}) == 1
    cv = cv_auc_by_layer(acts, labels, folds)
    assert cv.shape == (5, 3)
    assert cv[:, 2].mean() > 0.95
    assert cv[:, 0].mean() < 0.8


def test_rescale_maps_class_means_to_plus_minus_one():
    acts, labels, _ = _toy()
    v = md_direction(acts, labels)
    mid, half = scale_reference(acts, labels, v)
    s = rescale(project(acts, v), mid, half)
    assert s[labels == 1, 2].mean() == pytest.approx(1.0)
    assert s[labels == 0, 2].mean() == pytest.approx(-1.0)


def test_summary_pieces():
    validity = pd.DataFrame(
        [
            {
                "model": "m",
                "context": c,
                "layer": k,
                "depth": k / 2,
                "auc_mean": a,
                "auc_min": a - 0.1,
            }
            for c, aucs in (("bare", (0.5, 0.7, 0.6)), ("template", (0.5, 0.9, 0.8)))
            for k, a in enumerate(aucs)
        ]
    )
    best = best_layers(validity)
    assert best.iloc[0]["context"] == "template" and best.iloc[0]["layer"] == 1
    rows = []
    for n in range(20):
        group = "idiom_only_wrong" if n < 4 else ("literal_only_wrong" if n < 7 else "both_right")
        score = -1.0 if group == "idiom_only_wrong" else 1.0
        for c in ("bare", "template"):
            for k in range(3):
                rows.append(
                    {
                        "model": "m",
                        "task": "t",
                        "id": str(n),
                        "group": group,
                        "context": c,
                        "layer": k,
                        "score_idi": score + 0.01 * n,
                        "score_lit": 0.0,
                        "pg_lit": 0.9,
                        "pg_idi": 0.9 - (0.3 if score < 0 else 0.0) + 0.001 * n,
                    }
                )
    chosen = at_best(pd.DataFrame(rows), best)
    assert len(chosen) == 20
    g = group_tests(chosen, n_perm=500)
    idi = g[(g["flip"] == "idiom_only_wrong") & (g["task"] == "all")].iloc[0]
    assert idi["score_flip"] < idi["score_both_right"] and idi["p_perm"] < 0.05
    rho = score_vs_pgold(chosen)
    assert float(rho.set_index("task").loc["all", "rho_score_vs_pgold_drop"]) < 0
