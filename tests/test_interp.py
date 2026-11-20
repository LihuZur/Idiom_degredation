import numpy as np
import pandas as pd
import pytest

from analysis.interp import (
    cosine_distance_layers,
    depth_bins,
    group_of,
    item_rows,
    layer_means,
    pooled_flip_test,
    summarize,
)
from models.decoder_runner import span_token_indices


def test_cosine_distance_layers():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    b = np.array([[1.0, 0.0], [-1.0, -1.0], [0.0, 3.0]])
    assert cosine_distance_layers(a, b) == pytest.approx([0.0, 2.0, 1.0])


def test_span_token_indices_skips_special_and_partial_overlap():
    offsets = [(0, 0), (0, 5), (5, 6), (6, 11), (11, 17), (0, 0)]
    assert span_token_indices(offsets, 6, 11) == [3]
    assert span_token_indices(offsets, 4, 12) == [1, 2, 3, 4]
    assert span_token_indices(offsets, 20, 25) == []


def test_group_of():
    assert group_of(1, 0) == "idiom_only_wrong"
    assert group_of(0, 1) == "literal_only_wrong"
    assert group_of(1, 1) == "both_right"


def _rows(item: str, group: str, shift: float, pg: tuple[float, float]):
    base = np.ones((3, 4))
    last = {"original": base, "span_literal": base + 0.1, "idiomatic": base + np.eye(3, 4) * shift}
    meta = {"model": "m", "task": "t", "id": item, "group": group, "pg_lit": pg[0], "pg_idi": pg[1]}
    return item_rows(meta, last, base, base * 2)


def test_item_rows_and_summary():
    rows = []
    for k in range(6):
        g = "idiom_only_wrong" if k < 2 else "both_right"
        rows += _rows(str(k), g, shift=float(k + 1), pg=(0.9, 0.9 - 0.1 * k))
    df = pd.DataFrame(rows)
    assert len(df) == 18 and set(df["layer"]) == {0, 1, 2}
    assert df["d_span"].max() == pytest.approx(0.0)  # parallel vectors
    assert df["depth"].max() == pytest.approx(1.0)
    s = summarize(df)
    assert list(s["n_idiom_only_wrong"]) == [2, 2, 2]
    # bigger shift -> bigger distance and bigger p_gold drop: positive rank correlation
    assert (s["rho_d_il_vs_pgold_drop"] > 0).all()
    d = depth_bins(s)
    assert set(d["depth_bin"].astype(str)) <= {"0-25%", "25-50%", "50-75%", "75-100%"}


def test_item_rows_without_span():
    base = np.ones((2, 3))
    rows = item_rows(
        {"id": "x"}, {"original": base, "span_literal": base, "idiomatic": base}, None, None
    )
    assert all(np.isnan(float(str(r["d_span"]))) for r in rows)


def test_centering_removes_shared_direction():
    shared = np.array([[100.0, 0.0, 0.0]])
    a = shared + np.array([[0.0, 1.0, 0.0]])
    b = shared + np.array([[0.0, -1.0, 0.0]])
    o = shared + np.array([[0.0, 0.0, 1.0]])
    last = {"idiomatic": a, "span_literal": b, "original": o}
    center = layer_means([a, b, o])
    (row,) = item_rows({}, last, None, None, center, None)
    assert float(str(row["d_il_raw"])) < 0.001  # raw: almost identical
    assert float(str(row["d_il"])) > 1.5  # centred: opposite directions


def test_pooled_flip_test_detects_far_flips():
    rows = []
    for k in range(12):
        g = "idiom_only_wrong" if k < 3 else ("literal_only_wrong" if k < 5 else "both_right")
        far = 10.0 if g == "idiom_only_wrong" else 1.0
        for layer in range(4):
            rows.append(
                {
                    "model": "m",
                    "task": "t",
                    "id": str(k),
                    "group": g,
                    "depth": layer / 3,
                    "d_il": far,
                }
            )
    res = pooled_flip_test(pd.DataFrame(rows), n_perm=500)
    idi = res[(res["flip"] == "idiom_only_wrong") & (res["task"] == "all")]
    lit = res[(res["flip"] == "literal_only_wrong") & (res["task"] == "all")]
    assert (idi["rel_flip"] > idi["rel_both_right"]).all() and (idi["p_perm"] < 0.05).all()
    assert (lit["p_perm"] > 0.05).all()
