"""Keep predictor and ordered pools (local, no API calls)."""

import numpy as np

from augmentation.keep_predictor import auc, cross_validated_auc, order_pool


def _embed(texts):  # type: ignore[no-untyped-def]
    # one informative feature: texts containing "gave up" are the ones that get kept
    return np.array([[1.0 if "gave up" in t else 0.0, 1.0] for t in texts])


def test_auc_perfect_and_reversed() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_order_pool_keeps_prefix_and_ranks_the_rest() -> None:
    texts = ["they gave up", "a fact", "she gave up", "another fact", "he gave up", "a date"]
    ids = [f"i{n}" for n in range(len(texts))]
    labels = {"i0": True, "i1": False, "i2": True, "i3": False}
    order, quality = order_pool(ids, texts, labels, prefix=2, embed=_embed)
    assert order[:2] == ["i0", "i1"]  # processed prefix unchanged
    assert sorted(order) == sorted(ids)
    assert order.index("i2") < order.index("i3")  # likely keepers first
    assert order.index("i4") < order.index("i5")
    assert 0.0 <= quality <= 1.0


def test_cross_validated_auc_on_separable_data() -> None:
    x = np.array([[float(n % 2), 1.0] for n in range(40)])
    y = np.array([float(n % 2) for n in range(40)])
    assert cross_validated_auc(x, y) > 0.9
