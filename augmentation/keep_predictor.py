"""Rank pool items by how likely the generator is to keep them (no API calls; seconds on a laptop).

A logistic regression over frozen sentence embeddings (the BGE model used for idiom retrieval), fitted
on earlier generation outcomes (kept vs not kept). It only orders the items that are processed next;
it never decides alone that an item is dropped. `order_pool` keeps the already-processed prefix in
place, so a longer run over the ordered pool replays the earlier waves from the response cache.
"""

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]
_FOLDS = 5
_STEPS = 3000
_LR = 0.5
_L2 = 1.0


def fit_logistic(
    x: Matrix, y: Matrix, *, l2: float = _L2, steps: int = _STEPS
) -> tuple[Matrix, float]:
    """Class-balanced, L2-regularised logistic regression by gradient descent."""
    w = np.zeros(x.shape[1])
    b = 0.0
    pos = float(y.mean())
    weight = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        g = (p - y) * weight
        w -= _LR * (x.T @ g / len(y) + l2 * w / len(y))
        b -= _LR * float(g.mean())
    return w, b


def auc(y: Matrix, score: Matrix) -> float:
    """Area under the ROC curve (rank formulation)."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = float(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cross_validated_auc(x: Matrix, y: Matrix, seed: int = 0) -> float:
    """Stratified k-fold AUC, to report how well the ranking predicts before trusting it."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), dtype=int)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % _FOLDS
    scores = np.zeros(len(y))
    for k in range(_FOLDS):
        train, test = folds != k, folds == k
        w, b = fit_logistic(x[train], y[train])
        scores[test] = x[test] @ w + b
    return auc(y, scores)


def order_pool(
    ids: Sequence[str],
    texts: Sequence[str],
    labels: dict[str, bool],
    *,
    prefix: int,
    embed: Callable[[Sequence[str]], Matrix],
) -> tuple[list[str], float]:
    """Ids in run order: the first `prefix` unchanged, the rest ranked by predicted keep chance.

    Returns the order and the cross-validated AUC of the predictor on `labels`.
    """
    known = [n for n, i in enumerate(ids) if i in labels]
    x_all = embed(texts)
    y = np.array([float(labels[ids[n]]) for n in known])
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("need both kept and not-kept examples to fit the predictor")
    quality = cross_validated_auc(x_all[known], y)
    w, b = fit_logistic(x_all[known], y)
    head = [str(ids[n]) for n in range(prefix)]
    rest = list(range(prefix, len(ids)))
    scores = x_all[rest] @ w + b
    tail = [str(ids[rest[int(j)]]) for j in np.argsort(-scores, kind="stable")]
    return head + tail, quality
