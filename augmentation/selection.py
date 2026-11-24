"""Final selection of a full-study dataset from checked pairs (plan v3 O53, decided 2026-11-25).

Order: (1) drop pairs that keep < `min_kept` of the original's words (O2, decided 80%); (2) at most `cap` pairs per
idiom (O53 d, decided 5; generation already stops showing capped idioms, this is the final guarantee); (3) the same
number of pairs per label, and within a label the strata (text-length tercile of the pool, plus genre for MNLI)
in the shares they have in the pool for that label. A stratum short of pairs is filled from the other strata of the
same label; a label short of pairs is reported (how many more to generate), never filled from another label.
Deterministic for a given seed.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Pair:
    id: str
    label: str
    stratum: str  # e.g. "medium|telephone"
    idiom_id: str
    words_kept: float


@dataclass(slots=True)
class Selection:
    ids: list[str]
    complete: bool  # every label reached its quota
    report: dict[str, Any] = field(default_factory=dict[str, Any])


def length_bins(word_counts: Sequence[int]) -> tuple[int, int]:
    """Tercile cut points of a pool's word counts (as `idiom-check balance`)."""
    w = sorted(word_counts)
    return w[len(w) // 3], w[2 * len(w) // 3]


def length_bin(words: int, cuts: tuple[int, int]) -> str:
    return ("short", "medium", "long")[(words >= cuts[0]) + (words >= cuts[1])]


def allocate(total: int, shares: Mapping[str, float]) -> dict[str, int]:
    """Largest-remainder split of `total` by `shares` (shares need not sum to 1)."""
    s = sum(shares.values())
    raw = {k: total * v / s for k, v in shares.items()} if s else dict.fromkeys(shares, 0.0)
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (out[k] - raw[k], k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def select_pairs(
    pairs: Sequence[Pair],
    n: int,
    pool_shares: Mapping[str, Mapping[str, float]],
    *,
    min_kept: float = 0.80,
    cap: int = 5,
    seed: int = 0,
) -> Selection:
    """`pool_shares[label][stratum]`: share of that stratum among the pool's items with that label."""
    rng = random.Random(seed)
    order = sorted(pairs, key=lambda p: p.id)
    rng.shuffle(order)
    report: dict[str, Any] = {"checked_pairs": len(order)}
    kept = [p for p in order if p.words_kept >= min_kept]
    report["dropped_words_kept"] = len(order) - len(kept)
    uses: Counter[str] = Counter()
    capped: list[Pair] = []
    for p in kept:
        if uses[p.idiom_id] < cap:
            uses[p.idiom_id] += 1
            capped.append(p)
    report["dropped_idiom_cap"] = len(kept) - len(capped)
    by_label: dict[str, list[Pair]] = defaultdict(list)
    for p in capped:
        by_label[p.label].append(p)
    labels = sorted(pool_shares)
    quota = allocate(n, dict.fromkeys(labels, 1.0))
    chosen: list[str] = []
    per_label: dict[str, Any] = {}
    complete = True
    for lab in labels:
        avail = by_label.get(lab, [])
        want = quota[lab]
        if len(avail) < want:
            complete = False
        strata: dict[str, list[Pair]] = defaultdict(list)
        for p in avail:
            strata[p.stratum].append(p)
        target = allocate(min(want, len(avail)), pool_shares[lab])
        take = {s: min(t, len(strata.get(s, []))) for s, t in target.items()}
        missing = min(want, len(avail)) - sum(take.values())
        while missing > 0:  # fill shortfalls from the strata with the most spare pairs
            spare = {
                s: len(v) - take.get(s, 0) for s, v in strata.items() if len(v) > take.get(s, 0)
            }
            if not spare:
                break
            s = max(sorted(spare), key=lambda k: spare[k])
            take[s] = take.get(s, 0) + 1
            missing -= 1
        for s, k in take.items():
            chosen += [p.id for p in strata.get(s, [])[:k]]
        per_label[lab] = {
            "quota": want,
            "available": len(avail),
            "selected": sum(take.values()),
            "short_by": max(0, want - len(avail)),
            "strata": {
                s: {"target": target.get(s, 0), "selected": take.get(s, 0)}
                for s in sorted(set(target) | set(take))
            },
        }
    report["per_label"] = per_label
    report["selected"] = len(chosen)
    return Selection(sorted(chosen), complete, report)
