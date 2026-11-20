"""Figurative vs literal training sentences from MAGPIE for the interpretability test A (plan v3 O42, 2026-11-20).

MAGPIE (Haagsma et al., LREC 2020, §4-5): potentially idiomatic expressions in BNC / PMB sentences, each labelled by
crowdworkers as idiomatic ("i"), literal ("l"), other or unclear, with the majority label and a confidence score.
The corpus file stays local / on Drive (licence of the corpus not verified; the paper is CC-BY-NC), never in git.

Selection: confidence 1.0 (all annotators agree), labels i / l only, idiom written as listed or inflected or
re-cased (variant types `identical`, `inflection`, `case`, so the offsets mark the idiom words), idiom types with at
least `min_per_label` sentences of each use, at most `cap` per use and type (balanced), idioms of our POC pairs
excluded (the direction is built on other idioms, as IdiomSteer §3 keeps its evaluation idioms disjoint).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VARIANTS = ("identical", "inflection", "case")
TARGET = 2  # MAGPIE `context` holds five sentences; the labelled one is the middle one


def select_magpie(
    records: Iterable[dict[str, Any]],
    exclude: set[str],
    *,
    min_per_label: int = 5,
    cap: int = 10,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Balanced figurative (label 1) / literal (label 0) sentences with the idiom's character span."""
    by_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"i": [], "l": []})
    for r in records:
        if r.get("confidence") != 1.0 or r.get("label") not in ("i", "l"):
            continue
        if r.get("variant_type") not in VARIANTS or r["idiom"].lower() in exclude:
            continue
        text = r["context"][TARGET]
        offsets = r.get("offsets") or []
        if not text or not offsets:
            continue
        lo, hi = int(offsets[0][0]), int(offsets[-1][1])
        if not 0 <= lo < hi <= len(text):
            continue
        by_type[r["idiom"]][r["label"]].append(
            {"id": f"magpie-{r['id']}", "idiom": r["idiom"], "text": text, "span": [lo, hi]}
        )
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for idiom in sorted(by_type):
        uses = by_type[idiom]
        if min(len(uses["i"]), len(uses["l"])) < min_per_label:
            continue
        for lab, y in (("i", 1), ("l", 0)):
            pool = sorted(uses[lab], key=lambda d: d["id"])
            rng.shuffle(pool)
            out += [{**d, "label": y} for d in pool[:cap]]
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
