"""Candidate SST-2 replacements (plan v3 O40, advisor note F9, 2026-11-19): baseline check only, no idioms.

DynaSent (rounds 1 and 2) and BESSTIE, as reported by a web search (`local/sst2_replacement_search.txt`; facts not yet
verified against the papers). Only positive / negative items are kept (a neutral class could be broken by an inserted
evaluative idiom). Column names are discovered from the loaded data; the run stops with the actual columns if they
cannot be mapped, instead of guessing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TEXT_COLUMNS = ("sentence", "text", "review", "comment", "content")
LABEL_COLUMNS = ("gold_label", "sentiment", "label", "polarity")
SOURCE_COLUMNS = ("source", "domain", "platform", "task", "dataset")
POSITIVE = {"positive", "pos", "1", "1.0"}
NEGATIVE = {"negative", "neg", "0", "0.0"}


def pick(columns: Sequence[str], wanted: Sequence[str], what: str) -> str:
    """First of `wanted` present in `columns` (case-insensitive); fails with the real column list."""
    low = {c.lower(): c for c in columns}
    for w in wanted:
        if w in low:
            return low[w]
    raise ValueError(f"no {what} column among {sorted(columns)} (looked for {list(wanted)})")


def binary_label(value: Any) -> str | None:
    """'1' positive, '0' negative, None for anything else (neutral, mixed, missing)."""
    v = str(value).strip().lower()
    if v in POSITIVE:
        return "1"
    if v in NEGATIVE:
        return "0"
    return None


def normalize(records: Sequence[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    """Uniform binary rows: id, text, label ('0' negative / '1' positive), and the source column if any."""
    if not records:
        raise ValueError(f"{source}: no rows")
    cols = list(records[0])
    tcol, lcol = pick(cols, TEXT_COLUMNS, "text"), pick(cols, LABEL_COLUMNS, "label")
    scol = next((c for c in cols if c.lower() in SOURCE_COLUMNS), None)
    out: list[dict[str, Any]] = []
    for n, r in enumerate(records):
        y = binary_label(r[lcol])
        text = str(r[tcol] or "").strip()
        if y is None or not text:
            continue
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        out.append(
            {
                "id": f"{source}-{h}-{n}",
                "text": text,
                "label": y,
                "source": str(r[scol]) if scol else "",
            }
        )
    if not out:
        raise ValueError(
            f"{source}: no positive/negative rows (label values: {sorted({str(r[lcol]) for r in records})[:10]})"
        )
    return out


def sample_balanced(rows: Sequence[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n rows, half positive and half negative, deterministic per seed."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n // 2, len(pos), len(neg))
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def write_dataset(rows: Sequence[dict[str, Any]], out_dir: Path) -> Path:
    """`original.csv` in the evaluation pipeline's format (same fields as SST-2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "original.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "variant", "x", "y", "meta"])
        w.writeheader()
        for r in rows:
            meta = {"source": r["source"], "words": len(r["text"].split())}
            w.writerow(
                {
                    "id": r["id"],
                    "variant": "original",
                    "x": r["text"],
                    "y": r["label"],
                    "meta": json.dumps(meta),
                }
            )
    return path
