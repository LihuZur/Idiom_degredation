"""Candidate replacement tasks for MMLU (plan v3 O47, 2026-11-18): multiple-choice reasoning over a short everyday
context. Baseline check only: do the 14 panel models score in a useful range with the context, and how much
without it (memorised / answerable from the question and options alone)?

Datasets (as reported by a web search, 2026-11-18; facts not yet verified against the papers, plan rule 1):
Social IQa (context + question + 3 answers) and CosmosQA (context + question + 4 answers). Field names are
detected from the loaded rows and the run fails loudly if they do not match, instead of guessing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCES: dict[str, dict[str, str]] = {
    "siqa": {"hf": "allenai/social_i_qa", "split": "validation"},
    "cosmosqa": {"hf": "allenai/cosmos_qa", "split": "validation"},
}
_ANSWER_KEY = re.compile(r"^answer([A-E]|[0-4])$")


def answer_fields(columns: Iterable[str]) -> list[str]:
    """The answer-option columns in order: answerA.. or answer0.. (fails if there are none)."""
    found = sorted(c for c in columns if _ANSWER_KEY.match(c))
    if len(found) < 2:  # noqa: PLR2004
        raise ValueError(f"no answer columns (answerA.. / answer0..) among {sorted(columns)}")
    return found


def normalize(records: Sequence[Mapping[str, Any]], source: str) -> list[dict[str, Any]]:
    """Uniform rows: id, context, question, choices, label as a 0-based index string."""
    if not records:
        raise ValueError(f"{source}: no rows")
    cols = set(records[0])
    missing = {"context", "question", "label"} - cols
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}; found {sorted(cols)}")
    fields = answer_fields(cols)
    raw_labels = [int(str(r["label"]).strip()) for r in records]
    lo, hi = min(raw_labels), max(raw_labels)
    if lo == 1 and hi == len(fields):
        offset = 1  # 1-based labels (e.g. "1".."3")
    elif lo == 0 and hi == len(fields) - 1:
        offset = 0
    else:
        raise ValueError(f"{source}: labels {lo}..{hi} do not fit {len(fields)} options")
    out: list[dict[str, Any]] = []
    for n, (r, lab) in enumerate(zip(records, raw_labels, strict=True)):
        out.append(
            {
                "id": str(r.get("id") or stable_id(source, r["context"], r["question"], n)),
                "context": str(r["context"]).strip(),
                "question": str(r["question"]).strip(),
                "choices": [str(r[f]).strip() for f in fields],
                "label": str(lab - offset),
            }
        )
    return out


def stable_id(source: str, context: Any, question: Any, n: int) -> str:
    """Datasets without an id column: a short hash of the text (+ row number against exact duplicates)."""
    h = hashlib.sha1(f"{context}\x1e{question}".encode()).hexdigest()[:12]
    return f"{source}-{h}-{n}"


def sample_balanced(rows: Sequence[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n rows spread evenly over the gold answer positions, deterministic per seed."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    rng = random.Random(seed)
    for group in by.values():
        rng.shuffle(group)
    per = n // len(by)
    out = [r for lab in sorted(by) for r in by[lab][:per]]
    rng.shuffle(out)
    return out


def write_dataset(rows: Sequence[dict[str, Any]], out_dir: Path, *, with_context: bool) -> Path:
    """`original.csv` in the evaluation pipeline's format; `x` is the context ("" without context)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "original.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "variant", "x", "y", "meta"])
        w.writeheader()
        for r in rows:
            meta = {
                "question": r["question"],
                "choices": r["choices"],
                "context_words": len(r["context"].split()),
            }
            w.writerow(
                {
                    "id": r["id"],
                    "variant": "original",
                    "x": r["context"] if with_context else "",
                    "y": r["label"],
                    "meta": json.dumps(meta, ensure_ascii=False),
                }
            )
    return path
