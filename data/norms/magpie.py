"""MAGPIE corpus loader (Haagsma et al. 2020) — idiom inventory + ambiguity signal.

RQ2 (paper_preparations/RQ2_idiom_properties.md §3) uses this to sanity-check
the LLM idiom extractor's output against a real, attested idiom list, and as
a fallback ambiguity proxy for idioms absent from the opacity norms
(`data/norms/opacity.py`).

MAGPIE does *not* contain a continuous opacity/decomposability score — each
instance is a per-sentence literal/idiomatic/figurative/other usage label
(confirmed against the real corpus; the plan's original assumption of an
`idiomaticity_score` field was wrong). What it does give, aggregated per
idiom type, is how often that idiom's usage was actually judged idiomatic
vs. literal in real text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).parent / "_files" / "magpie" / "MAGPIE_unfiltered.jsonl"


@dataclass(frozen=True, slots=True)
class MagpieStats:
    """Aggregated MAGPIE statistics for one idiom type."""

    instance_count: int
    idiomatic_fraction: float  # share of instances labeled "i" (idiomatic)
    mean_confidence: float


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def load_magpie_stats(path: Path = _DEFAULT_PATH) -> dict[str, MagpieStats]:
    """Load per-idiom-type stats from the MAGPIE corpus.

    Keys are idiom surface forms lowercased (MAGPIE's `idiom` field, as-is
    otherwise) — look up with `extracted_idiom.lower()`.
    """
    counts: dict[str, int] = defaultdict(int)
    idiomatic_counts: dict[str, int] = defaultdict(int)
    confidence_sums: dict[str, float] = defaultdict(float)
    for record in _iter_records(path):
        idiom = str(record["idiom"]).lower()
        counts[idiom] += 1
        if record["label"] == "i":
            idiomatic_counts[idiom] += 1
        confidence_sums[idiom] += float(record["confidence"])
    return {
        idiom: MagpieStats(
            instance_count=n,
            idiomatic_fraction=idiomatic_counts[idiom] / n,
            mean_confidence=confidence_sums[idiom] / n,
        )
        for idiom, n in counts.items()
    }
