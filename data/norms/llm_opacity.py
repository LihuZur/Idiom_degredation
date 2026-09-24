"""Persistent LLM-rated opacity fallback (RQ2 plan §3/§8 gap-fill).

`data/norms/opacity.py`'s two human norm sources (Bulkes & Tanner 2017,
Libben & Titone 2008) only cover ~1,080 idiom types total, in citation form.
Real idioms extracted from dataset text (`analysis/idiom_annotation.py`)
mostly aren't in either list, even after fuzzy matching
(`build_fuzzy_opacity_index`). `analysis/opacity_rating.py` fills that gap by
asking an LLM to rate the remaining idioms directly, once per idiom type,
and this module persists + reloads those ratings so the cost is paid once
across all datasets, not once per dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path

from data.norms.opacity import OpacityEntry

_DEFAULT_PATH = Path(__file__).parent / "_files" / "opacity" / "llm_ratings.csv"

_FIELDNAMES = ["idiom", "opacity", "model"]


def load_llm_opacity_ratings(path: Path = _DEFAULT_PATH) -> dict[str, OpacityEntry]:
    """Load the persistent LLM-rated opacity cache (idiom -> `OpacityEntry`, source='llm_rated').

    Returns an empty dict if the cache file doesn't exist yet (no ratings
    collected so far), rather than raising.
    """
    if not path.exists():
        return {}
    out: dict[str, OpacityEntry] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idiom = row["idiom"].strip().lower()
            out[idiom] = OpacityEntry(opacity=float(row["opacity"]), source="llm_rated")
    return out


def append_llm_opacity_ratings(
    ratings: dict[str, float], *, model: str, path: Path = _DEFAULT_PATH
) -> None:
    """Append newly rated `{idiom: opacity}` pairs to the persistent cache CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for idiom, opacity in ratings.items():
            writer.writerow({"idiom": idiom, "opacity": opacity, "model": model})
