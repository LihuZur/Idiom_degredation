"""Idiom opacity/decomposability norms — RQ2 H2a predictor.

See paper_preparations/RQ2_idiom_properties.md §3. Titone & Connine (1994)
itself has no verifiable public download (print journal appendix only), so
two real, verified substitutes are used instead:

- Primary: Bulkes & Tanner (2017), 870 English idioms, "Global
  Decomposability" (proportion of raters, 0-1, higher = more decomposable /
  less opaque).
- Secondary (fills gaps only): Libben & Titone (2008) — Titone's own
  follow-up to the original norming study, 210 idioms, "Global
  Decomposability Proportion" (0-1, higher = more decomposable).

Both are inverted here so the returned score is *opacity* (0-1, higher =
more opaque), matching H2a's sign convention (opacity coefficient < 0 =>
more opaque idioms hurt more).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import openpyxl

_DEFAULT_BULKES_TANNER_PATH = (
    Path(__file__).parent / "_files" / "opacity" / "bulkes_tanner_2017.xlsx"
)
_DEFAULT_LIBBEN_TITONE_PATH = (
    Path(__file__).parent / "_files" / "opacity" / "libben_titone_2008" / "LibbenTitone-Norms.txt"
)

OpacitySource = Literal[
    "bulkes_tanner_2017",
    "libben_titone_2008",
    "bulkes_tanner_2017_fuzzy",
    "libben_titone_2008_fuzzy",
    "llm_rated",
]

# Dropped when comparing surface forms so e.g. "steer clear of" lines up with
# the norms' citation form "be clear of something" — see `content_words`.
_FUZZY_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "your",
        "someone",
        "something",
        "one's",
        "be",
        "is",
        "are",
        "it",
        "its",
    }
)


@dataclass(frozen=True, slots=True)
class OpacityEntry:
    """One idiom's opacity score (0-1, higher = more opaque) and its source."""

    opacity: float
    source: OpacitySource


def content_words(idiom: str) -> frozenset[str]:
    """Stopword-stripped, lowercased word set used for fuzzy norms matching."""
    return frozenset(w for w in re.findall(r"[a-z']+", idiom.lower()) if w not in _FUZZY_STOPWORDS)


_OPACITY_TOLERANCE = 1e-9


def build_fuzzy_opacity_index(
    opacity_table: dict[str, OpacityEntry],
) -> dict[frozenset[str], OpacityEntry]:
    """Index `opacity_table` by content-word set, for idioms with no exact match.

    Content-word sets that collide across two entries with a *different*
    opacity score are dropped rather than guessed at, since we can't tell
    which one a stripped-down extracted phrase actually meant.
    """
    by_words: dict[frozenset[str], OpacityEntry] = {}
    ambiguous: set[frozenset[str]] = set()
    for idiom, entry in opacity_table.items():
        words = content_words(idiom)
        if not words:
            continue
        existing = by_words.get(words)
        if existing is None:
            by_words[words] = entry
        elif abs(existing.opacity - entry.opacity) > _OPACITY_TOLERANCE:
            ambiguous.add(words)
    for words in ambiguous:
        del by_words[words]
    fuzzy_source: dict[OpacitySource, OpacitySource] = {
        "bulkes_tanner_2017": "bulkes_tanner_2017_fuzzy",
        "libben_titone_2008": "libben_titone_2008_fuzzy",
    }
    return {
        words: OpacityEntry(opacity=entry.opacity, source=fuzzy_source[entry.source])
        for words, entry in by_words.items()
    }


def _load_bulkes_tanner(path: Path) -> dict[str, float]:
    """Return {idiom (lowercase): opacity} from the GLOBAL DECOMPOSABILITY sheet.

    That sheet's columns are (idiom, n, "PROPORTION RATED AS DECOMPOSABLE
    (%)") — despite the "(%)" label, values are already 0-1 proportions
    (e.g. 0.6804), not 0-100 percentages.
    """
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["GLOBAL DECOMPOSABILITY"]
    out: dict[str, float] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):  # row 1 is the header
        idiom, _n, proportion_decomposable = row[0], row[1], row[2]
        if idiom is None or not isinstance(proportion_decomposable, (int, float)):
            continue
        out[str(idiom).strip().lower()] = 1.0 - float(proportion_decomposable)
    return out


def _load_libben_titone(path: Path) -> dict[str, float]:
    """Return {idiom (lowercase): opacity} from `Global Decomposability Proportion`."""
    out: dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idiom = row["Idiom"].strip().lower()
            proportion = float(row["Global Decomposability Proportion"])
            out[idiom] = 1.0 - proportion
    return out


def load_opacity_norms(
    bulkes_tanner_path: Path = _DEFAULT_BULKES_TANNER_PATH,
    libben_titone_path: Path = _DEFAULT_LIBBEN_TITONE_PATH,
) -> dict[str, OpacityEntry]:
    """Merge both opacity sources, preferring Bulkes & Tanner (larger, n=870).

    Libben & Titone only fills idioms Bulkes & Tanner doesn't cover. Both raw
    tables stay on disk regardless, so the plan's sensitivity check (§5.6,
    refit dropping author/secondary-sourced rows) can re-derive an
    overlap-only subset later.
    """
    bulkes_tanner = _load_bulkes_tanner(bulkes_tanner_path)
    libben_titone = _load_libben_titone(libben_titone_path)
    merged: dict[str, OpacityEntry] = {
        idiom: OpacityEntry(opacity=score, source="bulkes_tanner_2017")
        for idiom, score in bulkes_tanner.items()
    }
    for idiom, score in libben_titone.items():
        if idiom not in merged:
            merged[idiom] = OpacityEntry(opacity=score, source="libben_titone_2008")
    return merged
