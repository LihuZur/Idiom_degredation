"""Wikipedia word-frequency norms — RQ2 H2b predictor.

See paper_preparations/RQ2_idiom_properties.md §3. Source: pre-computed
English Wikipedia word counts (IlyaSemenov/wikipedia-word-frequency,
enwiki-2023-04-13 snapshot). Word-level only — idiom (phrase-level)
frequency is approximated as the rarest constituent content word's count,
since that word is the bottleneck for how often a reader/model has
plausibly encountered the whole phrase.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / "_files" / "frequency" / "enwiki-2023-04-13.txt"


def load_word_frequencies(path: Path = _DEFAULT_PATH) -> dict[str, int]:
    """Return {word (lowercase): article-count} from the Wikipedia frequency list."""
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            word, _, count_str = line.strip().rpartition(" ")
            if not word:
                continue
            out[word.lower()] = int(count_str)
    return out


def idiom_frequency(idiom: str, word_freq: dict[str, int]) -> int | None:
    """Approximate an idiom's frequency as its rarest constituent word's count.

    Returns None if any alphabetic word in the idiom is missing from
    `word_freq` (e.g. a typo or a word not observed in the Wikipedia
    snapshot) — callers should treat that idiom as uncovered, same as an
    opacity-lookup miss.
    """
    # Hyphens are split like whitespace so hyphenated compounds (e.g.
    # "run-of-the-mill", "top-notch") tokenize into their constituent words
    # instead of failing `.isalpha()` as one hyphenated blob (found
    # 2026-09-16 — this was silently dropping ~10% of sst2's opacity-covered
    # rows from both H2a and H2b).
    words = [w.lower() for w in idiom.replace("-", " ").split() if w.isalpha()]
    if not words:
        return None
    counts = [word_freq.get(w) for w in words]
    if any(c is None for c in counts):
        return None
    return min(c for c in counts if c is not None)
