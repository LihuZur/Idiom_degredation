"""Find inventory idioms inside running text and read IMPLI idiom-to-gloss pairs.

Used for the natural-idiom check (rows that already contain an inventory idiom) and for the
IMPLI gloss seed list. See paper_preparations/THESIS_PLAN_v2.md §0b and §3.1.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from data.norms.inventory import stem_word

_POSS_SLOTS = frozenset({"one's", "someone's", "somebody's"})
_REFL_SLOTS = frozenset({"oneself", "yourself", "myself"})
_ANY_SLOTS = frozenset({"someone", "somebody", "something", "one"})
_POSS_WORDS = frozenset({"my", "your", "his", "her", "its", "our", "their", "one's"})
_REFL_WORDS = frozenset(
    {"myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves", "oneself"}
)
_FUNCTION_WORDS = frozenset({"the", "a", "an", "to", "of", "in", "on", "at", "for", "and", "or"})
_MIN_TOKENS = 3
_MIN_CONTENT = 2
_MAX_ANY = 3
_PAIR_COLS = 2

Pattern = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class IdiomRef:
    idiom: str
    midas_ids: str
    definition: str


@dataclass(frozen=True, slots=True)
class Hit:
    ref: IdiomRef
    form: str
    start: int
    end: int


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; a detached possessive ("britain 's") is re-attached."""
    text = re.sub(r"\s+'s\b", "'s", text.lower().replace("\u2019", "'"))
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def compile_form(
    form: str, *, min_tokens: int = _MIN_TOKENS, min_content: int = _MIN_CONTENT
) -> Pattern | None:
    """Token pattern for one MIDAS form, or None if it is too short to match reliably."""
    toks = tokenize(form)
    pattern: list[tuple[str, str]] = []
    for t in toks:
        if t in _POSS_SLOTS:
            pattern.append(("poss", ""))
        elif t in _REFL_SLOTS:
            pattern.append(("refl", ""))
        elif t in _ANY_SLOTS:
            pattern.append(("any", ""))
        else:
            pattern.append(("lit", stem_word(t)))
    literal = [v for k, v in pattern if k == "lit"]
    content = [v for v in literal if v not in {stem_word(w) for w in _FUNCTION_WORDS}]
    if len(toks) < min_tokens or len(content) < min_content:
        return None
    return tuple(pattern)


def _match(pat: Pattern, k: int, toks: list[str], stems: list[str], i: int) -> int | None:
    if k == len(pat):
        return i
    kind, val = pat[k]
    if kind in ("any", "poss"):
        for n in range(1, _MAX_ANY + 1):
            if kind == "poss" and not _is_possessive(toks, i + n - 1):
                continue
            end = _match(pat, k + 1, toks, stems, i + n)
            if end is not None:
                return end
        return None
    if i >= len(toks):
        return None
    ok = stems[i] == val if kind == "lit" else toks[i] in _REFL_WORDS
    return _match(pat, k + 1, toks, stems, i + 1) if ok else None


def _is_possessive(toks: list[str], i: int) -> bool:
    return i < len(toks) and (toks[i] in _POSS_WORDS or toks[i].endswith("'s"))


class IdiomIndex:
    """Index of compiled idiom forms for fast scanning of many texts."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        min_tokens: int = _MIN_TOKENS,
        min_content: int = _MIN_CONTENT,
    ) -> None:
        self._by_first: dict[str, list[tuple[IdiomRef, str, Pattern]]] = defaultdict(list)
        self._wild: list[tuple[IdiomRef, str, Pattern]] = []
        for r in rows:
            defs: list[str] = json.loads(r["definitions"])
            ref = IdiomRef(r["norm_idiom"], r["midas_ids"], " / ".join(defs))
            for form in json.loads(r["midas_forms"]):
                pat = compile_form(form, min_tokens=min_tokens, min_content=min_content)
                if pat is None:
                    continue
                entry = (ref, form, pat)
                kind, val = pat[0]
                if kind == "lit":
                    self._by_first[val].append(entry)
                else:
                    self._wild.append(entry)

    def find(self, text: str) -> list[Hit]:
        toks = tokenize(text)
        stems = [stem_word(t) for t in toks]
        hits: list[Hit] = []
        for i in range(len(toks)):
            for ref, form, pat in [*self._by_first.get(stems[i], []), *self._wild]:
                end = _match(pat, 0, toks, stems, i)
                if end is not None:
                    hits.append(Hit(ref, form, i, end))
        hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
        seen: set[str] = set()
        unique: list[Hit] = []
        for h in hits:
            if h.ref.idiom not in seen:
                seen.add(h.ref.idiom)
                unique.append(h)
        return unique


def gloss_for(idiom_sentence: str, literal_sentence: str, hit: Hit) -> str | None:
    """Words of the literal sentence that replace the idiom's token span, via a token diff."""
    a = tokenize(idiom_sentence)
    b = tokenize(literal_sentence)
    pieces: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        overlaps = (i1 < hit.end and i2 > hit.start) or (i1 == i2 and hit.start <= i1 <= hit.end)
        if tag != "equal" and overlaps:
            pieces.extend(b[j1:j2])
    return " ".join(pieces) if pieces else None


def read_pairs(path: Path) -> list[tuple[str, str]]:
    """(idiom sentence, literal sentence) pairs from an IMPLI TSV."""
    with path.open(encoding="utf-8", newline="") as f:
        return [
            (r[0], r[1])
            for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            if len(r) >= _PAIR_COLS
        ]
