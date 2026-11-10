"""v2 generation: one idiom from a closed list replaces one span; a literal gloss replaces the same span.

THESIS_PLAN_v2 §3. The model only chooses the idiom and returns the span, the idiom text and the literal
text; the two rewritten variants are built in code by substituting into the original, so they are
identical outside the span by construction and every sample records its idiom, span, definition and
source (Kai's point 4a-c).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from augmentation.cache import ResponseCache
from augmentation.idiom_retrieval import IdiomRetriever, SpanProposal, parse_proposals
from augmentation.prompts.loader import load_prompt, render_context
from augmentation.providers.base import EmptyResponseError, LLMClient, LLMError
from data.norms.idiom_text import IdiomIndex
from data.norms.inventory import MidasIdiom, content_key, surface_form

# Frozen 2026-10-18 after the step-5 pilots: v2 kept many meaning shifts, v3 was over-strict, v4 still
# rejected fits a reader would accept, v6 (looser strength rule, 20 idioms) let near-fits through.
TEMPLATE_FILE = "idiom_span_v5.txt"
# Templates that list every MIDAS sense of an idiom; older ones showed only the first sense, while
# retrieval always embedded all senses (found 2026-10-21: 188 of 961 idioms have several senses).
ALL_SENSES_TEMPLATES = frozenset({"idiom_span_v8.txt"})
# Call 1 default since 2026-10-22: the screening prompt (lists only phrases an idiom could replace,
# [] otherwise; on 350 items with known outcomes it skipped 24% and lost 2 of 49 good rows).
SPANS_TEMPLATE_FILE = "idiom_spans_v5.txt"
AG_LABEL_DOMAIN = {
    "World": "war-politics",
    "Sports": "sports",
    "Business": "finance-business",
    "Sci/Tech": "science-tech",
}
NEUTRAL_DOMAIN = "everyday"
_MAX_SPAN_WORDS = 12
_MAX_REPLACEMENT_WORDS = 16
# Window mode (prompt v7): the model may rewrite one short window so the idiom fits naturally.
_MAX_WINDOW_WORDS = 15
_MAX_WINDOW_REPLACEMENT_WORDS = 22
_EXTRA_DROPPED_WORDS = 1
_CONTEXT_WORDS = 3
_RETRIES = 3
_N_TEXTS = 3
_BACKOFF_S = 2.0
_TERCILES = 3
_MAX_EXAMPLE_CHARS = 140

DATASET_TEXT: dict[str, tuple[str, str]] = {
    "sst2": (
        "The text is a movie-review sentence.",
        "The sentiment of the text (label: {label}) must stay exactly the same, neither stronger nor weaker.",
    ),
    "mnli": (
        "The text is the premise of a natural language inference pair. The hypothesis shown below is "
        "never changed.",
        "The label ({label}) must stay the same: the text must keep exactly the same logical "
        "relationship to the hypothesis.",
    ),
    "mmlu": (
        "The text is the question stem of a multiple-choice question. The answer choices shown below "
        "are never changed.",
        "The correct answer must not change and must not be hinted at; do not touch technical terms, "
        "numbers or names.",
    ),
    "agnews": (
        "The text is a news snippet (title followed by description).",
        "The topic of the snippet (category: {label}) must stay the same for a human reader.",
    ),
}
_SST2_LABELS = {"1": "positive", "0": "negative"}


@dataclass(frozen=True, slots=True)
class Idiom:
    idiom_id: str
    midas_ids: str
    name: str
    forms: tuple[str, ...]
    definitions: tuple[str, ...]
    example: str
    opacity: float | None  # None for the unnormed MAGPIE extension tier
    familiarity: float | None
    log_freq: float | None
    norm_source: str
    domain: str | None
    impli_gloss: str | None


@dataclass(frozen=True, slots=True)
class Item:
    dataset: str
    id: str
    x: str
    y: str
    meta: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Parsed:
    ok: bool
    reason: str
    choice: str = ""
    span: str = ""
    idiom_text: str = ""
    literal_text: str = ""
    idiom_phrase: str = ""  # window mode: the idiom as used inside `idiom_text`
    literal_phrase: str = ""  # window mode: its literal explanation, swapped in by code


_AN_EXCEPTIONS = ("hour", "honest", "honor", "honour", "heir")
_A_EXCEPTIONS = ("uni", "use", "usu", "eu", "one", "once", "ewe", "ufo")


def _article_for(word: str) -> str:
    """ "a" or "an" by the usual spelling heuristic (vowel sound), with common exceptions."""
    w = word.lower().lstrip("\"'(")
    if w.startswith(_AN_EXCEPTIONS):
        return "an"
    if w.startswith(_A_EXCEPTIONS):
        return "a"
    return "an" if w[:1] in "aeiou" else "a"


def _article_mismatch(x: str, span: str, *replacements: str) -> bool:
    """True if "a"/"an" right before the span agreed with the span but not with a replacement
    ("a living, breathing" -> "a alive and kicking")."""
    before = x[: x.index(span)].split()
    if not before or before[-1].lower() not in ("a", "an") or not span.split():
        return False
    article = before[-1].lower()
    if _article_for(span.split(maxsplit=1)[0]) != article:
        return False  # the original itself does not follow the heuristic; do not judge
    return any(r.split() and _article_for(r.split()[0]) != article for r in replacements)


# Public names for use by other modules (the v3 insertion generator).
article_for = _article_for
article_mismatch = _article_mismatch


def _rewritten_too_much(window: str, idiomatic_window: str, idiom_phrase: str) -> bool:
    """True if the model dropped more of the window's words than the idiom can account for.

    The idiom may replace as many words as it has, plus one; any further dropped word means the
    window was rewritten beyond what the idiom needs (window mode).
    """
    rest = idiomatic_window.replace(idiom_phrase, " ", 1).lower().split()
    dropped = sum(w not in rest for w in window.lower().split())
    return dropped > len(idiom_phrase.split()) + _EXTRA_DROPPED_WORDS


@dataclass(frozen=True, slots=True)
class Prepared:
    item: Item
    shortlist: tuple[Idiom, ...]
    condition: str
    prompt: str
    attempt: int = 0
    mode: str = "random"
    suggested: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Outcome:
    item: Item
    ok: bool
    reason: str
    raw: str
    record: dict[str, Any] | None = None
    idiomatic: str | None = None
    span_literal: str | None = None
    attempts: int = 1
    history: tuple[str, ...] = ()


# ---------------------------------------------------------------- inventory


def _opt(value: str) -> float | None:
    return float(value) if value.strip() else None


def _short_example(examples: Sequence[str]) -> str:
    """The shortest stored example, cut at a word boundary; long corpus sentences bloat the prompt."""
    if not examples:
        return ""
    best = min(examples, key=len)
    if len(best) <= _MAX_EXAMPLE_CHARS:
        return best
    return best[:_MAX_EXAMPLE_CHARS].rsplit(" ", 1)[0] + " ..."


def load_inventory(
    final_csv: Path,
    *,
    impli_csv: Path | None = None,
    domains_csv: Path | Sequence[Path] | None = None,
    extra_csvs: Sequence[Path] = (),
) -> tuple[list[Idiom], list[dict[str, str]]]:
    """Inventory idioms plus the raw rows (for the idiom matcher).

    `extra_csvs` adds further tiers with the same columns (the unnormed MAGPIE tier); missing domain
    files are skipped.
    """
    rows: list[dict[str, str]] = []
    for path in (final_csv, *extra_csvs):
        with path.open(encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    domains: dict[str, str] = {}
    paths = [domains_csv] if isinstance(domains_csv, Path) else list(domains_csv or ())
    for path in paths:
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                domains.update({r["idiom_id"]: r["domain"] for r in csv.DictReader(f)})
    impli: dict[str, str] = {}
    if impli_csv is not None and impli_csv.exists():
        with impli_csv.open(encoding="utf-8", newline="") as f:
            best: dict[str, int] = {}
            for r in csv.DictReader(f):
                n = int(r["n_pairs"])
                if n > best.get(r["idiom"], 0):
                    best[r["idiom"]] = n
                    impli[r["idiom"]] = r["impli_gloss"]
    inventory: list[Idiom] = []
    for r in rows:
        examples: list[str] = json.loads(r["example_sentences"])
        example = _short_example(examples)
        freq = _opt(r["min_word_count_enwiki"])
        ids = r["midas_ids"].split("|")
        inventory.append(
            Idiom(
                idiom_id=ids[0],
                midas_ids=r["midas_ids"],
                name=r["norm_idiom"],
                forms=tuple(json.loads(r["midas_forms"])),
                definitions=tuple(json.loads(r["definitions"])),
                example=example,
                opacity=_opt(r["opacity"]),
                familiarity=_opt(r["familiarity"]),
                log_freq=math.log10(freq) if freq else None,
                norm_source=r["norm_source"],
                domain=domains.get(ids[0]),
                impli_gloss=impli.get(r["norm_idiom"]),
            )
        )
    return inventory, rows


def opacity_cuts(inventory: Sequence[Idiom]) -> tuple[float, float]:
    vals = sorted(i.opacity for i in inventory if i.opacity is not None) or [0.0]
    return vals[len(vals) // _TERCILES], vals[2 * len(vals) // _TERCILES]


# ---------------------------------------------------------------- shortlist and prompt


def item_domain(item: Item) -> str | None:
    return AG_LABEL_DOMAIN.get(str(item.meta.get("label_name", "")))


def candidate_pool(
    inventory: Sequence[Idiom],
    k: int,
    usage: Counter[str],
    cap: int,
    exclude: frozenset[str],
    domain_condition: str | None,
    domain: str | None,
) -> tuple[list[Idiom], str]:
    """Idioms that may be shown for an item, and which domain condition actually applied."""
    pool = [i for i in inventory if usage[i.idiom_id] < cap and i.idiom_id not in exclude]
    used = "unconstrained"
    if domain_condition and domain:
        tagged = [i for i in pool if i.domain is not None]
        if domain_condition == "cross":
            sel = [i for i in tagged if i.domain not in (domain, NEUTRAL_DOMAIN)]
        else:
            sel = [i for i in tagged if i.domain in (domain, NEUTRAL_DOMAIN)]
        if len(sel) >= k:
            pool, used = sel, domain_condition
        else:
            used = f"{domain_condition}_fallback"
    return pool, used


def retrieval_shortlist(
    pool: Sequence[Idiom],
    proposals: Sequence[SpanProposal],
    retriever: IdiomRetriever,
    k: int,
    rng: random.Random,
) -> list[Idiom]:
    """Best-matching idioms per proposed span, taken round-robin until k are chosen (order shuffled)."""
    by_id = {i.idiom_id: i for i in pool}
    ranked = [[j for j in retriever.rank(p.meaning) if j in by_id] for p in proposals]
    chosen: list[str] = []
    for depth in range(len(by_id)):
        for ids in ranked:
            if depth < len(ids) and ids[depth] not in chosen:
                chosen.append(ids[depth])
        if len(chosen) >= k:
            break
    out = [by_id[j] for j in chosen[:k]]
    rng.shuffle(out)
    return out


def build_shortlist(
    inventory: Sequence[Idiom],
    k: int,
    rng: random.Random,
    usage: Counter[str],
    cap: int,
    cuts: tuple[float, float],
    *,
    domain_condition: str | None = None,
    domain: str | None = None,
    exclude: frozenset[str] = frozenset(),
) -> tuple[list[Idiom], str]:
    """~k idioms, balanced over opacity terciles, skipping idioms at the reuse cap; unnormed idioms
    (no opacity) only fill the remaining places.

    `domain_condition` "cross" keeps idioms whose source domain differs from the item's domain,
    "same_or_neutral" the others; if too few remain the shortlist falls back to unconstrained.
    """
    pool, used = candidate_pool(inventory, k, usage, cap, exclude, domain_condition, domain)
    groups: list[list[Idiom]] = [[], [], []]
    unnormed = [i for i in pool if i.opacity is None]
    for i in pool:
        if i.opacity is not None:
            groups[0 if i.opacity < cuts[0] else 1 if i.opacity < cuts[1] else 2].append(i)
    for g in groups:
        rng.shuffle(g)
    chosen: list[Idiom] = []
    per = k // _TERCILES
    for g in groups:
        chosen.extend(g[:per])
    rest = [i for g in groups for i in g[per:]] + unnormed
    rng.shuffle(rest)
    chosen.extend(rest[: k - len(chosen)])
    rng.shuffle(chosen)
    return chosen, used


def label_text(item: Item) -> str:
    if item.dataset == "sst2":
        return _SST2_LABELS.get(item.y, item.y)
    return str(item.meta.get("label_name", ""))


def prompt_hash(dataset: str, examples: bool = False, template: str = TEMPLATE_FILE) -> str:
    """Hash of everything that shapes the pick prompt (template, dataset wording, example flag)."""
    text = (
        load_prompt(template)
        + "\x1e"
        + json.dumps(DATASET_TEXT[dataset])
        + f"\x1eexamples={examples}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def span_prompt_hash(dataset: str, spans_template: str = SPANS_TEMPLATE_FILE) -> str:
    """Cache namespace of the span-proposal stage, independent of the pick prompt."""
    text = load_prompt(spans_template) + "\x1e" + DATASET_TEXT[dataset][0]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def render_span_prompt(item: Item, spans_template: str = SPANS_TEMPLATE_FILE) -> str:
    return load_prompt(spans_template).format(dataset_note=DATASET_TEXT[item.dataset][0], x=item.x)


class Suggester:
    """Turns idioms the model names in call 1 into shortlist entries, if MIDAS knows them (Kai 4a,
    "leave it up to its own decision"; 4c: the MIDAS definition and id are recorded).

    A suggestion matches a MIDAS idiom when its stemmed content key equals the key of one of the
    idiom's forms; keys shared by several different MIDAS idioms are ignored (duplicate entries of
    the same idiom resolve to the lowest id). Inventory idioms keep their
    inventory record (norms, domain); other MIDAS idioms enter as `norm_source = "midas_only"`.
    `presence_index` covers all of MIDAS, so the "idiom present" check recognises them.
    """

    def __init__(self, midas: Sequence[MidasIdiom], inventory: Sequence[Idiom]) -> None:
        by_id = {i: idiom for idiom in inventory for i in idiom.midas_ids.split("|")}
        keys: dict[frozenset[str], set[str]] = {}
        for m in midas:
            for f in m.forms:
                if key := content_key(f):
                    keys.setdefault(key, set()).add(m.midas_id)
        # MIDAS has some idioms twice (same form, separate ids): those count as one idiom.
        first_form = {m.midas_id: surface_form(m.forms[0]) for m in midas}
        self._by_key = {
            k: min(v, key=lambda i: (len(i), i))
            for k, v in keys.items()
            if len({first_form[i] for i in v}) == 1
        }
        self._idioms = {
            m.midas_id: by_id.get(m.midas_id)
            or Idiom(
                idiom_id=m.midas_id,
                midas_ids=m.midas_id,
                name=m.forms[0],
                forms=m.forms,
                definitions=m.meanings,
                example="",
                opacity=None,
                familiarity=None,
                log_freq=None,
                norm_source="midas_only",
                domain=None,
                impli_gloss=None,
            )
            for m in midas
        }
        self.presence_index = IdiomIndex(
            [
                {
                    "norm_idiom": m.forms[0],
                    "midas_ids": m.midas_id,
                    "definitions": json.dumps(list(m.meanings)),
                    "midas_forms": json.dumps(list(m.forms)),
                }
                for m in midas
            ],
            min_tokens=2,
            min_content=1,
        )

    def lookup(self, suggestion: str) -> Idiom | None:
        mid = self._by_key.get(content_key(suggestion)) if suggestion else None
        return self._idioms.get(mid) if mid else None


def render_prompt(
    item: Item,
    shortlist: Sequence[Idiom],
    examples: bool = False,
    template: str = TEMPLATE_FILE,
    proposals: Sequence[SpanProposal] = (),
) -> str:
    """Templates without a `{candidates}` slot ignore the proposed spans (str.format drops extras)."""
    note, rule = DATASET_TEXT[item.dataset]
    lines = []
    all_senses = template in ALL_SENSES_TEMPLATES
    for i in shortlist:
        senses = i.definitions if all_senses else i.definitions[:1]
        meaning = " / ".join(d.rstrip(".") for d in senses)
        ex = f"; e.g. {i.example}" if examples and i.example else ""
        lines.append(f"{i.idiom_id}: {i.forms[0]} - {meaning}{ex}")
    return load_prompt(template).format(
        dataset_note=note,
        label_rule=rule.format(label=label_text(item)),
        idiom_list="\n".join(lines),
        context=render_context(item.meta),
        x=item.x,
        candidates="\n".join(f'- "{p.span}": {p.meaning}' for p in proposals) or "(none proposed)",
    )


# ---------------------------------------------------------------- parsing and validation

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def extract_json(raw: str) -> dict[str, Any] | None:
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj: Any = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _repeats_context(x: str, span: str, *replacements: str) -> bool:
    """True if a replacement re-copies the words just before or after the span."""
    start = x.index(span)
    left = x[:start].split()[-_CONTEXT_WORDS:]
    right = x[start + len(span) :].split()[:_CONTEXT_WORDS]
    return any(
        len(words) == _CONTEXT_WORDS and " ".join(words) in text
        for words in (left, right)
        for text in replacements
    )


def validate(
    raw: str,
    x: str,
    shortlist_ids: set[str],
    index: IdiomIndex,
    presence_index: IdiomIndex | None = None,
) -> Parsed:
    """Parse the model's JSON and check every constraint that code can check; first failure wins.

    `presence_index` (all of MIDAS, with model-suggested idioms) is used only to confirm that the
    chosen idiom is in the idiomatic text; the other idiom checks keep the inventory `index`.
    """
    present = presence_index or index
    obj = extract_json(raw)
    if obj is None:
        return Parsed(False, "bad_json")
    choice = str(obj.get("choice", "")).strip()
    # Window mode: the model rewrites a short window around the idiom; the literal text is built
    # here by swapping only the idiom phrase, so idiomatic and literal differ only in the idiom.
    window = "window" in obj
    idiom_phrase = literal_phrase = ""
    if window:
        span, idiom_text, idiom_phrase, literal_phrase = (
            str(obj.get(k, "") or "")
            for k in ("window", "idiomatic_window", "idiom_phrase", "literal_phrase")
        )
        found = bool(idiom_phrase) and idiom_text.count(idiom_phrase) == 1
        literal = idiom_text.replace(idiom_phrase, literal_phrase, 1) if found else ""
    else:
        span, idiom_text, literal = (
            str(obj.get(k, "")) for k in ("span", "idiom_text", "literal_text")
        )
    max_span = _MAX_WINDOW_WORDS if window else _MAX_SPAN_WORDS
    max_replacement = _MAX_WINDOW_REPLACEMENT_WORDS if window else _MAX_REPLACEMENT_WORDS
    verdict = str(obj.get("verdict", "KEEP")).strip().upper()
    checks: list[tuple[str, Callable[[], bool]]] = [
        ("none", lambda: choice.upper() == "NONE"),
        ("self_rejected", lambda: verdict != "KEEP"),
        ("unknown_choice", lambda: choice not in shortlist_ids),
        (
            "idiom_phrase_not_found",
            lambda: (
                window
                and not (
                    idiom_phrase
                    and literal_phrase.strip()
                    and idiom_text.count(idiom_phrase) == 1
                    and idiom_phrase != literal_phrase
                )
            ),
        ),
        ("empty_field", lambda: not (span.strip() and idiom_text.strip() and literal.strip())),
        ("span_not_unique", lambda: x.count(span) != 1),
        (
            "replacement_too_long",
            lambda: max(len(idiom_text.split()), len(literal.split())) > max_replacement,
        ),
        ("context_repeated", lambda: _repeats_context(x, span, idiom_text, literal)),
        ("article_mismatch", lambda: _article_mismatch(x, span, idiom_text, literal)),
        ("span_too_long", lambda: len(span.split()) > max_span),
        (
            "window_rewritten_too_much",
            lambda: window and _rewritten_too_much(span, idiom_text, idiom_phrase),
        ),
        # The literal must reword the span too, so idiomatic vs span-literal isolates the idiom from
        # the cost of rewording (a literal equal to the span would just be the original text).
        # Also catches an "idiom" that is the original phrase again up to case or inflection
        # ("young turks" -> "young Turks").
        (
            "no_change",
            lambda: (
                len({span, idiom_text, literal}) < _N_TEXTS
                or content_key(idiom_text) == content_key(span)
            ),
        ),
        ("span_has_idiom", lambda: bool(index.find(span))),
        (
            "idiom_not_present",
            lambda: (
                not any(
                    choice in h.ref.midas_ids.split("|")
                    for h in present.find(idiom_phrase if window else idiom_text)
                )
            ),
        ),
        ("idiom_in_literal", lambda: bool(index.find(literal))),
    ]
    for reason, failed in checks:
        if failed():
            return Parsed(False, reason, choice=choice, span=span)
    return Parsed(True, "ok", choice, span, idiom_text, literal, idiom_phrase, literal_phrase)


def outside_span_identical(
    original: str, variant: str, start: int, old_len: int, new_len: int
) -> bool:
    """True if `variant` equals `original` everywhere except the span replaced at `start`."""
    return (
        variant[:start] == original[:start]
        and variant[start + new_len :] == original[start + old_len :]
    )


def build_variants(x: str, p: Parsed) -> tuple[str, str, int]:
    start = x.index(p.span)
    end = start + len(p.span)
    return x[:start] + p.idiom_text + x[end:], x[:start] + p.literal_text + x[end:], start


_SPAN_INDEX: list[IdiomIndex] = []
# Light verbs and particles say little about whether a phrase is the idiom (stemmed like content_key).
_WEAK_WORDS = content_key(
    "do have get go take make put give come keep up out off in on down over away back about around"
)


def _midas_span_index() -> IdiomIndex:
    """All MIDAS idioms of at least 3 words with 2 content words (built once, lazily)."""
    if not _SPAN_INDEX:
        from data.norms.inventory import load_midas  # noqa: PLC0415

        _SPAN_INDEX.append(
            IdiomIndex(
                [
                    {
                        "norm_idiom": m.forms[0],
                        "midas_ids": m.midas_id,
                        "definitions": json.dumps(list(m.meanings)),
                        "midas_forms": json.dumps(list(m.forms)),
                    }
                    for m in load_midas()
                ]
            )
        )
    return _SPAN_INDEX[0]


def span_already_idiomatic(span: str, idiom_forms: Sequence[str]) -> bool:
    """Flag (not a rejection): the replaced phrase was itself an idiom, e.g. "hits a nerve" ->
    "touches a nerve". The idiomatic/literal pair stays valid; the original is not idiom-free.

    True if the span contains at least half of the chosen idiom's own content words, or any MIDAS
    idiom of at least 3 words with 2 content words.
    """
    words = content_key(span) - _WEAK_WORDS
    shares = any(
        (key := content_key(f) - _WEAK_WORDS) and len(key & words) * 2 >= len(key)
        for f in idiom_forms
    )
    return shares or bool(_midas_span_index().find(span))


def make_record(
    item: Item,
    idiom: Idiom,
    p: Parsed,
    start: int,
    *,
    condition: str,
    model: str,
    p_hash: str,
    shortlist_size: int,
    attempts: int = 1,
    mode: str = "random",
    suggested: bool = False,
) -> dict[str, Any]:
    dom = item_domain(item)
    source = {"magpie_bnc": "midas_x_magpie", "midas_only": "midas_only"}.get(
        idiom.norm_source, "midas_x_norms"
    )
    return {
        "idiom": idiom.name,
        "midas_ids": idiom.midas_ids,
        "span": p.span,
        "idiom_text": p.idiom_text,
        "literal_text": p.literal_text,
        "insertion": "window" if p.idiom_phrase else "span",
        "span_already_idiomatic": span_already_idiomatic(p.span, idiom.forms),
        "idiom_phrase": p.idiom_phrase or p.idiom_text,
        "literal_phrase": p.literal_phrase or p.literal_text,
        "definition": " / ".join(idiom.definitions),
        "definition_source": "midas+impli" if idiom.impli_gloss else "midas",
        "impli_gloss": idiom.impli_gloss,
        "inventory_source": source,
        "suggested_by_model": suggested,
        "norm_source": idiom.norm_source,
        "opacity": idiom.opacity,
        "familiarity": idiom.familiarity,
        "log_freq": idiom.log_freq,
        "idiom_domain": idiom.domain,
        "item_domain": dom,
        "idiom_domain_mismatch": (
            None
            if dom is None or idiom.domain is None
            else idiom.domain not in (dom, NEUTRAL_DOMAIN)
        ),
        "domain_condition": condition,
        "span_start": start,
        "idiomatic_span": [start, start + len(p.idiom_text)],
        "literal_span": [start, start + len(p.literal_text)],
        "generator_model": model,
        "prompt_hash": p_hash,
        "shortlist_size": shortlist_size,
        "shortlist_mode": mode,
        "attempts": attempts,
    }


# ---------------------------------------------------------------- orchestration

CompleteFn = Callable[[Sequence[str]], list[str | None]]


def prepare(
    items: Sequence[Item],
    inventory: Sequence[Idiom],
    *,
    k: int,
    cap: int,
    seed: int,
    usage: Counter[str],
    cross_fraction: float = 0.5,
    attempt: int = 0,
    exclude: Mapping[str, frozenset[str]] | None = None,
    proposals: Mapping[str, Sequence[SpanProposal]] | None = None,
    retriever: IdiomRetriever | None = None,
    examples: bool = False,
    template: str = TEMPLATE_FILE,
    suggester: Suggester | None = None,
) -> list[Prepared]:
    """Retries (attempt > 0) keep the item's domain condition but draw a fresh shortlist without
    any idiom already shown for that item. With a retriever and proposed spans the shortlist is the
    idioms whose definitions best match the proposed meanings; otherwise it is a random draw. With a
    `suggester`, idioms the model named for its spans (and MIDAS knows) are added to the shortlist."""
    cuts = opacity_cuts(inventory)
    out: list[Prepared] = []
    for item in items:
        rng = random.Random(f"{seed}:{item.id}")
        dom = item_domain(item)
        cond = ("cross" if rng.random() < cross_fraction else "same_or_neutral") if dom else None
        if attempt:
            rng = random.Random(f"{seed}:{item.id}:{attempt}")
        skip = (exclude or {}).get(item.id, frozenset())
        short: list[Idiom] = []
        mode = "random"
        used = "unconstrained"
        if retriever is not None and (props := (proposals or {}).get(item.id)):
            pool, used = candidate_pool(inventory, k, usage, cap, skip, cond, dom)
            short = retrieval_shortlist(pool, props, retriever, k, rng)
            mode = "retrieval"
        suggested: set[str] = set()
        if suggester is not None:
            for p in (proposals or {}).get(item.id, ()):
                idiom = suggester.lookup(p.idiom)
                if idiom is None or idiom.idiom_id in skip or usage[idiom.idiom_id] >= cap:
                    continue
                suggested.add(idiom.idiom_id)
                if all(i.idiom_id != idiom.idiom_id for i in short):
                    short.append(idiom)
        if not short:
            mode = "random"
            short, used = build_shortlist(
                inventory,
                k,
                rng,
                usage,
                cap,
                cuts,
                domain_condition=cond,
                domain=dom,
                exclude=skip,
            )
        out.append(
            Prepared(
                item,
                tuple(short),
                used,
                render_prompt(item, short, examples, template, (proposals or {}).get(item.id, ())),
                attempt,
                mode,
                frozenset(suggested),
            )
        )
    return out


def finish(
    prepared: Sequence[Prepared],
    raws: Sequence[str | None],
    index: IdiomIndex,
    *,
    model: str,
    p_hash: str,
    presence_index: IdiomIndex | None = None,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for prep, raw in zip(prepared, raws, strict=True):
        item = prep.item
        if raw is None:
            outcomes.append(Outcome(item, False, "empty_response", "", attempts=prep.attempt + 1))
            continue
        ids = {i.idiom_id for i in prep.shortlist}
        parsed = validate(raw, item.x, ids, index, presence_index)
        if not parsed.ok:
            outcomes.append(Outcome(item, False, parsed.reason, raw, attempts=prep.attempt + 1))
            continue
        idiom = next(i for i in prep.shortlist if i.idiom_id == parsed.choice)
        idiomatic, literal, start = build_variants(item.x, parsed)
        assert outside_span_identical(
            item.x, idiomatic, start, len(parsed.span), len(parsed.idiom_text)
        )
        assert outside_span_identical(
            item.x, literal, start, len(parsed.span), len(parsed.literal_text)
        )
        record = make_record(
            item,
            idiom,
            parsed,
            start,
            condition=prep.condition,
            model=model,
            p_hash=p_hash,
            shortlist_size=len(prep.shortlist),
            attempts=prep.attempt + 1,
            mode=prep.mode,
            suggested=idiom.idiom_id in prep.suggested,
        )
        outcomes.append(
            Outcome(item, True, "ok", raw, record, idiomatic, literal, attempts=prep.attempt + 1)
        )
    return outcomes


def request_key(prompt: str, temperature: float, max_output_tokens: int) -> str:
    """Content address of one request: resuming reuses a reply only for an identical request."""
    raw = f"{temperature}\x1e{max_output_tokens}\x1e{prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def sync_completer(
    client: LLMClient,
    cache: ResponseCache | None,
    p_hash: str,
    *,
    temperature: float,
    max_output_tokens: int,
    workers: int,
) -> CompleteFn:
    """Thread-pool completer with a response cache and retries on transient errors.

    Every reply is cached as soon as it arrives, so a crashed or interrupted run resumes by
    rerunning the same command: finished requests are served from disk.
    """

    def one(prompt: str) -> str | None:
        key = request_key(prompt, temperature, max_output_tokens)
        model = f"{client.provider}/{client.model}"
        if cache is not None and (hit := cache.get(p_hash, model, key)) is not None:
            return str(hit["raw"])
        for attempt in range(_RETRIES):
            try:
                raw = client.complete(
                    system="",
                    user=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except EmptyResponseError:
                return None
            except LLMError:
                if attempt == _RETRIES - 1:
                    raise
                time.sleep(_BACKOFF_S * 2**attempt)
                continue
            if cache is not None:
                cache.put(p_hash, model, key, {"raw": raw})
            return raw
        return None

    def complete(prompts: Sequence[str]) -> list[str | None]:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    return complete


def cached_completer(
    run: CompleteFn,
    cache: ResponseCache | None,
    p_hash: str,
    model: str,
    *,
    temperature: float,
    max_output_tokens: int,
) -> CompleteFn:
    """Serve cached replies and send only the misses to `run` (used for batch jobs)."""

    def complete(prompts: Sequence[str]) -> list[str | None]:
        keys = [request_key(p, temperature, max_output_tokens) for p in prompts]
        out: list[str | None] = [None] * len(prompts)
        missing: list[int] = []
        for n, key in enumerate(keys):
            hit = cache.get(p_hash, model, key) if cache is not None else None
            if hit is not None:
                out[n] = str(hit["raw"])
            else:
                missing.append(n)
        if missing:
            for n, raw in zip(missing, run([prompts[n] for n in missing]), strict=True):
                out[n] = raw
                if raw is not None and cache is not None:
                    cache.put(p_hash, model, keys[n], {"raw": raw})
        return out

    return complete


def generate(
    items: Sequence[Item],
    inventory: Sequence[Idiom],
    index: IdiomIndex,
    complete: CompleteFn,
    *,
    model: str,
    p_hash: str,
    k: int = 12,
    cap: int = 15,
    seed: int = 0,
    wave: int = 100,
    cross_fraction: float = 0.5,
    max_attempts: int = 1,
    retriever: IdiomRetriever | None = None,
    target_kept: int = 0,
    complete_spans: CompleteFn | None = None,
    examples: bool = False,
    template: str = TEMPLATE_FILE,
    suggester: Suggester | None = None,
    spans_template: str = SPANS_TEMPLATE_FILE,
    skip_empty: bool = True,
) -> list[Outcome]:
    """Waves keep the per-idiom reuse cap honest: usage from finished waves shapes later shortlists,
    and within a wave an idiom is never kept more often than the cap allows (`idiom_cap`).

    A failed item (NONE or a failed check) is retried up to `max_attempts` in total, each time with a
    fresh shortlist that excludes every idiom already shown for it. With a `retriever`, each wave
    first asks the model for replaceable spans and shortlists idioms by meaning (see
    `idiom_retrieval`); items without usable proposals fall back to random shortlists. With
    `target_kept` the loop ends after the wave in which that many items passed (items are in the
    pool's shuffled order, so the stop does not bias the sample). Items whose original text already
    contains an inventory idiom are skipped without any API call (`original_has_idiom`). With a
    `suggester` (and a spans template that asks for idioms) the model's own idiom suggestions that
    MIDAS knows are added to each shortlist. With `skip_empty` (default) an item for which call 1
    proposed no usable phrase gets no call 2 (`no_candidate`) instead of a random shortlist, which
    almost never fits.
    """
    usage: Counter[str] = Counter()
    presence = suggester.presence_index if suggester is not None else None
    outcomes: list[Outcome] = []
    for start in range(0, len(items), wave):
        if target_kept and sum(o.ok for o in outcomes) >= target_kept:
            break
        window = items[start : start + wave]
        skipped: dict[str, Outcome] = {}
        blocked = {i.id for i in window if index.find(i.x)}
        batch = [i for i in window if i.id not in blocked]
        proposals: dict[str, list[SpanProposal]] = {}
        if retriever is not None:
            replies = (complete_spans or complete)(
                [render_span_prompt(i, spans_template) for i in batch]
            )
            proposals = {i.id: parse_proposals(r, i.x) for i, r in zip(batch, replies, strict=True)}
            if skip_empty:
                skipped = {
                    i.id: Outcome(i, False, "no_candidate", "")
                    for i in batch
                    if not proposals.get(i.id)
                }
                batch = [i for i in batch if i.id not in skipped]

        def draw(
            subset: Sequence[Item],
            props: Mapping[str, Sequence[SpanProposal]],
            attempt: int = 0,
            exclude: Mapping[str, frozenset[str]] | None = None,
        ) -> list[Prepared]:
            return prepare(
                subset,
                inventory,
                k=k,
                cap=cap,
                seed=seed,
                usage=usage,
                cross_fraction=cross_fraction,
                attempt=attempt,
                exclude=exclude,
                proposals=props,
                retriever=retriever,
                examples=examples,
                template=template,
                suggester=suggester,
            )

        prepared = draw(batch, proposals)
        done = finish(
            prepared,
            complete([p.prompt for p in prepared]),
            index,
            model=model,
            p_hash=p_hash,
            presence_index=presence,
        )
        shown = {p.item.id: frozenset(i.idiom_id for i in p.shortlist) for p in prepared}
        for attempt in range(1, max_attempts):
            redo = [n for n, o in enumerate(done) if not o.ok]
            if not redo:
                break
            again = draw([done[n].item for n in redo], proposals, attempt, shown)
            fresh = finish(
                again,
                complete([p.prompt for p in again]),
                index,
                model=model,
                p_hash=p_hash,
                presence_index=presence,
            )
            for n, p, o in zip(redo, again, fresh, strict=True):
                shown[p.item.id] = shown[p.item.id] | {i.idiom_id for i in p.shortlist}
                done[n] = replace(o, history=(*done[n].history, done[n].reason))
        for n, o in enumerate(done):
            if o.ok and o.record:
                key = str(o.record["midas_ids"]).split("|")[0]
                if usage[key] >= cap:
                    done[n] = replace(
                        o,
                        ok=False,
                        reason="idiom_cap",
                        record=None,
                        idiomatic=None,
                        span_literal=None,
                    )
                else:
                    usage[key] += 1
        by_id = {o.item.id: o for o in done} | skipped
        outcomes.extend(
            by_id.get(i.id) or Outcome(i, False, "original_has_idiom", "") for i in window
        )
    return outcomes


# ---------------------------------------------------------------- item sampling and output

_COLUMNS = ["id", "variant", "x", "y", "meta", "augmenter_model", "prompt_hash", "validators"]


def load_pool(pool_csv: Path, dataset: str) -> tuple[list[Item], dict[str, dict[str, str]]]:
    with pool_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    items = [Item(dataset, r["id"], r["x"], r["y"], json.loads(r["meta"] or "{}")) for r in rows]
    return items, {r["id"]: r for r in rows}


def filter_items(items: Sequence[Item], where: Sequence[str]) -> list[Item]:
    """Items whose meta fields match every `field=value[,value]` condition (order kept)."""
    out = list(items)
    for cond in where:
        field, _, values = cond.partition("=")
        if not values:
            raise ValueError(f"condition needs field=value[,value]: {cond!r}")
        wanted = {v.strip() for v in values.split(",")}
        out = [i for i in out if str(i.meta.get(field.strip(), "")) in wanted]
    return out


def stratified_sample(items: Sequence[Item], n: int, seed: int) -> list[Item]:
    """n items spread proportionally over label x length tercile, deterministically."""
    if n >= len(items):
        return list(items)
    lengths = sorted(len(i.x.split()) for i in items)
    cuts = (lengths[len(lengths) // _TERCILES], lengths[2 * len(lengths) // _TERCILES])
    strata: dict[tuple[str, int], list[Item]] = {}
    for i in items:
        w = len(i.x.split())
        strata.setdefault((i.y, 0 if w < cuts[0] else 1 if w < cuts[1] else 2), []).append(i)
    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)
    out: list[Item] = []
    total = len(items)
    for group in strata.values():
        out.extend(group[: max(1, round(n * len(group) / total))])
    rng.shuffle(out)
    return out[:n]


def write_outputs(
    out_dir: Path,
    outcomes: Sequence[Outcome],
    pool_rows: dict[str, dict[str, str]],
    *,
    model: str,
    p_hash: str,
    meta: dict[str, Any],
) -> dict[str, int]:
    """original / idiomatic / span_literal CSVs for the successful items, plus a log and a sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [o for o in outcomes if o.ok and o.record and o.idiomatic and o.span_literal]
    writers = {v: [] for v in ("original", "idiomatic", "span_literal")}
    for o in ok:
        base = pool_rows[o.item.id]
        writers["original"].append(dict(base))
        for variant, text in (("idiomatic", o.idiomatic), ("span_literal", o.span_literal)):
            writers[variant].append(
                {
                    "id": o.item.id,
                    "variant": variant,
                    "x": text,
                    "y": base["y"],
                    "meta": json.dumps(
                        {**o.item.meta, "idiom": o.record}, sort_keys=True, ensure_ascii=False
                    ),
                    "augmenter_model": model,
                    "prompt_hash": p_hash,
                    "validators": "[]",
                }
            )
    for variant, rows in writers.items():
        with (out_dir / f"{variant}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    with (out_dir / "generation_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ok", "reason", "idiom", "attempts", "earlier_failures"])
        for o in outcomes:
            w.writerow(
                [
                    o.item.id,
                    o.ok,
                    o.reason,
                    (o.record or {}).get("idiom", ""),
                    o.attempts,
                    "|".join(o.history),
                ]
            )
    counts = dict(Counter(o.reason for o in outcomes))
    first_attempt = dict(Counter(o.history[0] if o.history else o.reason for o in outcomes))
    (out_dir / "generation.meta.json").write_text(
        json.dumps(
            {
                **meta,
                "model": model,
                "prompt_hash": p_hash,
                "counts": counts,
                "first_attempt_counts": first_attempt,
                "kept": len(ok),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return counts
