"""v3 generation: insert one bank idiom, then write the same text with only the idiom in plain words.

THESIS_PLAN_v3 §2. One call per item returns the idiomatic text, the idiom's words as they appear, their
plain meaning, and the full literal text; code checks that the literal text is exactly the idiomatic
text with that one span replaced, so the two versions the models are compared on differ only in the
idiom. The original stays the anchor (source text and label). The shortlist mixes idioms that fit the
sentence (local embedding retrieval) with random ones; a failed item gets one more try with a fresh
shortlist.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from augmentation.idiom_generation import (
    Idiom,
    Item,
    Outcome,
    article_for,
    article_mismatch,
    extract_json,
    label_text,
)
from augmentation.idiom_retrieval import IdiomRetriever
from augmentation.prompts.loader import load_prompt
from data.norms.idiom_text import IdiomIndex, tokenize

# FROZEN 2026-10-30 (author, after E2: 77/80 kept, ≈ 87% acceptable): idiom_insert_v2.txt.
# v2 (2026-10-30, after E1): small added details allowed (label must hold), the idiom may sit on words the
# label depends on, explicit naturalness / literal-clash / grammar rules; v1 is kept to reproduce E1.
TEMPLATE_FILE = "idiom_insert_v2.txt"
N_FIT = 20
N_RANDOM = 10
ATTEMPTS = 2
_LETTERS = "ABCDEFGH"

CompleteFn = Callable[[Sequence[str]], list[str | None]]


@dataclass(frozen=True, slots=True)
class DatasetText:
    note: str
    rule: str
    question: str
    example: str
    example_id: str  # pool id of the example sentence; never sampled for generation


DATASET_TEXT: dict[str, DatasetText] = {
    "sst2": DatasetText(
        note="The text is a sentence from a movie review.",
        rule="The sentiment must stay {label}; it may sound a little more vivid, but must not become "
        "clearly stronger or weaker.",
        question="Is the sentiment still {label}?",
        example=(
            'Text (label: negative): "is a tired one , with few moments of joy rising above the stale '
            'material ."\n'
            '{"choice": "<id>", "idiomatic": "is a tired one that goes through the motions , with few '
            'moments of joy rising above the stale material .", "idiom_text": "goes through the '
            'motions", "literal_text": "does everything mechanically and without enthusiasm", '
            '"literal": "is a tired one that does everything mechanically and without enthusiasm , with '
            'few moments of joy rising above the stale material .", "check": "still negative; same '
            'meaning; natural", "verdict": "KEEP"}'
        ),
        example_id="b8e87dad40ca8984",
    ),
    "mnli": DatasetText(
        note="The text is the premise of an inference pair; the hypothesis below is never changed.",
        rule="The relationship to the hypothesis must stay {label}: add nothing that makes the "
        "hypothesis more or less supported.",
        question="Is the relationship to the hypothesis still {label}?",
        example=(
            "Hypothesis (fixed, never changed): \"Kirby protected him and made sure that he didn't get "
            'in trouble."\n'
            'Text (label: neutral): "Kirby stopped you, but next time that could lead to real trouble."\n'
            '{"choice": "<id>", "idiomatic": "Kirby stopped you, but next time that could open a can of '
            'worms.", "idiom_text": "open a can of worms", "literal_text": "cause a lot of complicated '
            'trouble", "literal": "Kirby stopped you, but next time that could cause a lot of '
            'complicated trouble.", "check": "still neutral; same meaning; natural", "verdict": "KEEP"}'
        ),
        example_id="45008n",
    ),
    "mmlu": DatasetText(
        note="The text is the question of a multiple-choice item; the answer choices below are never "
        "changed.",
        rule="The correct answer must stay the same and must not be hinted at; do not touch technical "
        "terms, numbers or names.",
        question="Is the correct answer still {label} and not hinted at?",
        example=(
            "Answer choices (fixed, never changed): A) 23% B) 43% C) 63% D) 83%\n"
            'Text (label: C) 63%): "As of 2019, about what share of electricity came from fossil fuels '
            'in the United States?"\n'
            '{"choice": "<id>", "idiomatic": "As of 2019, what is a ballpark figure for the share of '
            'electricity that came from fossil fuels in the United States?", "idiom_text": "a ballpark '
            'figure", "literal_text": "a rough estimate", "literal": "As of 2019, what is a rough '
            'estimate for the share of electricity that came from fossil fuels in the United States?", '
            '"check": "answer still C, not hinted; same meaning; natural", "verdict": "KEEP"}'
        ),
        example_id="test/global_facts/2596",
    ),
    "agnews": DatasetText(
        note="The text is a news snippet (title, then description).",
        rule="The topic must stay {label} for a human reader.",
        question="Is the topic still {label}?",
        example=(
            'Text (label: World): "Truckers defy Kathmandu blockade More lorry drivers are bringing '
            "supplies to Nepal's capital in defiance of an indefinite blockade by Maoist rebels.\"\n"
            '{"choice": "<id>", "idiomatic": "Truckers defy Kathmandu blockade More lorry drivers are '
            "bringing supplies to Nepal's capital, flying in the face of an indefinite blockade by "
            'Maoist rebels.", "idiom_text": "flying in the face of", "literal_text": "openly defying", '
            '"literal": "Truckers defy Kathmandu blockade More lorry drivers are bringing supplies to '
            'Nepal\'s capital, openly defying an indefinite blockade by Maoist rebels.", "check": '
            '"still World; same meaning; natural", "verdict": "KEEP"}'
        ),
        example_id="80b240666d04a5b5",
    ),
}
# v2 rules differ from v1 only for SST-2 (no strength limit: both versions carry the same strength) and
# MMLU (the idiom may replace words the question turns on; numbers, names, technical terms stay).
DATASET_TEXT_V2: dict[str, DatasetText] = {
    **DATASET_TEXT,
    "sst2": replace(DATASET_TEXT["sst2"], rule="The sentiment must stay {label}."),
    "mmlu": replace(
        DATASET_TEXT["mmlu"],
        rule="The correct answer must stay {label} and must not be hinted at; numbers, names and "
        "technical terms stay exactly as they are. The idiom may replace other words, including the "
        "ones the question turns on, as long as the correct answer stays the same.",
    ),
}
TEXTS_BY_TEMPLATE: dict[str, dict[str, DatasetText]] = {
    "idiom_insert_v1.txt": DATASET_TEXT,
    "idiom_insert_v2.txt": DATASET_TEXT_V2,
}
EXAMPLE_IDS = frozenset(t.example_id for t in DATASET_TEXT.values())


def insert_label(item: Item) -> str:
    """The label as the prompt states it; MMLU shows the correct letter with its choice text."""
    if item.dataset == "mmlu":
        choices = list(item.meta.get("choices", []))
        idx = int(item.meta.get("answer_index", item.y))
        return f"{_LETTERS[idx]}) {choices[idx]}" if idx < len(choices) else item.y
    return label_text(item)


def insert_context(item: Item) -> str:
    """Fixed parts the model must see but never change; empty for SST-2 and AG News."""
    if item.dataset == "mnli" and item.meta.get("hypothesis"):
        return f'Hypothesis (fixed, never changed): "{item.meta["hypothesis"]}"\n\n'
    if item.dataset == "mmlu" and item.meta.get("choices"):
        opts = " ".join(f"{_LETTERS[n]}) {c}" for n, c in enumerate(item.meta["choices"]))
        return f"Answer choices (fixed, never changed): {opts}\n\n"
    return ""


def insert_prompt_hash(dataset: str, template: str = TEMPLATE_FILE) -> str:
    """Cache namespace: template plus this dataset's wording and example."""
    t = TEXTS_BY_TEMPLATE[template][dataset]
    text = load_prompt(template) + "\x1e" + json.dumps([t.note, t.rule, t.question, t.example])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def render_insert_prompt(
    item: Item, shortlist: Sequence[Idiom], template: str = TEMPLATE_FILE
) -> str:
    t = TEXTS_BY_TEMPLATE[template][item.dataset]
    label = insert_label(item)
    lines = [f"{i.idiom_id}: {i.forms[0]} - {i.definitions[0].rstrip('.')}" for i in shortlist]
    return load_prompt(template).format(
        dataset_note=t.note,
        label_rule=t.rule.format(label=label),
        label_question=t.question.format(label=label),
        example=t.example,
        idiom_list="\n".join(lines),
        context=insert_context(item),
        label=label,
        x=item.x,
    )


def mixed_shortlist(
    item: Item,
    pool: Sequence[Idiom],
    retriever: IdiomRetriever | None,
    rng: random.Random,
    *,
    n_fit: int = N_FIT,
    n_random: int = N_RANDOM,
) -> list[Idiom]:
    """≈ n_fit idioms closest in meaning to the sentence + n_random others, in shuffled order."""
    by_id = {i.idiom_id: i for i in pool}
    ranked = [j for j in retriever.rank(item.x) if j in by_id] if retriever else []
    fit = ranked[:n_fit]
    rest = [i for i in pool if i.idiom_id not in set(fit)]
    extra = rng.sample(rest, min(len(rest), n_random + (n_fit - len(fit))))
    out = [by_id[j] for j in fit] + extra
    rng.shuffle(out)
    return out


@dataclass(frozen=True, slots=True)
class Inserted:
    ok: bool
    reason: str
    choice: str = ""
    idiomatic: str = ""
    idiom_text: str = ""
    literal_text: str = ""
    literal: str = ""
    check: str = ""


def words_kept(original: str, idiomatic: str) -> float:
    """Share of the original's tokens that survive, in order, in the idiomatic text."""
    a, b = tokenize(original), tokenize(idiomatic)
    if not a:
        return 1.0
    kept = sum(
        m.size for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks()
    )
    return kept / len(a)


_ARTICLE_PAIR = re.compile(r"\b(a|an)\s+([A-Za-z][\w'-]*)", re.IGNORECASE)


def article_errors(text: str) -> int:
    """How many "a"/"an" disagree with the next word by the spelling heuristic ("a entertaining")."""
    return sum(article_for(w) != a.lower() for a, w in _ARTICLE_PAIR.findall(text))


def new_article_error(original: str, *rewrites: str) -> bool:
    """True if a rewrite has more a/an errors than the original (the original is not judged)."""
    base = article_errors(original)
    return any(article_errors(r) > base for r in rewrites)


def validate_insert(raw: str, x: str, shortlist_ids: set[str], index: IdiomIndex) -> Inserted:
    """Parse the reply and run every check code can do; the first failure wins (plan v3 §2.3)."""
    obj = extract_json(raw)
    if obj is None:
        return Inserted(False, "bad_json")

    def get(k: str) -> str:
        return str(obj.get(k, "") or "").strip()

    choice, idiomatic, idiom_text = get("choice"), get("idiomatic"), get("idiom_text")
    literal_text, literal, verdict = get("literal_text"), get("literal"), get("verdict").upper()
    checks: list[tuple[str, Callable[[], bool]]] = [
        ("none", lambda: choice.upper() == "NONE"),
        ("self_rejected", lambda: verdict != "KEEP"),
        ("unknown_choice", lambda: choice not in shortlist_ids),
        ("empty_field", lambda: not (idiomatic and idiom_text and literal_text and literal)),
        ("no_change", lambda: idiomatic == x.strip()),
        ("idiom_text_not_unique", lambda: idiomatic.count(idiom_text) != 1),
        ("literal_same_as_idiom", lambda: literal_text == idiom_text),
        ("literal_mismatch", lambda: literal != idiomatic.replace(idiom_text, literal_text, 1)),
        (
            "article_mismatch",
            lambda: (
                article_mismatch(idiomatic, idiom_text, literal_text)
                or new_article_error(x, idiomatic, literal)
            ),
        ),
        (
            "idiom_not_present",
            lambda: not any(choice in h.ref.midas_ids.split("|") for h in index.find(idiom_text)),
        ),
        ("idiom_in_literal", lambda: bool(index.find(literal_text))),
    ]
    for reason, failed in checks:
        if failed():
            return Inserted(False, reason, choice=choice)
    return Inserted(True, "ok", choice, idiomatic, idiom_text, literal_text, literal, get("check"))


@dataclass(slots=True)
class InsertState:
    """What carries over between attempts: idioms already shown per item, uses per idiom."""

    shown: dict[str, set[str]] = field(default_factory=dict)
    usage: Counter[str] = field(default_factory=Counter)


def insert_record(
    item: Item,
    idiom: Idiom,
    p: Inserted,
    *,
    attempt: int,
    model: str,
    p_hash: str,
    shortlist_size: int,
) -> dict[str, Any]:
    start = p.idiomatic.index(p.idiom_text)
    return {
        "idiom": idiom.name,
        "midas_ids": idiom.midas_ids,
        "definition": " / ".join(idiom.definitions),
        "definition_source": "midas",
        "inventory_source": {"magpie_bnc": "midas_x_magpie"}.get(
            idiom.norm_source, "midas_x_norms"
        ),
        "norm_source": idiom.norm_source,
        "opacity": idiom.opacity,
        "familiarity": idiom.familiarity,
        "log_freq": idiom.log_freq,
        "idiom_domain": idiom.domain,
        "idiom_text": p.idiom_text,
        "literal_text": p.literal_text,
        "idiomatic_span": [start, start + len(p.idiom_text)],
        "literal_span": [start, start + len(p.literal_text)],
        "original_words_kept": round(words_kept(item.x, p.idiomatic), 3),
        "self_check": p.check,
        "insertion": "v3",
        "attempts": attempt,
        "shortlist_size": shortlist_size,
        "shortlist_mode": f"mixed_{N_FIT}fit_{N_RANDOM}random",
        "generator_model": model,
        "prompt_hash": p_hash,
    }


def item_rng(seed: int, item_id: str, attempt: int) -> random.Random:
    """Shortlist randomness that depends only on the item and attempt, so the same item always gets the
    same prompt (cache reuse across samples and runs); E1/E2 used one run-wide sequence instead."""
    digest = hashlib.sha256(f"{seed}\x1e{item_id}\x1e{attempt}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def generate_insert(
    items: Sequence[Item],
    inventory: Sequence[Idiom],
    index: IdiomIndex,
    complete: CompleteFn,
    retriever: IdiomRetriever | None,
    *,
    model: str,
    p_hash: str,
    cap: int,
    seed: int,
    attempts: int = ATTEMPTS,
    template: str = TEMPLATE_FILE,
    per_item_seed: bool = True,
) -> list[Outcome]:
    """Items whose original already contains a bank idiom are skipped without an AI call."""
    rng = random.Random(seed)
    by_id = {i.idiom_id: i for i in inventory}
    state = InsertState()
    todo = [i for i in items if not index.find(i.x)]
    done: dict[str, Outcome] = {
        i.id: Outcome(i, False, "original_has_idiom", "") for i in items if index.find(i.x)
    }
    history: dict[str, list[str]] = {}
    for attempt in range(1, attempts + 1):
        prompts, lists = [], []
        for item in todo:
            open_ = [i for i in inventory if state.usage[i.idiom_id] < cap]
            fresh = [i for i in open_ if i.idiom_id not in state.shown.get(item.id, set())]
            # A retry shows idioms not seen before; only a tiny bank runs out, then all open ones.
            r = item_rng(seed, item.id, attempt) if per_item_seed else rng
            short = mixed_shortlist(item, fresh or open_, retriever, r)
            state.shown.setdefault(item.id, set()).update(i.idiom_id for i in short)
            lists.append(short)
            prompts.append(render_insert_prompt(item, short, template))
        failed: list[Item] = []
        for item, short, raw in zip(todo, lists, complete(prompts), strict=True):
            p = (
                validate_insert(raw, item.x, {i.idiom_id for i in short}, index)
                if raw is not None
                else Inserted(False, "no_reply")
            )
            if p.ok and state.usage[p.choice] < cap:
                state.usage[p.choice] += 1
                record = insert_record(
                    item,
                    by_id[p.choice],
                    p,
                    attempt=attempt,
                    model=model,
                    p_hash=p_hash,
                    shortlist_size=len(short),
                )
                done[item.id] = Outcome(
                    item,
                    True,
                    "ok",
                    "",
                    record=record,
                    idiomatic=p.idiomatic,
                    span_literal=p.literal,
                    attempts=attempt,
                    history=tuple(history.get(item.id, [])),
                )
            else:
                history.setdefault(item.id, []).append(p.reason if not p.ok else "idiom_cap")
                failed.append(item)
        todo = failed
    for item in todo:
        h = history.get(item.id, [])
        done[item.id] = Outcome(item, False, h[-1], "", attempts=attempts, history=tuple(h[:-1]))
    return [done[i.id] for i in items]
