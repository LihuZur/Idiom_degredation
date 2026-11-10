"""Kind and domain tags for the MAGPIE extension tier (THESIS_PLAN_v2 §3.1 fallback).

MAGPIE attests that an expression is common, not that it is figurative: its most frequent types
include fixed phrases such as "in the long run" or "bear in mind", which annotators and the judge
would reject under the criterion "idiom present and used figuratively". One cheap LLM pass labels
each candidate's kind (figurative / fixed_phrase / obscure) and, in the same call, its source domain
(the `domains` tags used for AG News). Only figurative idioms enter the tier; the author spot-checks
a sample and can skim the rejected list.
"""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from data.norms.domains import DOMAINS, IdiomLine, batches

KINDS = ("figurative", "fixed_phrase", "obscure")

PROMPT = """For each English expression below, give two labels.

kind (exactly one):
- figurative: a real idiom whose meaning is not the plain sum of its words; it uses an image, metaphor \
or non-literal picture (e.g. "spill the beans", "take a back seat", "under lock and key")
- fixed_phrase: a common set phrase whose meaning is basically literal or that works as a connector or \
hedge, with no figurative image (e.g. "in the long run", "bear in mind", "for the time being", "in light of")
- obscure: figurative but not widely understood by ordinary American English speakers today (e.g. dated \
or regional British expressions)

domain (exactly one), judging by where the expression comes from or where it is typically used:
- sports (games, athletics, betting on contests)
- finance-business (money, trade, commerce, work)
- war-politics (war, weapons, government, law, diplomacy)
- science-tech (science, medicine, machines, technology)
- everyday (no particular domain: body parts, emotions, daily life, animals, weather, general conversation)

Answer with one JSON object mapping each id to {{"kind": "...", "domain": "..."}}, and nothing else.

Expressions (id: expression - meaning):
{lines}
"""


def render(batch: Sequence[IdiomLine]) -> str:
    return PROMPT.format(lines="\n".join(f"{x.idiom_id}: {x.form} - {x.meaning}" for x in batch))


def parse_tags(raw: str, ids: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Valid id -> (kind, domain) pairs; anything else is dropped (and retried later)."""
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match is None:
        return {}
    try:
        obj: Any = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    wanted = set(ids)
    out: dict[str, tuple[str, str]] = {}
    for k, v in obj.items():  # pyright: ignore[reportUnknownVariableType]
        if str(k) not in wanted or not isinstance(v, dict):
            continue
        kind = str(v.get("kind", "")).strip().lower()  # pyright: ignore[reportUnknownMemberType]
        domain = str(v.get("domain", "")).strip().lower()  # pyright: ignore[reportUnknownMemberType]
        if kind in KINDS and domain in DOMAINS:
            out[str(k)] = (kind, domain)
    return out


def tag_all(
    lines: Sequence[IdiomLine], complete: Callable[[str], str], *, max_rounds: int = 3
) -> dict[str, tuple[str, str]]:
    """Tag every candidate, re-asking for the ones a round missed."""
    tags: dict[str, tuple[str, str]] = {}
    pending = list(lines)
    for _ in range(max_rounds):
        if not pending:
            break
        for batch in batches(pending):
            tags.update(parse_tags(complete(render(batch)), [x.idiom_id for x in batch]))
        pending = [x for x in pending if x.idiom_id not in tags]
    return tags
