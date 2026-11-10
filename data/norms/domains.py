"""Source-domain tags for inventory idioms (THESIS_PLAN_v2 §1, AG News cross-domain stratification).

One cheap LLM call per batch of idioms assigns each idiom the domain it is most associated with. The tags
only steer which idioms are offered for AG News items; they are not a result, and the author spot-checks a
sample.
"""

import csv
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOMAINS = ("sports", "finance-business", "war-politics", "science-tech", "everyday")
_BATCH = 40

PROMPT = """For each English idiom below, give the one domain it is most strongly associated with, judging by where \
the expression comes from or where it is typically used. Choose exactly one of:
- sports (games, athletics, betting on contests)
- finance-business (money, trade, commerce, work)
- war-politics (war, weapons, government, law, diplomacy)
- science-tech (science, medicine, machines, technology)
- everyday (no particular domain: body parts, emotions, daily life, animals, weather, general conversation)

Answer with one JSON object mapping each id to its domain, and nothing else.

Idioms (id: idiom - meaning):
{lines}
"""


@dataclass(frozen=True, slots=True)
class IdiomLine:
    idiom_id: str
    form: str
    meaning: str


def batches(lines: Sequence[IdiomLine], size: int = _BATCH) -> list[list[IdiomLine]]:
    return [list(lines[i : i + size]) for i in range(0, len(lines), size)]


def render(batch: Sequence[IdiomLine]) -> str:
    return PROMPT.format(lines="\n".join(f"{x.idiom_id}: {x.form} - {x.meaning}" for x in batch))


def parse_domains(raw: str, ids: Sequence[str]) -> dict[str, str]:
    """Valid id -> domain pairs from the model's JSON; anything else is dropped (and retried later)."""
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
    return {
        str(k): str(v).strip().lower()
        for k, v in obj.items()  # pyright: ignore[reportUnknownVariableType]
        if str(k) in wanted and str(v).strip().lower() in DOMAINS
    }


def tag_all(
    lines: Sequence[IdiomLine], complete: Callable[[str], str], *, max_rounds: int = 3
) -> dict[str, str]:
    """Tag every idiom, re-asking for the ones a round missed."""
    tags: dict[str, str] = {}
    pending = list(lines)
    for _ in range(max_rounds):
        if not pending:
            break
        for batch in batches(pending):
            tags.update(parse_domains(complete(render(batch)), [x.idiom_id for x in batch]))
        pending = [x for x in pending if x.idiom_id not in tags]
    return tags


def write_tags(path: Path, lines: Sequence[IdiomLine], tags: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idiom_id", "idiom", "domain"])
        for x in lines:
            w.writerow([x.idiom_id, x.form, tags.get(x.idiom_id, "")])
