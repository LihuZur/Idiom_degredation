"""Repair loop for v3 pairs the AI checker rejects (THESIS_PLAN_v3 §4 step 7; Kai's mail 6b).

A rejected item goes back to the generator with the checker's reason. The repair prompt is a separate file
(`idiom_repair_v1.txt`) that wraps the frozen generation prompt (`idiom_insert_v2.txt`) and appends the earlier
answer and the reason, so the frozen prompt is never edited and repaired rows carry their own prompt hash. The
shortlist is drawn fresh (its own seed per round) and always includes the earlier idiom, so the generator may
either fix it or switch. The reply passes the same code checks as first-try rows (`validate_insert`). Precedent
for an AI correction step before humans: ID10M-JAM (there the generating model itself validated; here the checker
is from another family). At most `ROUNDS` rounds; what is still rejected goes to a person.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from augmentation.checker import Verdict
from augmentation.idiom_generation import Idiom, Item
from augmentation.idiom_insertion import (
    TEMPLATE_FILE,
    Inserted,
    insert_prompt_hash,
    item_rng,
    mixed_shortlist,
    render_insert_prompt,
    validate_insert,
    words_kept,
)
from augmentation.idiom_retrieval import IdiomRetriever
from augmentation.prompts.loader import load_prompt
from data.norms.idiom_text import IdiomIndex

REPAIR_TEMPLATE = "idiom_repair_v1.txt"
ROUNDS = 2
_SEED_OFFSET = 100  # repair shortlists never reuse the first-try / second-try seeds (attempts 1, 2)

CompleteFn = Callable[[Sequence[str]], list[str | None]]


def repair_prompt_hash(dataset: str, template: str = TEMPLATE_FILE) -> str:
    text = insert_prompt_hash(dataset, template) + "\x1e" + load_prompt(REPAIR_TEMPLATE)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def item_from_v3(v3: dict[str, Any]) -> Item:
    """The generator's view of an item loaded with `augmentation.labelstudio.load_v3`."""
    return Item(v3["dataset"], v3["id"], v3["original"], v3["y"], dict(v3["original_meta"]))


def repair_reason(v: Verdict) -> str:
    """The checker's reason in one line: failed steps with their reasons, then the primary reason."""
    if not v.ok:
        return "the reviewer could not read the earlier answer; write a clean new answer"
    failed = [
        f"{s.replace('_', ' ')}: {v.reasons.get(s, '')}".strip()
        for s, x in v.steps.items()
        if x == "fail"
    ]
    parts = failed or [f"overall {v.action}"]
    if v.primary_reason and v.primary_reason != "none":
        parts.append(f"main problem: {v.primary_reason.replace('_', ' ')}")
    return "; ".join(parts)


def repair_shortlist(
    item: Item,
    inventory: Sequence[Idiom],
    retriever: IdiomRetriever | None,
    *,
    seed: int,
    round_: int,
    previous_idiom_id: str,
) -> list[Idiom]:
    short = mixed_shortlist(
        item, inventory, retriever, item_rng(seed, item.id, _SEED_OFFSET + round_)
    )
    if previous_idiom_id not in {i.idiom_id for i in short}:
        prev = next((i for i in inventory if i.idiom_id == previous_idiom_id), None)
        if prev is not None:
            short = [*short[:-1], prev] if short else [prev]
    return short


def render_repair_prompt(
    item: Item,
    shortlist: Sequence[Idiom],
    *,
    previous_idiomatic: str,
    previous_literal: str,
    reason: str,
    template: str = TEMPLATE_FILE,
) -> str:
    return load_prompt(REPAIR_TEMPLATE).format(
        base_prompt=render_insert_prompt(item, shortlist, template),
        previous_idiomatic=previous_idiomatic,
        previous_literal=previous_literal,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class RepairTask:
    key: str
    item: Item
    previous_idiomatic: str
    previous_literal: str
    previous_idiom_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class RepairResult:
    key: str
    round_: int
    inserted: Inserted
    idiom: Idiom | None  # the chosen idiom when the reply passed the code checks


def repair_round(
    tasks: Sequence[RepairTask],
    inventory: Sequence[Idiom],
    index: IdiomIndex,
    retriever: IdiomRetriever | None,
    complete: CompleteFn,
    *,
    seed: int,
    round_: int,
    template: str = TEMPLATE_FILE,
) -> list[RepairResult]:
    """One repair call per task; each reply runs through the same code checks as first-try rows."""
    shortlists = [
        repair_shortlist(
            t.item,
            inventory,
            retriever,
            seed=seed,
            round_=round_,
            previous_idiom_id=t.previous_idiom_id,
        )
        for t in tasks
    ]
    prompts = [
        render_repair_prompt(
            t.item,
            s,
            previous_idiomatic=t.previous_idiomatic,
            previous_literal=t.previous_literal,
            reason=t.reason,
            template=template,
        )
        for t, s in zip(tasks, shortlists, strict=True)
    ]
    out: list[RepairResult] = []
    for t, s, raw in zip(tasks, shortlists, complete(prompts), strict=True):
        p = (
            validate_insert(raw, t.item.x, {i.idiom_id for i in s}, index)
            if raw is not None
            else Inserted(False, "no_reply")
        )
        by_id = {i.idiom_id: i for i in s}
        out.append(RepairResult(t.key, round_, p, by_id.get(p.choice) if p.ok else None))
    return out


CheckFn = Callable[[Sequence[dict[str, Any]]], dict[str, Verdict]]


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    log: list[dict[str, Any]]  # one row per item and round: code check result, checker result
    repaired: list[dict[str, Any]]  # accepted repaired pairs
    unresolved: list[RepairTask]  # still rejected after the last round: for a person


def _candidate(base: dict[str, Any], x: RepairResult) -> dict[str, Any]:
    idiom = x.idiom
    assert idiom is not None
    return base | {
        "idiomatic": x.inserted.idiomatic,
        "literal": x.inserted.literal,
        "idiom": idiom.name,
        "definition": " / ".join(idiom.definitions),
        "idiom_text": x.inserted.idiom_text,
        "literal_text": x.inserted.literal_text,
        "idiom_id": idiom.idiom_id,
    }


def run_repair(
    tasks: Sequence[RepairTask],
    base_items: dict[str, dict[str, Any]],
    inventory: Sequence[Idiom],
    index: IdiomIndex,
    retriever: IdiomRetriever | None,
    gen_for: Callable[[str], CompleteFn],
    check: CheckFn,
    *,
    seed: int,
    rounds: int = ROUNDS,
) -> RepairOutcome:
    """Up to `rounds` rounds: regenerate (per dataset, `gen_for(dataset)`), code checks, re-check.

    A reply that fails a code check goes to the next round with that failure as its reason; a reply the
    checker rejects goes on with the checker's new reason and becomes the new "earlier answer".
    """
    log: list[dict[str, Any]] = []
    repaired: list[dict[str, Any]] = []
    todo = list(tasks)
    for r in range(1, rounds + 1):
        if not todo:
            break
        by_ds: dict[str, list[RepairTask]] = {}
        for t in todo:
            by_ds.setdefault(t.item.dataset, []).append(t)
        results = [
            res
            for d, ts in by_ds.items()
            for res in repair_round(
                ts, inventory, index, retriever, gen_for(d), seed=seed, round_=r
            )
        ]
        cands = {
            x.key: _candidate(base_items[x.key], x)
            for x in results
            if x.inserted.ok and x.idiom is not None
        }
        verdicts = check(list(cands.values())) if cands else {}
        task_by_key = {t.key: t for t in todo}
        next_todo: list[RepairTask] = []
        for x in results:
            t = task_by_key[x.key]
            if x.key not in cands:
                log.append(
                    {"key": x.key, "round": r, "code_check": x.inserted.reason, "checker": ""}
                )
                reason = f"the earlier repair failed an automatic check ({x.inserted.reason}); {t.reason}"
                next_todo.append(replace(t, reason=reason))
                continue
            v, c = verdicts[x.key], cands[x.key]
            log.append(
                {
                    "key": x.key,
                    "round": r,
                    "code_check": "ok",
                    "checker": "reject:" + (v.primary_reason or v.error) if v.rejects else "accept",
                }
            )
            if v.rejects:
                next_todo.append(
                    RepairTask(
                        t.key, t.item, c["idiomatic"], c["literal"], c["idiom_id"], repair_reason(v)
                    )
                )
                continue
            keep = (
                "key",
                "dataset",
                "id",
                "idiomatic",
                "literal",
                "idiom",
                "idiom_text",
                "literal_text",
            )
            repaired.append(
                {k: c[k] for k in keep}
                | {
                    "round": r,
                    "prompt_hash": repair_prompt_hash(c["dataset"]),
                    "original_words_kept": round(words_kept(c["original"], c["idiomatic"]), 3),
                }
            )
        todo = next_todo
    return RepairOutcome(log, repaired, todo)
