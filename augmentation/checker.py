"""AI checker for v3 pairs: prompt, reply parsing, dev / test split and agreement with human labels.

THESIS_PLAN_v3 §4 steps 3-5. The checker applies the annotation guidelines
(`paper_preparations/annotation_guidelines.md`) with a model from another family than the generator (Kai's mail
6a; precedent IdioLink §3.2.4). Candidate model x prompt pairs are compared with human labels on dev items and
the chosen pair is frozen before it is scored once on held-out test items (selection procedure from IdiomSteer
App. J; the held-out test is our addition). The checker sees the label and the fixed context (MMLU choices, MNLI
hypothesis), otherwise it could not catch answer hints or label changes. Prompt variants are template files;
which models to try is open (plan O18), so this module never picks one.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from augmentation.idiom_generation import extract_json
from augmentation.labelstudio import V3_ACTIONS, V3_NATURALNESS, V3_REASONS, V3_STEPS
from augmentation.prompts.loader import load_prompt

TEMPLATES = ("checker_v1_direct.txt", "checker_v1_structured.txt")
BAD_ACTIONS = frozenset({"fix", "invalid"})
_LETTERS = "ABCDEFGH"
TASK_NOTES = {
    "sst2": "Task: the text is a sentence from a movie review; the label is its sentiment.",
    "mnli": "Task: the text is the premise of an inference pair; the label is its relation to the fixed "
    "hypothesis (entailment, neutral or contradiction).",
    "mmlu": "Task: the text is the question of a multiple-choice item; the label is the correct answer among "
    "the fixed choices.",
    "agnews": "Task: the text is a news snippet (title, then description); the label is its topic (World, "
    "Sports, Business or Sci/Tech).",
}

CompleteFn = Callable[[Sequence[str]], list[str | None]]


def checker_hash(template: str) -> str:
    """Cache namespace for one prompt variant."""
    return hashlib.sha256(("checker\x1e" + load_prompt(template)).encode("utf-8")).hexdigest()[:16]


def _context(item: dict[str, Any]) -> str:
    if item["dataset"] == "mnli" and item.get("hypothesis"):
        return f"<hypothesis_never_changed>{item['hypothesis']}</hypothesis_never_changed>\n"
    if item["dataset"] == "mmlu" and item.get("choices"):
        opts = " ".join(f"{_LETTERS[n]}) {c}" for n, c in enumerate(item["choices"]))
        return f"<answer_choices_never_changed>{opts}</answer_choices_never_changed>\n"
    return ""


def render_checker_prompt(item: dict[str, Any], template: str) -> str:
    """Fill one prompt variant with an item from `augmentation.labelstudio.load_v3`."""
    return load_prompt(template).format(
        task_note=TASK_NOTES[item["dataset"]],
        context=_context(item),
        label=item["label"],
        original=item["original"],
        idiomatic=item["idiomatic"],
        literal=item["literal"],
        idiom=item["idiom"],
        definition=item["definition"],
        idiom_text=item["idiom_text"],
        literal_text=item["literal_text"],
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool  # False: the reply could not be parsed into the required fields
    steps: dict[str, str] = field(default_factory=dict)  # step -> "pass" / "fail"
    reasons: dict[str, str] = field(default_factory=dict)
    naturalness: str = ""
    action: str = ""
    primary_reason: str = ""
    error: str = ""

    @property
    def rejects(self) -> bool:
        """The checker calls the item bad: action fix / invalid, or any step failed. Unparsable = reject."""
        return (not self.ok) or self.action in BAD_ACTIONS or "fail" in self.steps.values()


def parse_verdict(raw: str | None) -> Verdict:
    if raw is None:
        return Verdict(False, error="no_reply")
    obj = extract_json(raw)
    if obj is None:
        return Verdict(False, error="bad_json")
    steps: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for s in V3_STEPS:
        part = obj.get(s)
        verdict = (
            str(part.get("verdict", "") if isinstance(part, dict) else part or "").strip().lower()
        )
        if verdict not in ("pass", "fail"):
            return Verdict(False, error=f"bad_{s}")
        steps[s] = verdict
        reasons[s] = str(part.get("reason", "")).strip() if isinstance(part, dict) else ""
    nat = str(obj.get("naturalness", "")).strip().lower()
    action = str(obj.get("action", "")).strip().lower()
    reason = str(obj.get("primary_reason", "")).strip().lower()
    if nat not in V3_NATURALNESS:
        return Verdict(False, error="bad_naturalness")
    if action not in V3_ACTIONS:
        return Verdict(False, error="bad_action")
    if reason not in (*V3_REASONS, "none"):
        reason = "other"
    return Verdict(True, steps, reasons, nat, action, reason)


def run_checker(
    items: Sequence[dict[str, Any]], complete: CompleteFn, template: str
) -> dict[str, tuple[str | None, Verdict]]:
    """key -> (raw reply, parsed verdict) for every item."""
    replies = complete([render_checker_prompt(i, template) for i in items])
    return {i["key"]: (r, parse_verdict(r)) for i, r in zip(items, replies, strict=True)}


def human_bad(row: dict[str, str]) -> bool:
    """The human gold decision the checker is scored on: action fix or invalid (guidelines §4)."""
    return row["action"] in BAD_ACTIONS


def split_dev_test(
    human: dict[str, dict[str, str]], seed: int, dev_share: float = 0.5
) -> dict[str, str]:
    """key -> "dev" / "test", per dataset and separately for bad and good items, so both splits get bad ones."""
    groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for key, row in sorted(human.items()):
        groups[(row["dataset"], human_bad(row))].append(key)
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for _, keys in sorted(groups.items()):
        rng.shuffle(keys)
        n_dev = round(len(keys) * dev_share)
        out.update(dict.fromkeys(keys[:n_dev], "dev"))
        out.update(dict.fromkeys(keys[n_dev:], "test"))
    return out


def _kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def compare(
    verdicts: dict[str, Verdict], human: dict[str, dict[str, str]], keys: Sequence[str]
) -> dict[str, Any]:
    """Checker vs human on `keys` ("bad" = should be rejected). Key number first (IdioLink §3.2.4): bad items
    the checker passes."""
    keys = [k for k in keys if k in verdicts and k in human]
    if not keys:
        return {"n": 0}
    h = [human_bad(human[k]) for k in keys]
    c = [verdicts[k].rejects for k in keys]
    tp = sum(x and y for x, y in zip(h, c, strict=True))
    fn = sum(x and not y for x, y in zip(h, c, strict=True))
    fp = sum(not x and y for x, y in zip(h, c, strict=True))
    tn = len(keys) - tp - fn - fp
    return {
        "n": len(keys),
        "human_bad": tp + fn,
        "bad_passed": fn,
        "bad_passed_share_of_items": fn / len(keys),
        "recall_bad": tp / (tp + fn) if tp + fn else float("nan"),
        "precision_reject": tp / (tp + fp) if tp + fp else float("nan"),
        "good_rejected": fp,
        "agreement": (tp + tn) / len(keys),
        "cohen_kappa": _kappa(h, c),
        "unparsed": sum(not verdicts[k].ok for k in keys),
    }


def disagreements(
    verdicts: dict[str, Verdict], human: dict[str, dict[str, str]], keys: Sequence[str]
) -> list[dict[str, str]]:
    """Items where checker and human differ, bad items passed first: what to read when editing the prompt."""
    out: list[dict[str, str]] = []
    for k in keys:
        if k not in verdicts or k not in human:
            continue
        v, hb = verdicts[k], human_bad(human[k])
        if v.rejects == hb:
            continue
        out.append(
            {
                "key": k,
                "kind": "bad_passed" if hb else "good_rejected",
                "human_action": human[k]["action"],
                "human_reason": human[k].get("reason", ""),
                "checker_action": v.action or v.error,
                "checker_reason": v.primary_reason,
                "checker_failed_steps": " ".join(s for s, x in v.steps.items() if x == "fail"),
                "checker_reasons": " | ".join(f"{s}: {r}" for s, r in v.reasons.items() if r),
            }
        )
    return sorted(out, key=lambda r: (r["kind"] != "bad_passed", r["key"]))


def per_dataset(
    verdicts: dict[str, Verdict], human: dict[str, dict[str, str]], keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    by: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        if k in human:
            by[human[k]["dataset"]].append(k)
    return {d: compare(verdicts, human, ks) for d, ks in sorted(by.items())}
