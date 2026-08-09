"""Load frozen prompt templates and render their read-only `{context}` slot."""

import string
from pathlib import Path
from typing import Any, cast


def load_prompt(filename: str) -> str:
    """Read a frozen prompt template's raw text from this package's directory."""
    return (Path(__file__).parent / filename).read_text(encoding="utf-8")


# R9: self-describing, because `render_context` is the *only* channel that reaches
# an LLM judge — `JUDGE_PROMPTS` is a hardcoded dict in `augmentation/llm_validators.py`
# and cannot be varied per dataset, so a constraint stated in a variant template
# would never be seen by the judge that has to enforce it.
_HYPOTHESIS_BLOCK = (
    "Hypothesis (fixed reference — it is never rewritten; the rewritten text must\n"
    "preserve exactly the same logical relationship to it, and must not restate,\n"
    "quote, negate, or otherwise give away that relationship):\n"
    "{hypothesis}"
)


def render_context(meta: dict[str, Any]) -> str:
    """Render read-only reference context for an augmenter prompt's `{context}` slot.

    Strictly allowlist-based: only the keys handled below are ever rendered, so
    bookkeeping fields (MMLU's `answer_index`/`subject`, MNLI's `prompt_id`/`genre`)
    can never leak into a prompt. Returns `""` when a dataset carries none of them
    (e.g. SST-2), so the template slot collapses. Rendered sections, in order:

    - `choices` — multiple-choice options (e.g. MMLU), so the rewritten stem stays
      consistent with options Stage 2 never rewrites (`meta` is preserved verbatim).
    - `hypothesis` — the fixed half of a sentence-pair item (e.g. MNLI). Required:
      an entailment label is a *relation* between premise and hypothesis, so the
      `label_preservation` judge cannot decide one without seeing both sides.
    - `label_name` — the gold label in words, for datasets whose `y` is an opaque
      integer code.

    Note on leakage: `answer_index` is omitted from the *context*, but the gold
    label is deliberately supplied to the augmenter through the template's `{y}`
    slot — it has to know the label in order to preserve it. What guards against
    the rewrite encoding the answer is the explicit instruction in the variant
    templates plus the fact that Stage 2 only ever rewrites `x`.
    """
    sections: list[str] = []

    choices = meta.get("choices")
    if isinstance(choices, list) and choices:
        lines = [
            f"{letter}) {choice}"
            for letter, choice in zip(
                string.ascii_uppercase, cast(list[Any], choices), strict=False
            )
        ]
        sections.append("Answer choices:\n" + "\n".join(lines))

    hypothesis = meta.get("hypothesis")
    if isinstance(hypothesis, str) and hypothesis:
        sections.append(_HYPOTHESIS_BLOCK.format(hypothesis=hypothesis))

    label_name = meta.get("label_name")
    if isinstance(label_name, str) and label_name:
        sections.append(f"Gold label in words: {label_name}")

    if not sections:
        return ""
    return "\n\n".join(sections) + "\n\n"
