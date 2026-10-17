"""Load frozen prompt templates for analysis-time LLM judges."""

from pathlib import Path
from typing import Any


def load_prompt(filename: str) -> str:
    """Read a frozen prompt template's raw text from this package's directory."""
    return (Path(__file__).parent / filename).read_text(encoding="utf-8")


def render_context_block(meta: dict[str, Any]) -> str:
    """Render fixed reference material (MNLI hypothesis, MMLU choices) for a judge prompt.

    Returns an empty string when the row carries none of the allowlisted keys
    (e.g. SST-2), so the template slot collapses cleanly. Mirrors the strict
    allowlist approach in `augmentation/prompts/loader.py::render_context`.
    """
    sections: list[str] = []

    hypothesis = meta.get("hypothesis")
    if isinstance(hypothesis, str) and hypothesis.strip():
        sections.append(
            "CONTEXT (fixed hypothesis — same for ORIGINAL and REWRITE; the "
            "entailment relationship must be preserved):\n"
            f"{hypothesis}"
        )

    premise = meta.get("premise")
    if isinstance(premise, str) and premise.strip():
        sections.append(
            "CONTEXT (fixed premise — same for ORIGINAL and REWRITE; the "
            "entailment relationship must be preserved):\n"
            f"{premise}"
        )

    choices = meta.get("choices")
    if isinstance(choices, list) and choices:
        rendered = "\n".join(f"  {i}. {c}" for i, c in enumerate(choices) if isinstance(c, str))
        answer_index = meta.get("answer_index")
        if isinstance(answer_index, int):
            sections.append(
                "CONTEXT (fixed answer choices — same for ORIGINAL and REWRITE; "
                f"the correct choice remains index {answer_index}):\n{rendered}"
            )
        else:
            sections.append(
                f"CONTEXT (fixed answer choices — same for ORIGINAL and REWRITE):\n{rendered}"
            )

    question = meta.get("question")
    if isinstance(question, str) and question.strip():
        sections.append(
            "CONTEXT (fixed question — same for ORIGINAL and REWRITE; the "
            "rewritten choices must still answer it the same way, with the "
            "correct choice still uniquely correct):\n"
            f"{question}"
        )

    if not sections:
        return ""
    return "\n\n".join(sections) + "\n\n"
