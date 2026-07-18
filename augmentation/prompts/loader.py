"""Load frozen prompt templates and render their read-only `{context}` slot."""

import string
from pathlib import Path
from typing import Any


def load_prompt(filename: str) -> str:
    """Read a frozen prompt template's raw text from this package's directory."""
    return (Path(__file__).parent / filename).read_text(encoding="utf-8")


def render_context(meta: dict[str, Any]) -> str:
    """Render read-only reference context for an augmenter prompt's `{context}` slot.

    For multiple-choice items (e.g. MMLU) this exposes the answer choices so the
    rewritten stem stays consistent with them. The choices are never rewritten
    (Stage 2 preserves `meta` verbatim), and the correct-answer index is
    deliberately omitted so it cannot leak into the prompt. Returns `""` when
    there is no such context (e.g. SST-2), so the template slot collapses.
    """
    choices = meta.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    lines = [
        f"{letter}) {choice}"
        for letter, choice in zip(string.ascii_uppercase, choices, strict=False)
    ]
    return "Answer choices:\n" + "\n".join(lines) + "\n\n"
