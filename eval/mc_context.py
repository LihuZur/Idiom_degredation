"""Multiple-choice questions about a short context (MMLU-replacement baseline, plan v3 O47, 2026-11-18).

`<source>_base`: context + question + options; `<source>_noctx`: the same question and options without the context
(how much the models answer from memory or from the question alone). The answer is one letter.
"""

import re
from typing import Any, ClassVar, Literal

from augmentation.base import AugmentedRow
from eval.base import BaseEvaluator, strip_reasoning_trace
from eval.registry import register_evaluator
from models.base import FormattedInput

_LETTERS = "ABCDE"


def letters_for(n: int) -> str:
    return _LETTERS[:n]


def build_system(n: int, with_context: bool) -> str:
    opts = ", ".join(letters_for(n)[:-1]) + f", or {letters_for(n)[-1]}"
    what = "a short text, a question about it," if with_context else "a question"
    return (
        f"You are answering a multiple-choice question. Read {what} and the choices, "
        f"then respond with exactly one letter: {opts}."
    )


def build_user(context: str, question: str, choices: list[str], with_context: bool) -> str:
    lines = [f"Text: {context}"] if with_context else []
    lines.append(f"Question: {question}")
    lines += [f"{letters_for(len(choices))[k]}. {c}" for k, c in enumerate(choices)]
    lines.append("Answer:")
    return "\n".join(lines)


def parse_letter(raw: str, n: int) -> str | None:
    """0-based index of the chosen option, or None. Same rules as the MMLU parser, restricted to n letters."""
    text = strip_reasoning_trace(raw).strip()
    allowed = letters_for(n)
    if whole := re.fullmatch(rf"\(?([{allowed}{allowed.lower()}])[).:]?", text):
        return str(allowed.index(whole.group(1).upper()))
    if lead := re.match(rf"^\(?([{allowed}])(?:[).:,\s]|$|(?=[A-Z]))", text):
        return str(allowed.index(lead.group(1)))
    if m := re.search(rf"\b([{allowed}])\b", text):
        return str(allowed.index(m.group(1)))
    return None


class _McContextEvaluator(BaseEvaluator):
    with_context: ClassVar[bool] = True
    # Stored in each result's prompt hash (scripts/eval.py); the text is built per item by build_system/build_user.
    system_prompt: ClassVar[str] = (
        "mc_context v1: " + build_system(3, True) + " | " + build_system(4, False)
    )
    user_template: ClassVar[str] = "Text: {context}\nQuestion: {question}\nA. ..\nAnswer:"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gen_kwargs = {
            "temperature": self.cfg.decoding.temperature,
            "max_new_tokens": self.cfg.decoding.max_new_tokens,
            "do_sample": self.cfg.decoding.do_sample,
        }

    def format(self, ex: AugmentedRow) -> FormattedInput:
        choices = list(ex.meta["choices"])
        messages = [
            {"role": "system", "content": build_system(len(choices), self.with_context)},
            {
                "role": "user",
                "content": build_user(ex.x, str(ex.meta["question"]), choices, self.with_context),
            },
        ]
        return FormattedInput(
            id=ex.id, prompt="", meta={"messages": messages, "generate_kwargs": self._gen_kwargs}
        )

    def parse(self, raw: str, ex: AugmentedRow) -> tuple[str | None, Literal["ok", "unparseable"]]:
        idx = parse_letter(raw, len(ex.meta["choices"]))
        return (idx, "ok") if idx is not None else (None, "unparseable")


@register_evaluator("siqa_base")
class SiqaBaseEvaluator(_McContextEvaluator):
    dataset = "siqa_base"


@register_evaluator("siqa_noctx")
class SiqaNoCtxEvaluator(_McContextEvaluator):
    dataset = "siqa_noctx"
    with_context = False


@register_evaluator("cosmosqa_base")
class CosmosQaBaseEvaluator(_McContextEvaluator):
    dataset = "cosmosqa_base"


@register_evaluator("cosmosqa_noctx")
class CosmosQaNoCtxEvaluator(_McContextEvaluator):
    dataset = "cosmosqa_noctx"
    with_context = False


# POC-3 extension (2026-11-18): Social IQa pairs from the generator; x = the idiomatic / literal / original story.
@register_evaluator("siqa_v3poc3")
class SiqaV3Poc3Evaluator(_McContextEvaluator):
    dataset = "siqa_v3poc3"
