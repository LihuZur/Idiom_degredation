"""MMLU Evaluator implementation (STAGE3_PLAN §3.3)."""

import re
from typing import Any, Literal

import eval.prompts.mmlu as mmlu_prompts
from augmentation.base import AugmentedRow
from eval.base import BaseEvaluator, strip_reasoning_trace
from eval.registry import register_evaluator
from models.base import FormattedInput


@register_evaluator("mmlu")
class MmluEvaluator(BaseEvaluator):
    """Evaluator for MMLU multiple-choice QA task."""

    dataset = "mmlu"
    system_prompt = mmlu_prompts.SYSTEM
    user_template = mmlu_prompts.USER_TEMPLATE

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gen_kwargs = {
            "temperature": self.cfg.decoding.temperature,
            "max_new_tokens": self.cfg.decoding.max_new_tokens,
            "do_sample": self.cfg.decoding.do_sample,
        }

    def format(self, ex: AugmentedRow) -> FormattedInput:
        choices = ex.meta["choices"]
        messages = [
            {"role": "system", "content": mmlu_prompts.SYSTEM},
            {
                "role": "user",
                "content": mmlu_prompts.USER_TEMPLATE.format(
                    question=ex.x,
                    A=choices[0],
                    B=choices[1],
                    C=choices[2],
                    D=choices[3],
                ),
            },
        ]
        return FormattedInput(
            id=ex.id,
            prompt="",
            meta={"messages": messages, "generate_kwargs": self._gen_kwargs},
        )

    def parse(self, raw: str, ex: AugmentedRow) -> tuple[str | None, Literal["ok", "unparseable"]]:
        s = strip_reasoning_trace(raw).strip().upper()
        # Find first standalone character A, B, C, or D using word boundaries
        match = re.search(r"\b([A-D])\b", s)
        if match:
            char = match.group(1)
            mapping = {"A": "0", "B": "1", "C": "2", "D": "3"}
            return mapping[char], "ok"

        return None, "unparseable"
