"""SST-2 Evaluator implementation (STAGE3_PLAN §3.3)."""

from typing import Any, ClassVar, Literal

import eval.prompts.sst2 as sst2_prompts
from augmentation.base import AugmentedRow
from eval.base import BaseEvaluator, reasoning_output_truncated, strip_reasoning_trace
from eval.registry import register_evaluator
from models.base import FormattedInput


@register_evaluator("sst2")
class Sst2Evaluator(BaseEvaluator):
    """Evaluator for SST-2 sentiment classification task."""

    dataset = "sst2"
    system_prompt = sst2_prompts.SYSTEM
    user_template = sst2_prompts.USER_TEMPLATE
    _LABELS: ClassVar[dict[str, str]] = {"positive": "1", "negative": "0"}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gen_kwargs = {
            "temperature": self.cfg.decoding.temperature,
            "max_new_tokens": self.cfg.decoding.max_new_tokens,
            "do_sample": self.cfg.decoding.do_sample,
        }

    def format(self, ex: AugmentedRow) -> FormattedInput:
        messages = [
            {"role": "system", "content": sst2_prompts.SYSTEM},
            {"role": "user", "content": sst2_prompts.USER_TEMPLATE.format(x=ex.x)},
        ]
        return FormattedInput(
            id=ex.id,
            prompt="",
            meta={"messages": messages, "generate_kwargs": self._gen_kwargs},
        )

    def parse(self, raw: str, ex: AugmentedRow) -> tuple[str | None, Literal["ok", "unparseable"]]:
        if reasoning_output_truncated(raw, self.cfg.model):
            return None, "unparseable"
        s = strip_reasoning_trace(raw).strip().lower()
        pos_idx = s.find("positive")
        neg_idx = s.find("negative")

        if pos_idx != -1 and neg_idx != -1:
            if pos_idx < neg_idx:
                return self._LABELS["positive"], "ok"
            else:
                return self._LABELS["negative"], "ok"
        elif pos_idx != -1:
            return self._LABELS["positive"], "ok"
        elif neg_idx != -1:
            return self._LABELS["negative"], "ok"

        return None, "unparseable"
