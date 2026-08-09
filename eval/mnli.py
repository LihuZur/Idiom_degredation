"""MNLI Evaluator implementation (MNLI_DATASET_PLAN §5.2)."""

from typing import Any, ClassVar, Literal

import eval.prompts.mnli as mnli_prompts
from augmentation.base import AugmentedRow
from eval.base import BaseEvaluator, reasoning_output_truncated, strip_reasoning_trace
from eval.registry import register_evaluator
from models.base import FormattedInput


@register_evaluator("mnli")
class MnliEvaluator(BaseEvaluator):
    """Evaluator for MNLI 3-way natural language inference (D1)."""

    dataset = "mnli"
    system_prompt = mnli_prompts.SYSTEM
    user_template = mnli_prompts.USER_TEMPLATE
    # Keys are the label words scanned for; values are the `y` codes written by
    # `data/mnli.py` (MNLI's ClassLabel order).
    _LABELS: ClassVar[dict[str, str]] = {
        "entailment": "0",
        "neutral": "1",
        "contradiction": "2",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._gen_kwargs = {
            "temperature": self.cfg.decoding.temperature,
            "max_new_tokens": self.cfg.decoding.max_new_tokens,
            "do_sample": self.cfg.decoding.do_sample,
        }

    def format(self, ex: AugmentedRow) -> FormattedInput:
        messages = [
            {"role": "system", "content": mnli_prompts.SYSTEM},
            {
                "role": "user",
                "content": mnli_prompts.USER_TEMPLATE.format(
                    premise=ex.x,
                    hypothesis=ex.meta["hypothesis"],
                ),
            },
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
        # R10: three label words, so resolve by an explicit earliest-index scan
        # rather than SST-2's two-way pairwise compare.
        earliest_label: str | None = None
        earliest_idx = len(s)
        for word, code in self._LABELS.items():
            idx = s.find(word)
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
                earliest_label = code

        if earliest_label is None:
            return None, "unparseable"
        return earliest_label, "ok"
