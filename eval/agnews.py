"""AG News Evaluator: four-way topic classification (World / Sports / Business / Sci/Tech)."""

import re
from typing import Any, ClassVar, Literal

import eval.prompts.agnews as agnews_prompts
from augmentation.base import AugmentedRow
from eval.base import BaseEvaluator, strip_reasoning_trace
from eval.registry import register_evaluator
from models.base import FormattedInput

# Label codes follow the HF ClassLabel order (data/agnews.py LABEL_NAMES).
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("0", re.compile(r"\bworld\b")),
    ("1", re.compile(r"\bsports?\b")),
    ("2", re.compile(r"\bbusiness\b")),
    ("3", re.compile(r"\b(?:sci/tech|sci-tech|science|technology|sci|tech)\b")),
)


@register_evaluator("agnews")
class AgNewsEvaluator(BaseEvaluator):
    dataset = "agnews"
    system_prompt = agnews_prompts.SYSTEM
    user_template = agnews_prompts.USER_TEMPLATE
    _LABELS: ClassVar[dict[str, str]] = {
        "world": "0",
        "sports": "1",
        "business": "2",
        "sci/tech": "3",
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
            {"role": "system", "content": agnews_prompts.SYSTEM},
            {"role": "user", "content": agnews_prompts.USER_TEMPLATE.format(x=ex.x)},
        ]
        return FormattedInput(
            id=ex.id,
            prompt="",
            meta={"messages": messages, "generate_kwargs": self._gen_kwargs},
        )

    def parse(self, raw: str, ex: AugmentedRow) -> tuple[str | None, Literal["ok", "unparseable"]]:
        s = strip_reasoning_trace(raw).strip().lower()
        best: tuple[int, str] | None = None
        for code, pattern in _PATTERNS:
            m = pattern.search(s)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), code)
        if best is None:
            return None, "unparseable"
        return best[1], "ok"
