"""OpenAI GPT augmenter registration (D1).

Thin `@register_augmenter` subclass so `openai` is selectable via
`cfg.augmenter`; the shared `LLMClient` is injected by the pipeline (D6).
"""

from augmentation.llm_augmenter import LLMAugmenter
from augmentation.registry import register_augmenter


@register_augmenter("openai")
class OpenAIAugmenter(LLMAugmenter):
    """Augmenter backed by OpenAI GPT (`openai`)."""

    provider: str = "openai"
