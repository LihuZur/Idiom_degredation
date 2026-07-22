"""Anthropic Claude augmenter registration (D1).

Thin `@register_augmenter` subclass so `anthropic` is selectable via
`cfg.augmenter`; the shared `LLMClient` is injected by the pipeline (D6).
"""

from augmentation.llm_augmenter import LLMAugmenter
from augmentation.registry import register_augmenter


@register_augmenter("anthropic")
class AnthropicAugmenter(LLMAugmenter):
    """Augmenter backed by Anthropic Claude (`anthropic`)."""

    provider: str = "anthropic"
