"""Gemini-backed augmenter registration (default provider, D5).

Thin `@register_augmenter` subclass mirroring the `models/*.py` pattern: it
exists so `gemini` is selectable purely via `cfg.augmenter`. The shared
`LLMClient` (used by both augmenter and judge, D6) is built by the pipeline and
injected, so no provider-specific logic lives here.
"""

from augmentation.llm_augmenter import LLMAugmenter
from augmentation.registry import register_augmenter


@register_augmenter("gemini")
class GeminiAugmenter(LLMAugmenter):
    """Augmenter backed by Google Gemini (`google-genai`)."""

    provider: str = "gemini"
