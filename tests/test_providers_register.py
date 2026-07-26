"""Tests that the hosted-LLM augmenters register themselves (and identity does not)."""

from augmentation.anthropic_augmenter import AnthropicAugmenter
from augmentation.gemini_augmenter import GeminiAugmenter
from augmentation.openai_augmenter import OpenAIAugmenter
from augmentation.registry import get_augmenter, list_augmenters


def test_registered_augmenters_are_exactly_the_three_providers() -> None:
    assert list_augmenters() == ["anthropic", "gemini", "openai"]
    assert "identity" not in list_augmenters()


def test_get_augmenter_maps_to_correct_provider_classes() -> None:
    assert get_augmenter("gemini") is GeminiAugmenter
    assert get_augmenter("anthropic") is AnthropicAugmenter
    assert get_augmenter("openai") is OpenAIAugmenter
