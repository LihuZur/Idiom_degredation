"""Tests for augmentation/registry.py (STAGE2_CONTRACT REGISTRY API section).

Registers fake augmenters/validators under unique names per-test and cleans
them up via an autouse fixture (mirrors tests/test_pipeline.py) so they don't
leak into tests/test_smoke.py's exact-registry assertions.
"""

import itertools
from collections.abc import Iterable

import pytest

# Registering the three hosted-LLM augmenters here (rather than only in
# tests/test_providers_register.py) is what makes tests/test_smoke.py's
# exact-registry assertion hold even when this file's tests run standalone
# (module imports execute once, at collection time, and are cached thereafter).
from augmentation.anthropic_augmenter import AnthropicAugmenter
from augmentation.base import AugmentedRow, ValidationResult, Variant
from augmentation.gemini_augmenter import GeminiAugmenter
from augmentation.llm_validators import (
    IdiomAbsenceValidator,
    IdiomPresenceValidator,
    LabelPreservationValidator,
)
from augmentation.openai_augmenter import OpenAIAugmenter
from augmentation.providers.base import LLMClient
from augmentation.registry import (
    AUGMENTERS,
    VALIDATORS,
    get_augmenter,
    get_validator,
    list_augmenters,
    list_validators,
    register_augmenter,
    register_validator,
)
from augmentation.validators import SemanticSimilarityValidator
from data.base import DatasetRow

_counter = itertools.count()


@pytest.fixture(autouse=True)
def _cleanup_fake_registrations() -> Iterable[None]:  # pyright: ignore[reportUnusedFunction]
    """Remove any fake augmenters/validators registered by this file."""
    before_augmenters = set(AUGMENTERS)
    before_validators = set(VALIDATORS)
    yield
    for name in set(AUGMENTERS) - before_augmenters:
        del AUGMENTERS[name]
    for name in set(VALIDATORS) - before_validators:
        del VALIDATORS[name]


def _fake_validator_name() -> str:
    return f"fake_validator_{next(_counter)}"


def _fake_augmenter_name() -> str:
    return f"fake_augmenter_{next(_counter)}"


def test_registered_augmenters_and_validators() -> None:
    assert {"anthropic", "gemini", "openai"} <= set(list_augmenters())
    assert "identity" not in list_augmenters()
    assert get_augmenter("anthropic") is AnthropicAugmenter
    assert get_augmenter("gemini") is GeminiAugmenter
    assert get_augmenter("openai") is OpenAIAugmenter
    assert set(list_validators()) >= {
        "semantic_similarity",
        "label_preservation",
        "idiom_presence",
        "idiom_absence",
    }
    assert get_validator("semantic_similarity") is SemanticSimilarityValidator
    assert get_validator("label_preservation") is LabelPreservationValidator
    assert get_validator("idiom_presence") is IdiomPresenceValidator
    assert get_validator("idiom_absence") is IdiomAbsenceValidator


def test_register_and_get_validator() -> None:
    validator_name = _fake_validator_name()

    class _FakeValidator:
        name: str = validator_name

        def validate(self, ex: AugmentedRow, original: DatasetRow) -> ValidationResult:
            return ValidationResult(name=self.name, passed=True)

    register_validator(validator_name)(_FakeValidator)
    assert get_validator(validator_name) is _FakeValidator
    assert validator_name in list_validators()


def test_register_and_get_augmenter() -> None:
    name = _fake_augmenter_name()

    class _FakeAugmenter:
        variant: Variant
        augmenter_model: str = name

        def __init__(
            self,
            *,
            variant: Variant,
            prompt_hash: str,
            client: LLMClient,
            prompt_template: str,
            temperature: float,
            max_output_tokens: int,
        ) -> None:
            self.variant = variant
            self._prompt_hash = prompt_hash

        def augment(self, ex: DatasetRow) -> AugmentedRow:
            return AugmentedRow(
                id=ex.id,
                variant=self.variant,
                x=ex.x,
                y=ex.y,
                augmenter_model=self.augmenter_model,
                prompt_hash=self._prompt_hash,
                meta=dict(ex.meta),
            )

    register_augmenter(name)(_FakeAugmenter)
    assert get_augmenter(name) is _FakeAugmenter
    assert name in list_augmenters()


def test_duplicate_validator_register_raises() -> None:
    validator_name = _fake_validator_name()

    class _FakeValidator:
        name: str = validator_name

        def validate(self, ex: AugmentedRow, original: DatasetRow) -> ValidationResult:
            return ValidationResult(name=self.name, passed=True)

    register_validator(validator_name)(_FakeValidator)
    with pytest.raises(ValueError, match="already registered"):
        register_validator(validator_name)(_FakeValidator)


def test_duplicate_augmenter_register_raises() -> None:
    name = _fake_augmenter_name()

    class _FakeAugmenter:
        variant: Variant
        augmenter_model: str = name

        def __init__(
            self,
            *,
            variant: Variant,
            prompt_hash: str,
            client: LLMClient,
            prompt_template: str,
            temperature: float,
            max_output_tokens: int,
        ) -> None:
            self.variant = variant

        def augment(self, ex: DatasetRow) -> AugmentedRow:
            raise NotImplementedError

    register_augmenter(name)(_FakeAugmenter)
    with pytest.raises(ValueError, match="already registered"):
        register_augmenter(name)(_FakeAugmenter)


def test_get_validator_unknown_raises_keyerror_with_hint() -> None:
    with pytest.raises(KeyError) as exc_info:
        get_validator("definitely_not_a_real_validator")
    message = str(exc_info.value)
    assert "definitely_not_a_real_validator" in message
    assert "known" in message


def test_get_augmenter_unknown_raises_keyerror_with_hint() -> None:
    with pytest.raises(KeyError) as exc_info:
        get_augmenter("definitely_not_a_real_augmenter")
    message = str(exc_info.value)
    assert "definitely_not_a_real_augmenter" in message
    assert "known" in message


def test_list_validators_is_sorted() -> None:
    assert list_validators() == sorted(list_validators())


def test_list_augmenters_is_sorted() -> None:
    assert list_augmenters() == sorted(list_augmenters())
