"""Tests for augmentation/validators.py stub + the pipeline's variant->validator mapping."""

from augmentation.base import AugmentedRow
from augmentation.pipeline import (
    _VARIANT_VALIDATORS,  # pyright: ignore[reportPrivateUsage]  # test-only introspection
)
from augmentation.validators import SemanticSimilarityValidator
from data.base import DatasetRow


def test_semantic_similarity_stub_returns_expected_result() -> None:
    aug = AugmentedRow(
        id="1",
        variant="paraphrase",
        x="some text",
        y=0,
        augmenter_model="gemini/fake-model",
        prompt_hash="abc123",
    )
    original = DatasetRow(id="1", x="some original text", y=0, meta={})

    result = SemanticSimilarityValidator().validate(aug, original)

    assert result.name == "semantic_similarity"
    assert result.passed is True
    assert result.score == 1.0
    assert result.details == {"stub": True}


def test_variant_validator_mapping_idiomatic_has_idiom_presence() -> None:
    assert "idiom_presence" in _VARIANT_VALIDATORS["idiomatic"]
    assert "idiom_absence" not in _VARIANT_VALIDATORS["idiomatic"]


def test_variant_validator_mapping_paraphrase_has_idiom_absence() -> None:
    assert "idiom_absence" in _VARIANT_VALIDATORS["paraphrase"]
    assert "idiom_presence" not in _VARIANT_VALIDATORS["paraphrase"]


def test_variant_validator_mapping_both_variants_share_common_validators() -> None:
    for variant in ("paraphrase", "idiomatic"):
        assert "semantic_similarity" in _VARIANT_VALIDATORS[variant]
        assert "label_preservation" in _VARIANT_VALIDATORS[variant]


def test_variant_validator_mapping_exact_lists() -> None:
    assert _VARIANT_VALIDATORS["paraphrase"] == [
        "semantic_similarity",
        "label_preservation",
        "idiom_absence",
    ]
    assert _VARIANT_VALIDATORS["idiomatic"] == [
        "semantic_similarity",
        "label_preservation",
        "idiom_presence",
    ]
