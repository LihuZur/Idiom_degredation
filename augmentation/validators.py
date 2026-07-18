"""No-op stub validators (ARCHITECTURE §2.3).

Each validator always passes; real checks (embedding similarity, LLM-judge
label preservation, idiom detection) are future work (README §9.7).
"""

from augmentation.base import AugmentedRow, ValidationResult
from augmentation.registry import register_validator


@register_validator("semantic_similarity")
class SemanticSimilarityValidator:
    """Stub: always passes with a perfect similarity score."""

    name: str = "semantic_similarity"

    def validate(self, ex: AugmentedRow) -> ValidationResult:
        return ValidationResult(name=self.name, passed=True, score=1.0, details={"stub": True})


@register_validator("label_preservation")
class LabelPreservationValidator:
    """Stub: always passes; no real label-preservation check yet."""

    name: str = "label_preservation"

    def validate(self, ex: AugmentedRow) -> ValidationResult:
        return ValidationResult(name=self.name, passed=True, score=None, details={"stub": True})


@register_validator("idiom_presence")
class IdiomPresenceValidator:
    """Stub: always passes; no real idiom detection yet."""

    name: str = "idiom_presence"

    def validate(self, ex: AugmentedRow) -> ValidationResult:
        return ValidationResult(
            name=self.name,
            passed=True,
            score=None,
            details={"stub": True, "note": "no real idiom detection yet"},
        )


@register_validator("idiom_absence")
class IdiomAbsenceValidator:
    """Stub: always passes; no real idiom detection yet."""

    name: str = "idiom_absence"

    def validate(self, ex: AugmentedRow) -> ValidationResult:
        return ValidationResult(name=self.name, passed=True, score=None, details={"stub": True})
