"""Semantic-similarity validator (deferred stub, D2 / STAGE2 §7).

The label-preservation and idiom presence/absence validators are real LLM
judges and live in `augmentation/llm_validators.py`. Semantic similarity stays
an always-pass stub this phase; a real embedding-cosine gate is future work
(needs an embedding-model decision + `min_cosine` threshold).
"""

from augmentation.base import AugmentedRow, ValidationResult
from augmentation.registry import register_validator
from data.base import DatasetRow


@register_validator("semantic_similarity")
class SemanticSimilarityValidator:
    """Stub: always passes with a perfect similarity score (embeddings deferred)."""

    name: str = "semantic_similarity"

    def validate(self, ex: AugmentedRow, original: DatasetRow) -> ValidationResult:
        """Return an always-pass result; the `original` row is ignored for now."""
        return ValidationResult(name=self.name, passed=True, score=1.0, details={"stub": True})
