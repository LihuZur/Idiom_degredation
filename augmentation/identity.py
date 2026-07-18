"""Identity augmenter (ARCHITECTURE §2.3).

Intentional stub standing in for the future hosted LLM augmenter (README
§9.7): copies `x` verbatim so the Stage 2 pipeline is exercisable end-to-end
before a real (non-HF) augmenter model is wired in.
"""

from augmentation.base import AugmentedRow, Variant
from augmentation.registry import register_augmenter
from data.base import DatasetRow


@register_augmenter("identity")
class IdentityAugmenter:
    """No-op augmenter: copies `x` verbatim, preserving `id`/`y`/`meta`."""

    augmenter_model: str = "identity"
    variant: Variant

    def __init__(self, *, variant: Variant, prompt_hash: str) -> None:
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
