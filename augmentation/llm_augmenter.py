"""Hosted-LLM augmenter base (D1/D7).

Renders a frozen variant template with the row's `{y}`/`{x}`/`{context}` and
rewrites `x` through the shared `LLMClient`. Retry/abort on failure is owned by
the pipeline (D3), not here — `augment` performs exactly one completion call and
raises `LLMError` (via the client) on empty/refused output.
"""

from augmentation.base import AugmentedRow, Variant
from augmentation.prompts.loader import render_context
from augmentation.providers.base import LLMClient
from data.base import DatasetRow


class LLMAugmenter:
    """Rewrite `x` via a hosted LLM, preserving `id`/`y`/`meta` (ARCHITECTURE §2.3)."""

    variant: Variant
    augmenter_model: str

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
        """Bind the augmenter to `variant`, its frozen `prompt_template`, and `client`."""
        self.variant = variant
        self.augmenter_model = f"{client.provider}/{client.model}"
        self._prompt_hash = prompt_hash
        self._client = client
        self._template = prompt_template
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def augment(self, ex: DatasetRow) -> AugmentedRow:
        """Render the template for `ex` and return the rewritten row (one API call)."""
        user = self._template.format(y=ex.y, x=ex.x, context=render_context(ex.meta))
        rewritten = self._client.complete(
            system="",
            user=user,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        return AugmentedRow(
            id=ex.id,
            variant=self.variant,
            x=rewritten,
            y=ex.y,
            augmenter_model=self.augmenter_model,
            prompt_hash=self._prompt_hash,
            meta=dict(ex.meta),
        )
