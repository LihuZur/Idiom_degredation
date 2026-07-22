"""Anthropic Claude client wrapper (`anthropic` SDK)."""

import os

from anthropic import Anthropic, omit
from anthropic.types import MessageParam

from augmentation.providers.base import LLMError

_API_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicClient:
    """`LLMClient` backed by the Anthropic Messages API."""

    provider: str = "anthropic"

    def __init__(self, model: str) -> None:
        api_key = os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise LLMError(f"missing {_API_KEY_ENV} environment variable")
        self.model = model
        self._client = Anthropic(api_key=api_key)

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        """Generate text for `user`, passing `system` as the top-level system prompt."""
        messages: list[MessageParam] = [{"role": "user", "content": user}]
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                temperature=temperature,
                system=system if system else omit,
                messages=messages,
            )
        except Exception as exc:  # SDK/transport error -> retryable
            raise LLMError(f"anthropic request failed: {exc}") from exc

        text = "".join(block.text for block in message.content if block.type == "text").strip()
        if not text:
            raise LLMError("anthropic returned empty output")
        return text
