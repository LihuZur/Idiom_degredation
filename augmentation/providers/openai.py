"""OpenAI GPT client wrapper (`openai` SDK)."""

import os

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from augmentation.providers.base import EmptyResponseError, LLMError

_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIClient:
    """`LLMClient` backed by the OpenAI Chat Completions API."""

    provider: str = "openai"

    def __init__(self, model: str) -> None:
        api_key = os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise LLMError(f"missing {_API_KEY_ENV} environment variable")
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        """Generate text for `user`, prepending `system` as a system message."""
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_output_tokens,
            )
        except Exception as exc:  # SDK/transport error -> retryable
            raise LLMError(f"openai request failed: {exc}") from exc

        content = completion.choices[0].message.content
        if content is None or not content.strip():
            raise EmptyResponseError("openai returned empty output")
        return content.strip()
