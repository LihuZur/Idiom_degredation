"""Google Gemini client wrapper (google-genai SDK); default provider (D5)."""

import os
import threading
from collections import Counter

from google import genai
from google.genai import types

from augmentation.providers.base import EmptyResponseError, LLMError

_API_KEY_ENV = "GEMINI_API_KEY"

# The SDK defaults to no request timeout, so a dropped response leaves the socket
# blocked forever and the run stalls silently. Observed rows take 5-10s; 120s is a
# wide margin that still surfaces a hang as a retryable LLMError.
_REQUEST_TIMEOUT_MS = 120_000


class GeminiClient:
    """`LLMClient` backed by the Gemini Developer API (`google-genai`)."""

    provider: str = "gemini"

    def __init__(self, model: str, thinking_level: str = "low") -> None:
        api_key = os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise LLMError(f"missing {_API_KEY_ENV} environment variable")
        self.model = model
        self.thinking_level = thinking_level
        # Billed tokens summed over this client's calls (cached replies never reach the client).
        self.usage: Counter[str] = Counter()
        self._usage_lock = threading.Lock()
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        """Generate text for `user`, applying `system` as a system instruction."""
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    # This model family always thinks (thinking_budget=0 is rejected);
                    # LOW (default) keeps the reasoning trace short so max_output_tokens
                    # isn't entirely consumed by invisible thought tokens before any
                    # visible answer text is emitted. Thought tokens are billed as output.
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel[self.thinking_level.upper()]
                    ),
                ),
            )
        except Exception as exc:  # SDK/transport error -> retryable
            raise LLMError(f"gemini request failed: {exc}") from exc
        self._record_usage(response.usage_metadata)
        candidates = response.candidates or []
        if candidates and candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
            # Thought tokens count against max_output_tokens; a reply cut off there is unusable and
            # must not be cached (2026-10-21: MEDIUM thinking at 1,536 tokens cut 61 of 100 replies).
            raise EmptyResponseError("gemini reply cut off at max_output_tokens")

        text = response.text
        if text is None or not text.strip():
            raise EmptyResponseError("gemini returned empty output")
        return text.strip()

    def _record_usage(self, meta: types.GenerateContentResponseUsageMetadata | None) -> None:
        with self._usage_lock:
            self.usage["calls"] += 1
            if meta is None:
                return
            self.usage["input_tokens"] += meta.prompt_token_count or 0
            self.usage["output_tokens"] += meta.candidates_token_count or 0
            self.usage["thinking_tokens"] += meta.thoughts_token_count or 0
