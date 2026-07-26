"""Provider-agnostic LLM client seam (README §9.7, ARCHITECTURE §2.3).

`LLMClient` is the one interface the augmenter and the LLM-judge validators
talk to; `build_client` maps a provider name + pinned model id to a concrete
client. Concrete clients are pure I/O (no prompt logic, no validation) so they
are trivially mockable in tests.
"""

from typing import Protocol, runtime_checkable

_PROVIDERS = ("anthropic", "gemini", "openai")


class LLMError(RuntimeError):
    """Raised when a provider returns empty/refused output or transport fails.

    The Stage 2 pipeline catches this to drive its retry loop. A bare `LLMError`
    is treated as *transient* (transport failure, 429 rate-limit, 5xx) and
    pauses the run for resume; `EmptyResponseError` is treated as row-specific
    (see below).
    """


class EmptyResponseError(LLMError):
    """Raised when the provider returns no text for a specific input.

    Unlike a transport failure this is a property of the row (a safety filter,
    or reasoning that consumed the whole output budget), so retrying the *same*
    row repeatedly never helps. The pipeline therefore skips the row (recorded
    in the sidecar) rather than pausing the whole run on it.
    """


@runtime_checkable
class LLMClient(Protocol):
    """One completion call against a hosted LLM, pinned to a single model."""

    provider: str
    model: str

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        """Return the model's text output for `user` (empty/refused -> `LLMError`)."""
        ...


def build_client(provider: str, model: str) -> LLMClient:
    """Construct the `LLMClient` for `provider`, pinned to `model` (D1, D6, D8).

    The vendor SDK is imported lazily here so a run only loads the SDK of the
    selected provider, and the offline tests (which inject a mock client) load
    none of them.
    """
    if provider == "gemini":
        from augmentation.providers.gemini import GeminiClient  # noqa: PLC0415

        return GeminiClient(model)
    if provider == "anthropic":
        from augmentation.providers.anthropic import AnthropicClient  # noqa: PLC0415

        return AnthropicClient(model)
    if provider == "openai":
        from augmentation.providers.openai import OpenAIClient  # noqa: PLC0415

        return OpenAIClient(model)
    raise LLMError(f"unknown provider: {provider!r}; known: {list(_PROVIDERS)}")
