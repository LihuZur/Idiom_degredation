"""Shared test fixtures for Stage 2 (augmentation) tests: a scriptable fake
`LLMClient` plus a config builder pointed at the real (frozen) prompt files.

Not itself a test module (no `test_` functions); imported by the Stage 2 test
files that need to drive `AugmentPipeline` / the LLM-judge validators without
hitting a real provider.
"""

from collections.abc import Callable
from pathlib import Path

from augmentation.config import AugmentConfig, CacheCfg, PromptsCfg, RetryCfg

# One "call script" entry is either:
#   - a fixed string, returned on every call;
#   - a list of strings, consumed one-per-call-index (the last entry repeats
#     once the list is exhausted);
#   - an `Exception` instance, raised on every call (e.g. `LLMError`);
#   - a callable taking the `FakeClient` itself and returning a string, so a
#     test can script behavior off of `client.augment_calls` / `judge_calls`.
ResultSpec = str | list[str] | BaseException | Callable[["FakeClient"], str]


class FakeClient:
    """A scriptable stand-in for `augmentation.providers.base.LLMClient`.

    Discriminates augment vs. judge calls by prompt content: every judge
    template ends with "Output only ... PASS or FAIL", so a `user` prompt
    containing the substring "PASS or FAIL" is a judge call; anything else is
    an augment call.
    """

    provider: str = "gemini"
    model: str = "fake-model"

    def __init__(
        self,
        *,
        augment_result: ResultSpec = "a rewritten sentence",
        judge_result: ResultSpec = "PASS",
        model: str | None = None,
    ) -> None:
        self.augment_result = augment_result
        self.judge_result = judge_result
        self.augment_calls = 0
        self.judge_calls = 0
        self.calls: list[str] = []
        if model is not None:
            self.model = model

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        self.calls.append(user)
        if "PASS or FAIL" in user:
            self.judge_calls += 1
            return self._resolve(self.judge_result, self.judge_calls)
        self.augment_calls += 1
        return self._resolve(self.augment_result, self.augment_calls)

    def _resolve(self, spec: ResultSpec, call_index: int) -> str:
        if isinstance(spec, BaseException):
            raise spec
        if callable(spec):
            return spec(self)
        if isinstance(spec, list):
            idx = min(call_index, len(spec)) - 1
            return spec[idx]
        return spec


def make_cfg(
    *,
    cache_dir: Path,
    dataset: str = "sst2",
    max_attempts: int = 3,
    backoff_seconds: float = 0.0,
    augmenter: str = "gemini",
    augmenter_model: str = "fake-model",
) -> AugmentConfig:
    """Build a minimal `AugmentConfig` pointed at the real prompt files.

    `backoff_seconds` defaults to 0 so tests that exercise the retry loop
    never sleep.
    """
    return AugmentConfig(
        dataset=dataset,
        seed=0,
        augmenter=augmenter,
        augmenter_model=augmenter_model,
        prompts=PromptsCfg(paraphrase="paraphrase_v1.txt", idiomatic="idiomatic_v1.txt"),
        retry=RetryCfg(max_attempts=max_attempts, backoff_seconds=backoff_seconds),
        cache=CacheCfg(enabled=True, dir=str(cache_dir)),
    )
