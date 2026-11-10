"""Resume and cache behaviour of the generation path: reruns must not repeat paid requests."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from augmentation.cache import ResponseCache
from augmentation.idiom_generation import cached_completer, request_key, sync_completer
from augmentation.providers import gemini_batch
from augmentation.providers.base import EmptyResponseError
from augmentation.providers.env import load_env


class FakeClient:
    provider = "fake"
    model = "m"

    def __init__(self, empty: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self._empty = empty or set()

    def complete(
        self, *, system: str, user: str, temperature: float, max_output_tokens: int
    ) -> str:
        self.calls.append(user)
        if user in self._empty:
            raise EmptyResponseError("blocked")
        return f"reply:{user}"


def _sync(client: FakeClient, cache: ResponseCache, temperature: float = 0.5) -> Any:
    return sync_completer(
        client, cache, "ph", temperature=temperature, max_output_tokens=100, workers=2
    )


def test_sync_rerun_serves_finished_requests_from_disk(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    first = FakeClient()
    assert _sync(first, cache)(["a", "b"]) == ["reply:a", "reply:b"]
    second = FakeClient()
    assert _sync(second, cache)(["b", "c", "a"]) == ["reply:b", "reply:c", "reply:a"]
    assert second.calls == ["c"]


def test_sync_cache_is_content_addressed_not_positional(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    _sync(FakeClient(), cache)(["a", "b", "c"])
    again = FakeClient()
    _sync(again, cache)(["c"])
    assert again.calls == []


def test_changed_temperature_does_not_reuse_stale_replies(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    _sync(FakeClient(), cache, temperature=0.5)(["a"])
    other = FakeClient()
    _sync(other, cache, temperature=0.0)(["a"])
    assert other.calls == ["a"]
    assert request_key("a", 0.5, 100) != request_key("a", 0.0, 100)


def test_empty_response_is_not_cached(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    assert _sync(FakeClient(empty={"a"}), cache)(["a"]) == [None]
    retry = FakeClient()
    assert _sync(retry, cache)(["a"]) == ["reply:a"]


def test_batch_wrapper_sends_only_misses_and_caches_results(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    sent: list[list[str]] = []

    def run(prompts: Any) -> list[str | None]:
        sent.append(list(prompts))
        return [None if p == "bad" else f"b:{p}" for p in prompts]

    complete = cached_completer(run, cache, "ph", "g/m", temperature=0.5, max_output_tokens=100)
    assert complete(["a", "bad"]) == ["b:a", None]
    assert complete(["a", "c", "bad"]) == ["b:a", "b:c", None]
    assert sent == [["a", "bad"], ["c", "bad"]]


def test_batch_job_is_reattached_after_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    fetched: list[str] = []
    job = SimpleNamespace(
        name="jobs/1",
        state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
        dest=SimpleNamespace(
            inlined_responses=[SimpleNamespace(response=SimpleNamespace(text=" hi "))]
        ),
    )

    class Batches:
        def create(self, **_: Any) -> Any:
            created.append("x")
            return job

        def get(self, *, name: str) -> Any:
            fetched.append(name)
            return job

    class FakeGenai:
        def __init__(self, **_: Any) -> None:
            self.batches = Batches()

    monkeypatch.setattr(gemini_batch.genai, "Client", FakeGenai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def call() -> list[str | None]:
        return gemini_batch.run_batch(
            ["p"], model="m", temperature=0.5, max_output_tokens=10, state_dir=tmp_path
        )

    assert call() == ["hi"]
    assert call() == ["hi"]
    assert (len(created), fetched) == (1, ["jobs/1"])


def test_load_env_fills_missing_variables_and_keeps_the_shell_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# c\nNEW_KEY_X='abc'\nexport OTHER_KEY_X=def\nSET_KEY_X=file\n")
    for name in ("NEW_KEY_X", "OTHER_KEY_X"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SET_KEY_X", "shell")
    load_env([env_file])
    assert (os.environ["NEW_KEY_X"], os.environ["OTHER_KEY_X"]) == ("abc", "def")
    assert os.environ["SET_KEY_X"] == "shell"
