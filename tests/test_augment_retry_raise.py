"""Tests for the pipeline's retry-then-abort loop (`_augment_with_retry`, D3).

Exercises `AugmentPipeline._augment_with_retry` directly (rather than the full
two-variant `run()`) so the augment/judge call counts are precise and
attributable to a single variant/row, mirroring how other Stage 2 tests reach
into pipeline internals (e.g. `_VARIANT_VALIDATORS`).
"""

from pathlib import Path

import pytest

from augmentation.pipeline import AugmentationError, AugmentPipeline
from augmentation.prompts.loader import load_prompt
from augmentation.providers.base import LLMError
from augmentation.registry import get_augmenter
from data.base import DatasetRow
from tests._augment_helpers import FakeClient, make_cfg

_MAX_ATTEMPTS = 3


def _make_pipeline(tmp_path: Path, *, max_attempts: int = _MAX_ATTEMPTS) -> AugmentPipeline:
    cfg = make_cfg(cache_dir=tmp_path / "cache", max_attempts=max_attempts, backoff_seconds=0.0)
    return AugmentPipeline(
        cfg, tmp_path / "original.csv", tmp_path / "out", tmp_path / "config.yaml"
    )


def _make_augmenter(pipeline: AugmentPipeline, client: FakeClient):
    cfg = pipeline._cfg  # pyright: ignore[reportPrivateUsage]  # test-only introspection
    template = load_prompt(cfg.prompts.paraphrase)
    return get_augmenter(cfg.augmenter)(
        variant="paraphrase",
        prompt_hash="ph",
        client=client,
        prompt_template=template,
        temperature=cfg.decoding.temperature,
        max_output_tokens=cfg.decoding.max_output_tokens,
    )


def test_judge_always_fails_raises_after_max_attempts(tmp_path: Path) -> None:
    fake = FakeClient(augment_result="rewritten", judge_result="FAIL")
    pipeline = _make_pipeline(tmp_path)
    augmenter = _make_augmenter(pipeline, fake)
    validators = pipeline._build_validators("paraphrase", fake)  # pyright: ignore[reportPrivateUsage]
    row = DatasetRow(id="r1", x="some text", y=1, meta={})

    with pytest.raises(AugmentationError) as exc_info:
        pipeline._augment_with_retry(  # pyright: ignore[reportPrivateUsage]
            augmenter, validators, row, "paraphrase"
        )

    err = exc_info.value
    assert err.attempts == _MAX_ATTEMPTS
    assert err.row_id == "r1"
    assert err.variant == "paraphrase"
    assert fake.augment_calls == _MAX_ATTEMPTS


def test_augment_llm_error_every_attempt_raises_after_max_attempts(tmp_path: Path) -> None:
    fake = FakeClient(augment_result=LLMError("boom"), judge_result="PASS")
    pipeline = _make_pipeline(tmp_path)
    augmenter = _make_augmenter(pipeline, fake)
    validators = pipeline._build_validators("paraphrase", fake)  # pyright: ignore[reportPrivateUsage]
    row = DatasetRow(id="r1", x="some text", y=1, meta={})

    with pytest.raises(AugmentationError) as exc_info:
        pipeline._augment_with_retry(  # pyright: ignore[reportPrivateUsage]
            augmenter, validators, row, "paraphrase"
        )

    err = exc_info.value
    assert err.attempts == _MAX_ATTEMPTS
    assert err.row_id == "r1"
    assert err.variant == "paraphrase"
    assert fake.augment_calls == _MAX_ATTEMPTS


def test_fail_then_pass_succeeds_on_second_attempt(tmp_path: Path) -> None:
    def _judge(client: FakeClient) -> str:
        return "FAIL" if client.augment_calls == 1 else "PASS"

    fake = FakeClient(augment_result="rewritten", judge_result=_judge)
    pipeline = _make_pipeline(tmp_path)
    augmenter = _make_augmenter(pipeline, fake)
    validators = pipeline._build_validators("paraphrase", fake)  # pyright: ignore[reportPrivateUsage]
    row = DatasetRow(id="r1", x="some text", y=1, meta={})

    aug, results = pipeline._augment_with_retry(  # pyright: ignore[reportPrivateUsage]
        augmenter, validators, row, "paraphrase"
    )

    assert aug.x == "rewritten"
    assert all(r.passed for r in results)
    assert fake.augment_calls == 2
