"""Tests for augmentation/llm_validators.py (parse_verdict + build_judge)."""

import pytest

from augmentation.base import AugmentedRow
from augmentation.llm_validators import build_judge, parse_verdict
from data.base import DatasetRow
from tests._augment_helpers import FakeClient

_JUDGE_MAX_OUTPUT_TOKENS = 16


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PASS", (True, "PASS")),
        ("pass", (True, "PASS")),
        ("FAIL", (False, "FAIL")),
        ("fail", (False, "FAIL")),
        ("maybe", (False, "MALFORMED")),
        ("PASS and FAIL both mentioned", (False, "MALFORMED")),
        ("", (False, "MALFORMED")),
    ],
)
def test_parse_verdict(text: str, expected: tuple[bool, str]) -> None:
    assert parse_verdict(text) == expected


def _aug(variant: str = "paraphrase", x: str = "rewritten text") -> AugmentedRow:
    return AugmentedRow(
        id="1",
        variant=variant,  # type: ignore[arg-type]
        x=x,
        y=1,
        augmenter_model="gemini/fake-model",
        prompt_hash="abc123",
    )


def _original() -> DatasetRow:
    return DatasetRow(id="1", x="original text", y=1, meta={})


@pytest.mark.parametrize("judge_name", ["label_preservation", "idiom_presence", "idiom_absence"])
def test_judge_pass_verdict(judge_name: str) -> None:
    fake = FakeClient(judge_result="PASS")
    judge = build_judge(
        judge_name, client=fake, temperature=0.0, max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS
    )

    result = judge.validate(_aug(), _original())

    assert result.passed is True
    assert result.name == judge_name
    assert result.details["verdict"] == "PASS"
    assert fake.judge_calls == 1


@pytest.mark.parametrize("judge_name", ["label_preservation", "idiom_presence", "idiom_absence"])
def test_judge_fail_verdict(judge_name: str) -> None:
    fake = FakeClient(judge_result="FAIL")
    judge = build_judge(
        judge_name, client=fake, temperature=0.0, max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS
    )

    result = judge.validate(_aug(), _original())

    assert result.passed is False
    assert result.details["verdict"] == "FAIL"
    assert fake.judge_calls == 1


@pytest.mark.parametrize("judge_name", ["label_preservation", "idiom_presence", "idiom_absence"])
def test_judge_malformed_verdict_fails_closed(judge_name: str) -> None:
    fake = FakeClient(judge_result="maybe")
    judge = build_judge(
        judge_name, client=fake, temperature=0.0, max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS
    )

    result = judge.validate(_aug(), _original())

    assert result.passed is False
    assert result.details["verdict"] == "MALFORMED"


def test_judge_details_include_model_and_prompt_hash() -> None:
    fake = FakeClient(judge_result="PASS")
    judge = build_judge(
        "label_preservation",
        client=fake,
        temperature=0.0,
        max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
    )

    result = judge.validate(_aug(), _original())

    assert result.details["judge_model"] == "gemini/fake-model"
    assert isinstance(result.details["judge_prompt_hash"], str)
    assert len(result.details["judge_prompt_hash"]) > 0


def test_judge_calls_the_client_exactly_once_per_validate() -> None:
    fake = FakeClient(judge_result="PASS")
    judge = build_judge(
        "idiom_presence", client=fake, temperature=0.0, max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS
    )

    assert fake.judge_calls == 0
    judge.validate(_aug(variant="idiomatic"), _original())
    assert fake.judge_calls == 1
    judge.validate(_aug(variant="idiomatic"), _original())
    assert fake.judge_calls == 2
    assert fake.augment_calls == 0
