"""Unit tests for SST-2 and MMLU parsed outputs (STAGE3_PLAN §5.3)."""

from typing import Any

from augmentation.base import AugmentedRow
from eval.config import EvalConfig
from eval.mmlu import MmluEvaluator
from eval.sst2 import Sst2Evaluator


def make_dummy_ex(x: str = "", y: Any = "", meta: dict[str, Any] | None = None) -> AugmentedRow:
    return AugmentedRow(
        id="dummy",
        variant="original",
        x=x,
        y=y,
        augmenter_model="",
        prompt_hash="",
        meta=meta or {},
    )


def test_sst2_parser() -> None:
    cfg = EvalConfig(model="dummy-model")
    evaluator = Sst2Evaluator(cfg=cfg)
    ex = make_dummy_ex()

    # positive cases
    assert evaluator.parse("positive", ex) == ("1", "ok")
    assert evaluator.parse(" POSITIVE.\n", ex) == ("1", "ok")
    assert evaluator.parse("The answer is positive.", ex) == ("1", "ok")

    # negative cases
    assert evaluator.parse("negative", ex) == ("0", "ok")
    assert evaluator.parse(" NEGATIVE.\n", ex) == ("0", "ok")
    assert evaluator.parse("The answer is negative.", ex) == ("0", "ok")

    # unparseable cases
    assert evaluator.parse("neg", ex) == (None, "unparseable")
    assert evaluator.parse("", ex) == (None, "unparseable")
    assert evaluator.parse("unknown output", ex) == (None, "unparseable")

    # multiple occurrences (picks first)
    assert evaluator.parse("positive and negative", ex) == ("1", "ok")
    assert evaluator.parse("negative and positive", ex) == ("0", "ok")

    # reasoning-model <think> trace: mentions both words while reasoning, only the
    # text after </think> should be used to determine the answer
    assert evaluator.parse(
        "<think>positive... no wait, negative</think>\nThe answer is positive.", ex
    ) == ("1", "ok")
    assert evaluator.parse("<think>positive seems likely</think>\nnegative", ex) == ("0", "ok")


def test_mmlu_parser() -> None:
    cfg = EvalConfig(model="dummy-model")
    evaluator = MmluEvaluator(cfg=cfg)
    ex = make_dummy_ex()

    # parse returns "0", "1", "2", "3" for A, B, C, D
    assert evaluator.parse("A", ex) == ("0", "ok")
    assert evaluator.parse("b", ex) == ("1", "ok")
    assert evaluator.parse("C.\n", ex) == ("2", "ok")
    assert evaluator.parse("The correct choice is D.", ex) == ("3", "ok")

    # unparseable cases
    assert evaluator.parse("E", ex) == (None, "unparseable")
    assert evaluator.parse("5", ex) == (None, "unparseable")
    assert evaluator.parse("unknown output", ex) == (None, "unparseable")

    # reasoning-model <think> trace: mentions every option while reasoning, only the
    # text after </think> should be used to determine the answer
    assert evaluator.parse(
        "<think>Option A looks wrong, Option B looks wrong too</think>\nThe correct answer is C.",
        ex,
    ) == ("2", "ok")
    # a truncated <think> block with no closing tag falls back to the old
    # first-occurrence-in-full-text behavior for non-reasoning models (unreliable,
    # but this model isn't registered as a reasoning model so nothing changes)
    assert evaluator.parse("<think>Option A is discussed here", ex) == ("0", "ok")


def test_mmlu_parser_reasoning_model_truncation() -> None:
    cfg = EvalConfig(model="deepseek-r1-distill-qwen-7b")
    evaluator = MmluEvaluator(cfg=cfg)
    ex = make_dummy_ex()

    # closed </think> block: parse the final answer as usual
    assert evaluator.parse(
        "<think>Option A looks wrong, Option B looks wrong too</think>\nThe correct answer is C.",
        ex,
    ) == ("2", "ok")

    # no closing </think>: the model never reached its final answer, must be unparseable
    # even though "A" appears in the (incomplete) reasoning trace
    assert evaluator.parse("<think>Option A is discussed here", ex) == (None, "unparseable")
    assert evaluator.parse("", ex) == (None, "unparseable")


def test_sst2_parser_reasoning_model_truncation() -> None:
    cfg = EvalConfig(model="deepseek-r1-distill-qwen-7b")
    evaluator = Sst2Evaluator(cfg=cfg)
    ex = make_dummy_ex()

    # closed </think> block: parse the final answer as usual
    assert evaluator.parse(
        "<think>positive... no wait, negative</think>\nThe answer is positive.", ex
    ) == ("1", "ok")

    # no closing </think>: mentions "positive" while reasoning but never finishes
    assert evaluator.parse("<think>positive seems likely", ex) == (None, "unparseable")
    assert evaluator.parse("", ex) == (None, "unparseable")


def test_score() -> None:
    cfg = EvalConfig(model="dummy-model")
    evaluator = Sst2Evaluator(cfg=cfg)

    # score returns True/False
    assert evaluator.score("1", "1") is True
    assert evaluator.score("1", 1) is True
    assert evaluator.score("0", "1") is False
    assert evaluator.score(None, "1") is False
