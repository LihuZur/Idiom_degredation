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


def test_score() -> None:
    cfg = EvalConfig(model="dummy-model")
    evaluator = Sst2Evaluator(cfg=cfg)

    # score returns True/False
    assert evaluator.score("1", "1") is True
    assert evaluator.score("1", 1) is True
    assert evaluator.score("0", "1") is False
    assert evaluator.score(None, "1") is False
