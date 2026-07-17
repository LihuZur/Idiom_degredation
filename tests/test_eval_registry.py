"""Unit tests for Eval registry (STAGE3_PLAN §5.4)."""

import pytest

from eval.base import BaseEvaluator
from eval.registry import EVALUATORS, get_evaluator, list_evaluators, register_evaluator


def test_evaluator_registration() -> None:
    # Check that registering works
    @register_evaluator("dummy_dataset")
    class DummyEvaluator(BaseEvaluator):
        dataset = "dummy_dataset"

        def format(self, ex):  # type: ignore
            pass

        def parse(self, raw, ex):  # type: ignore
            pass

    assert "dummy_dataset" in EVALUATORS
    assert get_evaluator("dummy_dataset") is DummyEvaluator
    assert "dummy_dataset" in list_evaluators()

    _ = DummyEvaluator
    # Clean up dummy evaluator to prevent leaking to other tests
    del EVALUATORS["dummy_dataset"]


def test_evaluator_duplicate_registration_raises() -> None:
    @register_evaluator("duplicate_dataset")
    class Dummy1(BaseEvaluator):
        dataset = "duplicate_dataset"

        def format(self, ex):  # type: ignore
            pass

        def parse(self, raw, ex):  # type: ignore
            pass

    with pytest.raises(ValueError, match="evaluator already registered"):

        @register_evaluator("duplicate_dataset")
        class Dummy2(BaseEvaluator):
            dataset = "duplicate_dataset"

            def format(self, ex):  # type: ignore
                pass

            def parse(self, raw, ex):  # type: ignore
                pass

        _ = Dummy2

    _ = Dummy1
    # Clean up
    del EVALUATORS["duplicate_dataset"]


def test_get_nonexistent_evaluator_raises() -> None:
    with pytest.raises(KeyError, match="no evaluator registered for dataset"):
        get_evaluator("nonexistent_dataset")
