"""Registry for Stage 3 Evaluators (STAGE3_PLAN §1)."""

from collections.abc import Callable
from typing import TypeVar

from eval.base import BaseEvaluator

EVALUATORS: dict[str, type[BaseEvaluator]] = {}

_E = TypeVar("_E", bound=BaseEvaluator)


def register_evaluator(name: str) -> Callable[[type[_E]], type[_E]]:
    """Decorator: register a `BaseEvaluator` class under `name`.

    Generic over the decorated subclass (not just `BaseEvaluator`) so that a
    further subclass of an already-registered evaluator (e.g.
    `MnliHypothesisEvaluator(MnliEvaluator)`) keeps its own members visible to
    static analysis instead of being widened to `type[BaseEvaluator]`.
    """

    def decorator(cls: type[_E]) -> type[_E]:
        if name in EVALUATORS:
            raise ValueError(f"evaluator already registered: {name!r}")
        EVALUATORS[name] = cls
        return cls

    return decorator


def get_evaluator(name: str) -> type[BaseEvaluator]:
    """Retrieve the registered evaluator class for a dataset."""
    try:
        return EVALUATORS[name]
    except KeyError as e:
        raise KeyError(
            f"no evaluator registered for dataset {name!r}; known: {sorted(EVALUATORS)}"
        ) from e


def list_evaluators() -> list[str]:
    """List all registered evaluator names."""
    return sorted(EVALUATORS)
