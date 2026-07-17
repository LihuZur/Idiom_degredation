"""Registry for Stage 3 Evaluators (STAGE3_PLAN §1)."""

from collections.abc import Callable

from eval.base import BaseEvaluator

EVALUATORS: dict[str, type[BaseEvaluator]] = {}


def register_evaluator(name: str) -> Callable[[type[BaseEvaluator]], type[BaseEvaluator]]:
    """Decorator: register a `BaseEvaluator` class under `name`."""

    def decorator(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
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
