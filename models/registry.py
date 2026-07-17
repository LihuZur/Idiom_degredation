"""Model registry (ARCHITECTURE §2.5)."""

from collections.abc import Callable

from models.base import Model, ModelKind, ModelSpec, Precision

MODELS: dict[str, ModelSpec] = {}
_MODEL_CLASSES: dict[str, type[Model]] = {}


def register_model(
    name: str,
    *,
    hf_repo: str,
    revision: str | None = None,
    kind: ModelKind = "decoder",
    default_precision: Precision = "bf16",
) -> Callable[[type[Model]], type[Model]]:
    """Decorator: register a `Model` runner class under `name`.

    Args:
        name: Registry key used by configs and CLI.
        hf_repo: Hugging Face repo id (e.g. `"Qwen/Qwen3.5-7B-Instruct"`).
        revision: Pinned HF revision (commit sha / tag). Required for
            reproducibility once a model is used in a published run.
        kind: Only `"decoder"` is supported in the current phase (README §4).
        default_precision: Weight/inference precision on GPU. Runners must
            fall back to `fp32` when the selected device is `cpu`
            (README §9.4).
    """

    def decorator(cls: type[Model]) -> type[Model]:
        if name in MODELS:
            raise ValueError(f"model already registered: {name!r}")
        MODELS[name] = ModelSpec(
            name=name,
            hf_repo=hf_repo,
            revision=revision,
            kind=kind,
            default_precision=default_precision,
        )
        _MODEL_CLASSES[name] = cls
        return cls

    return decorator


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError as e:
        raise KeyError(f"no model registered as {name!r}; known: {sorted(MODELS)}") from e


def get_model_class(name: str) -> type[Model]:
    try:
        return _MODEL_CLASSES[name]
    except KeyError as e:
        raise KeyError(f"no model registered as {name!r}; known: {sorted(_MODEL_CLASSES)}") from e


def list_models() -> list[str]:
    return sorted(MODELS)
