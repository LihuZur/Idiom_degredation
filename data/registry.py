"""Dataset registry (ARCHITECTURE §2.1)."""

from collections.abc import Callable

from data.base import DatasetLoader

DATASETS: dict[str, type[DatasetLoader]] = {}


def register_dataset(
    name: str,
) -> Callable[[type[DatasetLoader]], type[DatasetLoader]]:
    """Decorator: register a `DatasetLoader` under `name`.

    Raises:
        ValueError: if `name` is already registered.
    """

    def decorator(cls: type[DatasetLoader]) -> type[DatasetLoader]:
        if name in DATASETS:
            raise ValueError(f"dataset already registered: {name!r}")
        DATASETS[name] = cls
        return cls

    return decorator


def get_dataset(name: str) -> type[DatasetLoader]:
    """Look up a registered `DatasetLoader` class by name."""
    try:
        return DATASETS[name]
    except KeyError as e:
        raise KeyError(f"no dataset registered as {name!r}; known: {sorted(DATASETS)}") from e


def list_datasets() -> list[str]:
    """Return the sorted list of registered dataset names."""
    return sorted(DATASETS)
