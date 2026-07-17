"""Augmenter registry (ARCHITECTURE §2.3)."""

from collections.abc import Callable

from augmentation.base import Augmenter

AUGMENTERS: dict[str, type[Augmenter]] = {}


def register_augmenter(
    name: str,
) -> Callable[[type[Augmenter]], type[Augmenter]]:
    """Decorator: register an `Augmenter` under `name`."""

    def decorator(cls: type[Augmenter]) -> type[Augmenter]:
        if name in AUGMENTERS:
            raise ValueError(f"augmenter already registered: {name!r}")
        AUGMENTERS[name] = cls
        return cls

    return decorator


def get_augmenter(name: str) -> type[Augmenter]:
    try:
        return AUGMENTERS[name]
    except KeyError as e:
        raise KeyError(f"no augmenter registered as {name!r}; known: {sorted(AUGMENTERS)}") from e


def list_augmenters() -> list[str]:
    return sorted(AUGMENTERS)
