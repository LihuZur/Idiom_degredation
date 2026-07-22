"""Augmenter / validator registries (ARCHITECTURE §2.3)."""

from collections.abc import Callable
from typing import TypeVar

from augmentation.base import Augmenter, Validator

_A = TypeVar("_A", bound=Augmenter)
_V = TypeVar("_V", bound=Validator)

AUGMENTERS: dict[str, type[Augmenter]] = {}


def register_augmenter(name: str) -> Callable[[type[_A]], type[_A]]:
    """Decorator: register an `Augmenter` under `name`."""

    def decorator(cls: type[_A]) -> type[_A]:
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


VALIDATORS: dict[str, type[Validator]] = {}


def register_validator(name: str) -> Callable[[type[_V]], type[_V]]:
    """Decorator: register a `Validator` under `name`."""

    def decorator(cls: type[_V]) -> type[_V]:
        if name in VALIDATORS:
            raise ValueError(f"validator already registered: {name!r}")
        VALIDATORS[name] = cls
        return cls

    return decorator


def get_validator(name: str) -> type[Validator]:
    try:
        return VALIDATORS[name]
    except KeyError as e:
        raise KeyError(f"no validator registered as {name!r}; known: {sorted(VALIDATORS)}") from e


def list_validators() -> list[str]:
    return sorted(VALIDATORS)
