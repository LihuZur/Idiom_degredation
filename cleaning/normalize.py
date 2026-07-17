"""Text normalizers for Stage 1 cleaning (STAGE1_PLAN §3.3).

`NORMALIZERS` is a registry mapping normalizer name -> function. The
`normalize:` key in a `CleanConfig` is an ordered list of these names,
applied in order via `apply()`.
"""

import re
import unicodedata
from collections.abc import Callable

NORMALIZERS: dict[str, Callable[[str], str]] = {}

_WHITESPACE_RE = re.compile(r"\s+")


def _register(name: str) -> Callable[[Callable[[str], str]], Callable[[str], str]]:
    def deco(fn: Callable[[str], str]) -> Callable[[str], str]:
        NORMALIZERS[name] = fn
        return fn

    return deco


@_register("nfc")
def nfc(s: str) -> str:
    """Normalize unicode to NFC (composed) form."""
    return unicodedata.normalize("NFC", s)


@_register("collapse_whitespace")
def collapse_whitespace(s: str) -> str:
    """Collapse runs of whitespace to a single space and strip the ends."""
    return _WHITESPACE_RE.sub(" ", s).strip()


def apply(names: list[str], s: str) -> str:
    """Apply the named normalizers, in order, to `s`.

    Raises:
        KeyError: if a name is not registered in `NORMALIZERS`.
    """
    for name in names:
        s = NORMALIZERS[name](s)
    return s
