"""Text normalizers for Stage 1 cleaning (STAGE1_PLAN §3.3).

`NORMALIZERS` is a registry mapping normalizer name -> function. The
`normalize:` key in a `CleanConfig` is an ordered list of these names,
applied in order via `apply()`.
"""

import html
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


_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_ELIDED_APOSTROPHE_RE = re.compile(r"\s?#(?:39|146);(?=(?:s|t|re|ll|ve|d|m)\b)")
_NUMERIC_ENTITY_RE = re.compile(r"#(\d{2,4});")
_CP1252_FIRST, _CP1252_LAST = 128, 159


def _numeric_entity(m: re.Match[str]) -> str:
    code = int(m.group(1))
    if _CP1252_FIRST <= code <= _CP1252_LAST:
        return bytes([code]).decode("cp1252", errors="replace")
    return chr(code)


@_register("html_artifacts")
def html_artifacts(s: str) -> str:
    """Undo the HTML damage in AG News text: "#39;s" -> "'s", "&lt;b&gt;" tags dropped, "\\" -> " ".

    AG News stores apostrophes, quotes, dashes and dollar signs as "#39;", "#146;", "#151;",
    "#36;" (the leading "&" was lost), some descriptions carry escaped HTML tags, and line breaks
    appear as backslashes. A space before a contraction apostrophe is removed ("country #39;s").
    """
    s = html.unescape(s)
    s = _HTML_TAG_RE.sub("", s)
    s = _ELIDED_APOSTROPHE_RE.sub("'", s)
    s = _NUMERIC_ENTITY_RE.sub(_numeric_entity, s)
    return s.replace("\\", " ")


def apply(names: list[str], s: str) -> str:
    """Apply the named normalizers, in order, to `s`.

    Raises:
        KeyError: if a name is not registered in `NORMALIZERS`.
    """
    for name in names:
        s = NORMALIZERS[name](s)
    return s
