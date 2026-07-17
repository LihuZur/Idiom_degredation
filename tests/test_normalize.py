"""Tests for cleaning/normalize.py (STAGE1_PLAN §5.1)."""

import pytest

from cleaning.normalize import apply, collapse_whitespace, nfc


def test_nfc_idempotent() -> None:
    s = "café"
    assert nfc(s) == nfc(nfc(s))


def test_nfc_folds_decomposed_to_composed() -> None:
    decomposed = "cafe\u0301"  # "e" + combining acute accent
    composed = "café"
    assert nfc(decomposed) == composed


def test_collapse_whitespace_collapses_runs_and_strips() -> None:
    assert collapse_whitespace("  a   b\t\tc\n\nd  ") == "a b c d"


def test_collapse_whitespace_empty_string() -> None:
    assert collapse_whitespace("") == ""
    assert collapse_whitespace("   ") == ""


def test_apply_empty_list_is_identity() -> None:
    assert apply([], "  a  b  ") == "  a  b  "


def test_apply_composes_in_order() -> None:
    decomposed = "cafe\u0301   noir"
    result = apply(["nfc", "collapse_whitespace"], decomposed)
    assert result == "café noir"


def test_apply_unknown_normalizer_raises_key_error() -> None:
    with pytest.raises(KeyError):
        apply(["not_a_real_normalizer"], "x")
