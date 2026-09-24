"""Tests for RQ2 external norm-data loaders (data/norms/*).

These run against the real downloaded files under `data/norms/_files/`
(gitignored — see paper_preparations/RQ2_idiom_properties.md §3 for
download instructions). Each test module-skips if its file isn't present,
so the suite stays green on a machine that hasn't sourced the norm data.
"""

from __future__ import annotations

import pytest

from data.norms.frequency import (
    _DEFAULT_PATH as FREQUENCY_PATH,  # pyright: ignore[reportPrivateUsage]
)
from data.norms.frequency import idiom_frequency, load_word_frequencies
from data.norms.magpie import _DEFAULT_PATH as MAGPIE_PATH  # pyright: ignore[reportPrivateUsage]
from data.norms.magpie import load_magpie_stats
from data.norms.opacity import (
    _DEFAULT_BULKES_TANNER_PATH as BULKES_TANNER_PATH,  # pyright: ignore[reportPrivateUsage]
)
from data.norms.opacity import (
    _DEFAULT_LIBBEN_TITONE_PATH as LIBBEN_TITONE_PATH,  # pyright: ignore[reportPrivateUsage]
)
from data.norms.opacity import load_opacity_norms

pytestmark_magpie = pytest.mark.skipif(
    not MAGPIE_PATH.exists(), reason="MAGPIE_unfiltered.jsonl not sourced locally"
)
pytestmark_opacity = pytest.mark.skipif(
    not BULKES_TANNER_PATH.exists() or not LIBBEN_TITONE_PATH.exists(),
    reason="opacity norm files not sourced locally",
)
pytestmark_frequency = pytest.mark.skipif(
    not FREQUENCY_PATH.exists(), reason="enwiki frequency file not sourced locally"
)


@pytestmark_magpie
def test_load_magpie_stats_has_known_idiom() -> None:
    stats = load_magpie_stats()
    assert "come to terms with" in stats
    entry = stats["come to terms with"]
    assert entry.instance_count >= 1
    assert 0.0 <= entry.idiomatic_fraction <= 1.0
    assert 0.0 <= entry.mean_confidence <= 1.0


@pytestmark_magpie
def test_load_magpie_stats_lowercases_keys() -> None:
    stats = load_magpie_stats()
    assert all(idiom == idiom.lower() for idiom in stats)


@pytestmark_opacity
def test_load_opacity_norms_covers_bulkes_tanner_example() -> None:
    norms = load_opacity_norms()
    entry = norms["bare one's soul"]
    assert entry.source == "bulkes_tanner_2017"
    assert 0.0 <= entry.opacity <= 1.0


@pytestmark_opacity
def test_load_opacity_norms_higher_opacity_for_less_decomposable() -> None:
    # "beat his breast" scored low global-decomposability (~0.19) in the raw
    # xlsx sample; "bent the law" scored high (~0.79). Opacity is the inverse.
    norms = load_opacity_norms()
    assert norms["he beat his breast"].opacity > norms["they bent the law"].opacity


@pytestmark_frequency
def test_load_word_frequencies_common_word_has_high_count() -> None:
    freqs = load_word_frequencies()
    assert freqs["the"] > freqs.get("aardvark", 0)


@pytestmark_frequency
def test_idiom_frequency_uses_rarest_constituent_word() -> None:
    freqs = load_word_frequencies()
    freq = idiom_frequency("kick the bucket", freqs)
    assert freq is not None
    assert freq == min(freqs["kick"], freqs["the"], freqs["bucket"])


@pytestmark_frequency
def test_idiom_frequency_returns_none_for_unknown_word() -> None:
    freqs = load_word_frequencies()
    assert idiom_frequency("zzznotarealword the bucket", freqs) is None


@pytestmark_frequency
def test_idiom_frequency_splits_hyphenated_compounds() -> None:
    freqs = load_word_frequencies()
    freq = idiom_frequency("run-of-the-mill", freqs)
    assert freq is not None
    assert freq == min(freqs["run"], freqs["of"], freqs["the"], freqs["mill"])
