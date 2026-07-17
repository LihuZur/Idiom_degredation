"""Tests for cleaning/config.py (STAGE1_PLAN §5.2)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from cleaning.config import CleanConfig, load_config

VALID_YAML = """
dataset: sst2
seed: 0
hf_revision: abc123
length:
  min_tokens: 5
  max_tokens: 60
max_rows: 2000
normalize:
  - nfc
  - collapse_whitespace
dedupe: true
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_valid_yaml_parses(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, VALID_YAML))
    assert cfg.dataset == "sst2"
    assert cfg.hf_revision == "abc123"
    assert cfg.length.min_tokens == 5
    assert cfg.max_rows == 2000
    assert cfg.normalize == ["nfc", "collapse_whitespace"]


def test_missing_hf_revision_raises(tmp_path: Path) -> None:
    text = VALID_YAML.replace("hf_revision: abc123\n", "")
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, text))


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    text = VALID_YAML + "\nunknown_key: 1\n"
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, text))


def test_max_rows_null_is_legal(tmp_path: Path) -> None:
    text = VALID_YAML.replace("max_rows: 2000", "max_rows: null")
    cfg = load_config(_write(tmp_path, text))
    assert cfg.max_rows is None


def test_unknown_normalizer_name_raises(tmp_path: Path) -> None:
    text = VALID_YAML + "\n"
    text = text.replace(
        "  - collapse_whitespace\n", "  - collapse_whitespace\n  - not_a_real_one\n"
    )
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, text))


def test_empty_normalize_list_is_legal() -> None:
    cfg = CleanConfig.model_validate(
        {
            "dataset": "sst2",
            "seed": 0,
            "hf_revision": "abc123",
            "length": {"min_tokens": 1, "max_tokens": 10},
            "max_rows": None,
            "normalize": [],
            "dedupe": True,
        }
    )
    assert cfg.normalize == []
