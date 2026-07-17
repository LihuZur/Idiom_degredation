"""Unit tests for EvalConfig schema and parsing (STAGE3_PLAN §5.1)."""

import pytest
import yaml
from pydantic import ValidationError

from eval.config import EvalConfig


def test_valid_config_parsing() -> None:
    yaml_content = """
model: qwen3.5-1.5b-instruct
seed: 42
precision: bf16
decoding:
  temperature: 0.7
  max_new_tokens: 64
  do_sample: true
batch:
  size: 16
"""
    cfg = EvalConfig.model_validate(yaml.safe_load(yaml_content))
    assert cfg.model == "qwen3.5-1.5b-instruct"
    assert cfg.seed == 42
    assert cfg.precision == "bf16"
    assert cfg.decoding.temperature == 0.7
    assert cfg.decoding.max_new_tokens == 64
    assert cfg.decoding.do_sample is True
    assert cfg.batch.size == 16


def test_default_config_parsing() -> None:
    yaml_content = """
model: mistral-7b-instruct-v0.3
"""
    cfg = EvalConfig.model_validate(yaml.safe_load(yaml_content))
    assert cfg.model == "mistral-7b-instruct-v0.3"
    assert cfg.seed == 0
    assert cfg.precision is None
    assert cfg.decoding.temperature == 0.0
    assert cfg.decoding.max_new_tokens == 32
    assert cfg.decoding.do_sample is False
    assert cfg.batch.size == 8


def test_extra_fields_forbidden() -> None:
    yaml_content = """
model: qwen3.5-1.5b-instruct
unknown_field: true
"""
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(yaml.safe_load(yaml_content))


def test_negative_temperature() -> None:
    yaml_content = """
model: qwen3.5-1.5b-instruct
decoding:
  temperature: -0.5
"""
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(yaml.safe_load(yaml_content))
