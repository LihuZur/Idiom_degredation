"""Tests for cleaning/hashing.py (STAGE1_PLAN §5.4)."""

from cleaning.hashing import canonical_json, config_hash


def test_canonical_json_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_is_compact() -> None:
    assert canonical_json({"a": 1}) == '{"a":1}'


def test_config_hash_stable_for_equal_configs() -> None:
    cfg1 = {"dataset": "sst2", "seed": 0, "length": {"min_tokens": 5, "max_tokens": 60}}
    cfg2 = {"seed": 0, "length": {"max_tokens": 60, "min_tokens": 5}, "dataset": "sst2"}
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_differs_for_different_configs() -> None:
    cfg1 = {"dataset": "sst2", "seed": 0}
    cfg2 = {"dataset": "sst2", "seed": 1}
    assert config_hash(cfg1) != config_hash(cfg2)


def test_config_hash_is_16_hex_chars() -> None:
    h = config_hash({"a": 1})
    assert len(h) == 16
    int(h, 16)  # raises ValueError if not hex
