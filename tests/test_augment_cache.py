"""Tests for augmentation/cache.py (STAGE2_CONTRACT Cache section)."""

from pathlib import Path

from augmentation.cache import ResponseCache


def test_miss_returns_none(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    assert cache.get("ph", "identity", "1") is None


def test_put_then_get_is_a_hit(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    cache.put("ph", "identity", "1", {"x": "hello"})
    assert cache.get("ph", "identity", "1") == {"x": "hello"}


def test_disabled_cache_get_always_none(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir, enabled=False)
    cache.put("ph", "identity", "1", {"x": "hello"})
    assert cache.get("ph", "identity", "1") is None


def test_disabled_cache_put_creates_no_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir, enabled=False)
    cache.put("ph", "identity", "1", {"x": "hello"})
    assert not cache_dir.exists()


def test_corrupt_json_file_returns_none(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir, enabled=True)
    cache.put("ph", "identity", "1", {"x": "hello"})

    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    files[0].write_text("not valid json{{{", encoding="utf-8")

    assert cache.get("ph", "identity", "1") is None


def test_missing_file_after_enabled_check_returns_none(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir, enabled=True)
    # cache_dir does not even exist yet
    assert cache.get("ph", "identity", "nonexistent") is None


def test_key_composition_distinguishes_different_input_id(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    cache.put("ph", "identity", "1", {"x": "a"})
    cache.put("ph", "identity", "2", {"x": "b"})
    assert cache.get("ph", "identity", "1") == {"x": "a"}
    assert cache.get("ph", "identity", "2") == {"x": "b"}


def test_key_composition_distinguishes_different_prompt_hash(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    cache.put("ph1", "identity", "1", {"x": "a"})
    cache.put("ph2", "identity", "1", {"x": "b"})
    assert cache.get("ph1", "identity", "1") == {"x": "a"}
    assert cache.get("ph2", "identity", "1") == {"x": "b"}


def test_key_composition_distinguishes_different_model(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache", enabled=True)
    cache.put("ph", "identity", "1", {"x": "a"})
    cache.put("ph", "other_model", "1", {"x": "b"})
    assert cache.get("ph", "identity", "1") == {"x": "a"}
    assert cache.get("ph", "other_model", "1") == {"x": "b"}
