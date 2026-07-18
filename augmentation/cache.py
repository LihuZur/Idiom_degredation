"""Per-key JSON file response cache for Stage 2 augmentation (STAGE2 contract)."""

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """Disk-backed cache mapping (prompt_hash, augmenter_model, input_id) -> JSON value."""

    def __init__(self, cache_dir: Path, *, enabled: bool = True) -> None:
        self._cache_dir = cache_dir
        self._enabled = enabled

    def _path_for(self, prompt_hash: str, augmenter_model: str, input_id: str) -> Path:
        key = f"{prompt_hash}\x1e{augmenter_model}\x1e{input_id}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"

    def get(self, prompt_hash: str, augmenter_model: str, input_id: str) -> dict[str, Any] | None:
        """Return the cached value, or None if absent/unreadable/not a JSON object."""
        if not self._enabled:
            return None
        path = self._path_for(prompt_hash, augmenter_model, input_id)
        try:
            parsed: Any = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def put(
        self, prompt_hash: str, augmenter_model: str, input_id: str, value: dict[str, Any]
    ) -> None:
        """Write `value` to the cache; no-op when caching is disabled."""
        if not self._enabled:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(prompt_hash, augmenter_model, input_id)
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
