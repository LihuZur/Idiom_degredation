"""Deterministic hashing helpers (STAGE1_PLAN §1 Reproducibility)."""

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize `obj` to a canonical JSON string (sorted keys, no spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(resolved_config: dict[str, Any]) -> str:
    """Return the first 16 hex chars of the sha256 of the canonical config JSON."""
    digest = hashlib.sha256(canonical_json(resolved_config).encode("utf-8")).hexdigest()
    return digest[:16]


def prompt_hash(text: str) -> str:
    """First 16 hex chars of sha256 of the raw template text (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
