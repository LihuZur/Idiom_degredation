"""Make API keys available from a gitignored `.env` without exporting them in the shell."""

import os
from collections.abc import Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(paths: Sequence[Path] | None = None) -> None:
    """Set variables from `.env` files that are not already in the environment (shell wins)."""
    for path in paths if paths is not None else (Path.cwd() / ".env", _PROJECT_ROOT / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip().removeprefix("export ").strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            name, _, value = text.partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))
