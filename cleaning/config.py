"""Stage 1 config schema (STAGE1_PLAN §1 Config schema)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

from cleaning.normalize import NORMALIZERS


class LengthCfg(BaseModel, extra="forbid"):
    """Inclusive word-count band used to filter rows after normalization."""

    min_tokens: int
    max_tokens: int


class CleanConfig(BaseModel, extra="forbid"):
    """Stage 1 (cleaning) configuration, parsed from `configs/clean/*.yaml`."""

    dataset: str
    seed: int
    hf_revision: str
    length: LengthCfg
    max_rows: int | None
    normalize: list[str]
    dedupe: bool
    # MMLU-only, reserved for future use; ignored today (STAGE1_PLAN Q3).
    subjects: list[str] | None = None

    @field_validator("normalize")
    @classmethod
    def _check_normalizers_known(cls, names: list[str]) -> list[str]:
        unknown = [n for n in names if n not in NORMALIZERS]
        if unknown:
            raise ValueError(f"unknown normalizer name(s): {unknown}; known: {sorted(NORMALIZERS)}")
        return names


def load_config(path: Path) -> CleanConfig:
    """Parse and validate a Stage 1 YAML config file."""
    raw: Any = yaml.safe_load(path.read_text())
    return CleanConfig.model_validate(raw)
