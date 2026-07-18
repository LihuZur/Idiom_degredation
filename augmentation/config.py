"""Stage 2 (augmentation) config schema (STAGE2_PLAN; mirrors cleaning/config.py)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

import augmentation.identity  # pyright: ignore[reportUnusedImport]
import augmentation.validators  # noqa: F401  # pyright: ignore[reportUnusedImport]
from augmentation.registry import AUGMENTERS


class PromptsCfg(BaseModel, extra="forbid"):
    """Frozen prompt template filenames under `augmentation/prompts/`."""

    paraphrase: str
    idiomatic: str


class SemanticSimilarityCfg(BaseModel, extra="forbid"):
    """Semantic-similarity validator settings."""

    embedding_model: str | None = None
    min_cosine: float = 0.80


class LabelPreservationCfg(BaseModel, extra="forbid"):
    """Label-preservation validator settings."""

    method: str = "llm_judge"


class IdiomPresenceCfg(BaseModel, extra="forbid"):
    """Idiom-presence validator settings (applied to the `idiomatic` variant)."""

    method: str = "llm_judge"


class IdiomAbsenceCfg(BaseModel, extra="forbid"):
    """Idiom-absence validator settings (applied to the `paraphrase` variant)."""

    method: str = "llm_judge"


class ValidatorsCfg(BaseModel, extra="forbid"):
    """Validator thresholds/config, keyed by validator name."""

    semantic_similarity: SemanticSimilarityCfg = Field(default_factory=SemanticSimilarityCfg)
    label_preservation: LabelPreservationCfg = Field(default_factory=LabelPreservationCfg)
    idiom_presence: IdiomPresenceCfg = Field(default_factory=IdiomPresenceCfg)
    idiom_absence: IdiomAbsenceCfg = Field(default_factory=IdiomAbsenceCfg)


class CacheCfg(BaseModel, extra="forbid"):
    """Response cache configuration, keyed by (prompt_hash, augmenter_model, input_id)."""

    enabled: bool = True
    dir: str = ".hf_cache/augment"


class AugmentConfig(BaseModel, extra="forbid"):
    """Stage 2 (augmentation) configuration, parsed from `configs/augment/*.yaml`."""

    dataset: str
    seed: int = 0
    augmenter: str
    prompts: PromptsCfg
    validators: ValidatorsCfg = Field(default_factory=ValidatorsCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)

    @field_validator("augmenter")
    @classmethod
    def _check_augmenter_known(cls, name: str) -> str:
        if name not in AUGMENTERS:
            raise ValueError(f"unknown augmenter: {name!r}; known: {sorted(AUGMENTERS)}")
        return name


def load_config(path: Path) -> AugmentConfig:
    """Parse and validate a Stage 2 YAML config file."""
    raw: Any = yaml.safe_load(path.read_text())
    return AugmentConfig.model_validate(raw)
