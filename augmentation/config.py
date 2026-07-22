"""Stage 2 (augmentation) config schema (STAGE2_PLAN; mirrors cleaning/config.py)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

import augmentation.anthropic_augmenter  # pyright: ignore[reportUnusedImport]
import augmentation.gemini_augmenter  # pyright: ignore[reportUnusedImport]
import augmentation.llm_validators  # pyright: ignore[reportUnusedImport]
import augmentation.openai_augmenter  # pyright: ignore[reportUnusedImport]
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


class DecodingCfg(BaseModel, extra="forbid"):
    """Augmenter decoding params (D7); the cache preserves reproducibility."""

    temperature: float = 0.7
    max_output_tokens: int = 512


class JudgeCfg(BaseModel, extra="forbid"):
    """LLM-judge decoding params (M2); deterministic, short verdicts."""

    temperature: float = 0.0
    max_output_tokens: int = 16


class RetryCfg(BaseModel, extra="forbid"):
    """Retry-then-abort policy for failing/empty rows (D3 / M1)."""

    max_attempts: int = 3
    backoff_seconds: float = 2.0


class CacheCfg(BaseModel, extra="forbid"):
    """Response cache configuration, keyed by (prompt_hash, augmenter_model, input_id)."""

    enabled: bool = True
    dir: str = ".hf_cache/augment"


class AugmentConfig(BaseModel, extra="forbid"):
    """Stage 2 (augmentation) configuration, parsed from `configs/augment/*.yaml`."""

    dataset: str
    seed: int = 0
    augmenter: str
    augmenter_model: str
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)
    judge: JudgeCfg = Field(default_factory=JudgeCfg)
    retry: RetryCfg = Field(default_factory=RetryCfg)
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
