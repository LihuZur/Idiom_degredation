"""Configuration schema for Stage 3 evaluation (STAGE3_PLAN §1)."""

from typing import Literal

from pydantic import BaseModel, Field

Precision = Literal["fp32", "fp16", "bf16"]


class DecodingCfg(BaseModel, extra="forbid"):
    """Decoding options for generative inference."""

    temperature: float = Field(default=0.0, ge=0.0)
    max_new_tokens: int = 32
    do_sample: bool = False


class BatchCfg(BaseModel, extra="forbid"):
    """Inference batching configuration."""

    size: int = 8


class EvalConfig(BaseModel, extra="forbid"):
    """Consolidated configuration for a Stage 3 evaluation run."""

    model: str
    seed: int = 0
    precision: Precision | None = None
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)
    batch: BatchCfg = Field(default_factory=BatchCfg)
