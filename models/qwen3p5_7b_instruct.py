"""Model runner registration for Qwen 3.5 7B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "qwen3.5-7b-instruct",
    hf_repo="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    kind="decoder",
    default_precision="bf16",
)
class Qwen7BRunner(DecoderRunner):
    """Runner for qwen3.5-7b-instruct model."""
