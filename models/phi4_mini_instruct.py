"""Model runner registration for Phi-4-mini Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "phi-4-mini-instruct",
    hf_repo="microsoft/Phi-4-mini-instruct",
    revision="cfbefacb99257ffa30c83adab238a50856ac3083",
    kind="decoder",
    default_precision="bf16",
)
class Phi4MiniRunner(DecoderRunner):
    """Runner for phi-4-mini-instruct model."""
