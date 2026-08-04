"""Model runner registration for Gemma 4 2B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "gemma-4-2b-instruct",
    hf_repo="google/gemma-2-2b-it",
    revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    kind="decoder",
    default_precision="bf16",
)
class Gemma2BRunner(DecoderRunner):
    """Runner for gemma-4-2b-instruct model."""
