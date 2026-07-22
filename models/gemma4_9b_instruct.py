"""Model runner registration for Gemma 4 9B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "gemma-4-9b-instruct",
    hf_repo="google/gemma-2-9b-it",
    revision="11c9b309abf73637e4b6f9a3fa1e92e615547819",
    kind="decoder",
    default_precision="bf16",
)
class Gemma9BRunner(DecoderRunner):
    """Runner for gemma-4-9b-instruct model."""
