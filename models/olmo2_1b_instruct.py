"""Model runner registration for AllenAI OLMo 2 1B Instruct."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "olmo-2-1b-instruct",
    hf_repo="allenai/OLMo-2-0425-1B-Instruct",
    revision="48d788eca847d4d7548f375ad03d3c9312f6139e",
    kind="decoder",
    default_precision="bf16",
)
class OlmoRunner(DecoderRunner):
    """Runner for olmo-2-1b-instruct model."""
