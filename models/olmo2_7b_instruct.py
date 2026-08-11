"""Model runner registration for AllenAI OLMo 2 7B Instruct."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "olmo-2-7b-instruct",
    hf_repo="allenai/OLMo-2-1124-7B-Instruct",
    revision="470b1fba1ae01581f270116362ee4aa1b97f4c84",
    kind="decoder",
    default_precision="bf16",
)
class Olmo7BRunner(DecoderRunner):
    """Runner for olmo-2-7b-instruct model."""
