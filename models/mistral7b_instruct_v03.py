"""Model runner registration for Mistral 7B Instruct v0.3 (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "mistral-7b-instruct-v0.3",
    hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
    revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
    kind="decoder",
    default_precision="bf16",
)
class MistralRunner(DecoderRunner):
    """Runner for mistral-7b-instruct-v0.3 model."""
