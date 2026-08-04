"""Model runner registration for Llama 3.2 1B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "llama-3.2-1b-instruct",
    hf_repo="meta-llama/Llama-3.2-1B-Instruct",
    revision="9213176726f574b556790deb65791e0c5aa438b6",
    kind="decoder",
    default_precision="bf16",
)
class LlamaRunner(DecoderRunner):
    """Runner for llama-3.2-1b-instruct model."""
