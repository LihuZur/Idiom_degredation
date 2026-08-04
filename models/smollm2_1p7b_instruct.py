"""Model runner registration for SmolLM2 1.7B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "smollm2-1.7b-instruct",
    hf_repo="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    revision="31b70e2e869a7173562077fd711b654946d38674",
    kind="decoder",
    default_precision="bf16",
)
class SmolLMRunner(DecoderRunner):
    """Runner for smollm2-1.7b-instruct model."""
