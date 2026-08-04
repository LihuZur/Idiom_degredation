"""Model runner registration for Qwen 3.5 3B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "qwen3.5-3b-instruct",
    hf_repo="Qwen/Qwen2.5-3B-Instruct",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    kind="decoder",
    default_precision="bf16",
)
class Qwen3BRunner(DecoderRunner):
    """Runner for qwen3.5-3b-instruct model."""
