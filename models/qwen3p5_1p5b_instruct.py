"""Model runner registration for Qwen 3.5 1.5B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "qwen3.5-1.5b-instruct",
    hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    kind="decoder",
    default_precision="bf16",
)
class QwenRunner(DecoderRunner):
    """Runner for qwen3.5-1.5b-instruct model."""
