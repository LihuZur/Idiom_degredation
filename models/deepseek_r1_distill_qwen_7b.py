"""Model runner registration for DeepSeek-R1-Distill-Qwen-7B (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "deepseek-r1-distill-qwen-7b",
    hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    revision="916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    kind="decoder",
    default_precision="bf16",
)
class DeepSeekR1DistillQwen7BRunner(DecoderRunner):
    """Runner for deepseek-r1-distill-qwen-7b model."""
