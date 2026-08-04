"""Model runner registration for Qwen 3.5 0.5B Instruct (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "qwen3.5-0.5b-instruct",
    hf_repo="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    kind="decoder",
    default_precision="bf16",
)
class Qwen0p5BRunner(DecoderRunner):
    """Runner for qwen3.5-0.5b-instruct model."""
