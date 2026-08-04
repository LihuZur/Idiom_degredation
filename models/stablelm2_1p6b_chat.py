"""Model runner registration for StableLM 2 1.6B Chat (STAGE3_PLAN §2)."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "stablelm-2-1.6b-chat",
    hf_repo="stabilityai/stablelm-2-1_6b-chat",
    revision="f3fe67057c2789ae1bb1fe42b038da99840d4f13",
    kind="decoder",
    default_precision="bf16",
)
class StableLMRunner(DecoderRunner):
    """Runner for stablelm-2-1.6b-chat model."""
