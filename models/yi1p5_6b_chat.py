"""Model runner registration for 01-AI Yi 1.5 6B Chat."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "yi-1.5-6b-chat",
    hf_repo="01-ai/Yi-1.5-6B-Chat",
    revision="771924d1c83d67527d665913415d7086f11ea9c0",
    kind="decoder",
    default_precision="bf16",
)
class Yi1p5Runner(DecoderRunner):
    """Runner for yi-1.5-6b-chat model."""
