"""Model runner registration for TII Falcon3 3B Instruct."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "falcon3-3b-instruct",
    hf_repo="tiiuae/Falcon3-3B-Instruct",
    revision="411bb94318f94f7a5735b77109f456b1e74b42a1",
    kind="decoder",
    default_precision="bf16",
)
class Falcon3Runner(DecoderRunner):
    """Runner for falcon3-3b-instruct model."""
