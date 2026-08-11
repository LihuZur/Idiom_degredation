"""Model runner registration for TII Falcon3 7B Instruct."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "falcon3-7b-instruct",
    hf_repo="tiiuae/Falcon3-7B-Instruct",
    revision="1e57a0ecd176c7c139f289c60a74e57f887c3dfb",
    kind="decoder",
    default_precision="bf16",
)
class Falcon37BRunner(DecoderRunner):
    """Runner for falcon3-7b-instruct model."""
