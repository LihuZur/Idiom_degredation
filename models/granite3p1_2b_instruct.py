"""Model runner registration for IBM Granite 3.1 2B Instruct."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "granite-3.1-2b-instruct",
    hf_repo="ibm-granite/granite-3.1-2b-instruct",
    revision="bbc2aed595bd38bd770263dc3ab831db9794441d",
    kind="decoder",
    default_precision="bf16",
)
class GraniteRunner(DecoderRunner):
    """Runner for granite-3.1-2b-instruct model."""
