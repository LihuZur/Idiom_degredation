"""Model runner registration for H2O.ai Danube3 4B Chat."""

from models.decoder_runner import DecoderRunner
from models.registry import register_model


@register_model(
    "h2o-danube3-4b-chat",
    hf_repo="h2oai/h2o-danube3-4b-chat",
    revision="1e5c6fa6620f8bf078958069ab4581cd88e0202c",
    kind="decoder",
    default_precision="bf16",
)
class Danube3Runner(DecoderRunner):
    """Runner for h2o-danube3-4b-chat model."""
