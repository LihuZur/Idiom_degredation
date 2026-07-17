"""Unit tests for DecoderRunner using a tiny model (STAGE3_PLAN §5.8)."""

from models.base import FormattedInput, ModelSpec
from models.decoder_runner import DecoderRunner


def test_decoder_runner_tiny_gpt2() -> None:
    # sshleifer/tiny-gpt2 is a tiny ~5MB model on HF
    spec = ModelSpec(
        name="tiny-gpt2",
        hf_repo="sshleifer/tiny-gpt2",
        revision="5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be",
        kind="decoder",
        default_precision="fp32",
    )

    # tiny-gpt2 has no chat template, so fallback template is required
    fallback_template = "<SYSTEM>{system}</SYSTEM><USER>{user}</USER>"

    runner = DecoderRunner(
        spec,
        precision="fp32",
        fallback_template=fallback_template,
    )

    # Check construction attributes
    assert runner.id == "tiny-gpt2"
    assert runner.device.type in ("cpu", "cuda", "mps")

    # Construct a sample FormattedInput
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    fi = FormattedInput(
        id="test-1",
        prompt="",
        meta={"messages": messages, "generate_kwargs": {"max_new_tokens": 5}},
    )

    # Run prediction
    preds = runner.predict([fi])
    assert len(preds) == 1
    assert preds[0].id == "test-1"
    assert isinstance(preds[0].raw, str)
    assert len(preds[0].raw) >= 0
