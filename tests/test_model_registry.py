from models.mistral7b_instruct_v03 import MistralRunner
from models.qwen3p5_1p5b_instruct import QwenRunner
from models.registry import MODELS, get_model_class, get_model_spec


def test_registered_models() -> None:

    assert "qwen3.5-1.5b-instruct" in MODELS
    assert "mistral-7b-instruct-v0.3" in MODELS

    # Verify specs
    qwen_spec = get_model_spec("qwen3.5-1.5b-instruct")
    assert qwen_spec.hf_repo == "Qwen/Qwen2.5-1.5B-Instruct"
    assert isinstance(qwen_spec.revision, str)
    assert len(qwen_spec.revision) > 0

    mistral_spec = get_model_spec("mistral-7b-instruct-v0.3")
    assert mistral_spec.hf_repo == "mistralai/Mistral-7B-Instruct-v0.3"
    assert isinstance(mistral_spec.revision, str)
    assert len(mistral_spec.revision) > 0

    assert get_model_class("qwen3.5-1.5b-instruct") is QwenRunner
    assert get_model_class("mistral-7b-instruct-v0.3") is MistralRunner
