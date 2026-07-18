"""Tests for augmentation/config.py (STAGE2_CONTRACT AugmentConfig section)."""

from typing import Any

import pytest
from pydantic import ValidationError

from augmentation.config import AugmentConfig


def _valid_dict(dataset: str = "sst2") -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": 0,
        "augmenter": "identity",
        "prompts": {"paraphrase": "paraphrase_v1.txt", "idiomatic": "idiomatic_v1.txt"},
        "validators": {
            "semantic_similarity": {"embedding_model": None, "min_cosine": 0.80},
            "label_preservation": {"method": "llm_judge"},
            "idiom_presence": {"method": "llm_judge"},
            "idiom_absence": {"method": "llm_judge"},
        },
        "cache": {"enabled": True, "dir": ".hf_cache/augment"},
    }


def test_valid_sst2_shaped_dict_parses() -> None:
    cfg = AugmentConfig.model_validate(_valid_dict("sst2"))
    assert cfg.dataset == "sst2"
    assert cfg.seed == 0
    assert cfg.augmenter == "identity"
    assert cfg.prompts.paraphrase == "paraphrase_v1.txt"
    assert cfg.prompts.idiomatic == "idiomatic_v1.txt"
    assert cfg.validators.semantic_similarity.min_cosine == 0.80
    assert cfg.cache.enabled is True
    assert cfg.cache.dir == ".hf_cache/augment"


def test_valid_mmlu_shaped_dict_parses() -> None:
    cfg = AugmentConfig.model_validate(_valid_dict("mmlu"))
    assert cfg.dataset == "mmlu"


def test_unknown_top_level_key_rejected() -> None:
    raw = _valid_dict()
    raw["unknown_key"] = 1
    with pytest.raises(ValidationError):
        AugmentConfig.model_validate(raw)


def test_unknown_augmenter_name_rejected() -> None:
    raw = _valid_dict()
    raw["augmenter"] = "not_a_real_augmenter"
    with pytest.raises(ValidationError):
        AugmentConfig.model_validate(raw)


def test_defaults_applied_when_validators_and_cache_omitted() -> None:
    raw = _valid_dict()
    del raw["validators"]
    del raw["cache"]
    cfg = AugmentConfig.model_validate(raw)
    assert cfg.validators.semantic_similarity.min_cosine == 0.80
    assert cfg.cache.enabled is True
    assert cfg.cache.dir == ".hf_cache/augment"


def test_unknown_nested_prompts_key_rejected() -> None:
    raw = _valid_dict()
    raw["prompts"]["extra"] = "nope"
    with pytest.raises(ValidationError):
        AugmentConfig.model_validate(raw)
