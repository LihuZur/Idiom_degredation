"""Smoke tests: package imports, registries start empty, device helper works."""

import torch

from augmentation import registry as aug_reg
from data import registry as data_reg
from models import registry as model_reg
from models.device import select_device


def test_dataset_registry_has_sst2_and_mmlu() -> None:
    assert set(data_reg.list_datasets()) == {"sst2", "mmlu"}
    assert model_reg.list_models() == ["mistral-7b-instruct-v0.3", "qwen3.5-1.5b-instruct"]
    assert aug_reg.list_augmenters() == ["identity"]
    assert set(aug_reg.list_validators()) == {
        "semantic_similarity",
        "label_preservation",
        "idiom_presence",
        "idiom_absence",
    }


def test_select_device_returns_valid_torch_device() -> None:
    dev = select_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}
