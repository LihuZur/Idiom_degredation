"""Smoke tests: package imports, registries start empty, device helper works."""

import torch

from augmentation import registry as aug_reg
from data import registry as data_reg
from models import registry as model_reg
from models.device import select_device


def test_registries_start_empty() -> None:
    assert data_reg.list_datasets() == []
    assert model_reg.list_models() == []
    assert aug_reg.list_augmenters() == []


def test_select_device_returns_valid_torch_device() -> None:
    dev = select_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}
