"""Compute-device selection helper (README §9.4).

Every model runner must obtain its device from `select_device()` — never
hard-code `"cuda"`.
"""

import torch


def select_device() -> torch.device:
    """Return the best available compute device.

    Order of preference:
        1. `cuda` if `torch.cuda.is_available()`.
        2. `mps` if `torch.backends.mps.is_available()` (Apple Silicon).
        3. `cpu` otherwise.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
