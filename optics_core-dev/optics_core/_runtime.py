from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


def default_device(system: MultiOpticalSystem) -> torch.device:
    configured_device = system.config.backend.device
    if configured_device is None:
        return torch.device("cpu")
    return torch.device(configured_device)