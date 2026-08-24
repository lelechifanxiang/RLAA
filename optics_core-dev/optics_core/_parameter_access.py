from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._runtime import default_device
from .types import Scalar

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


def surface_value(
    system: MultiOpticalSystem,
    surface_index: int,
    suffix: str,
    default: Scalar,
    *,
    batch_ndim: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """读取批量系统中的 surface 参数。"""
    path = f"surface[{surface_index}].{suffix}"
    parameter_index = next(
        (
            system.parameter_schema.index_of(spec.name)
            for spec in system.parameter_schema
            if spec.path == path
        ),
        None,
    )
    values = (
        [float(default)] * system.system_count
        if parameter_index is None
        else [float(system.parameters[index][parameter_index]) for index in range(system.system_count)]
    )
    target_device = default_device(system) if device is None else device
    return torch.tensor(values, dtype=torch.float64, device=target_device).reshape(
        (system.system_count,) + (1,) * batch_ndim
    )
