from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SurfaceHit:
    surface_index: int
    position: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    normal: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    valid: torch.Tensor
    axial_offset: torch.Tensor


@dataclass(slots=True)
class RayState:
    """追迹过程中使用的光线状态，明确区分位置和方向。"""

    position: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    direction: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class SurfaceTraceStep:
    """单个表面的全局交点和交互后光线。"""

    hit: SurfaceHit
    outgoing_ray: RayState
