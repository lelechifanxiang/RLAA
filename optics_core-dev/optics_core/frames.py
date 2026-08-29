from __future__ import annotations

import torch

from ._runtime import default_device
from .surfaces import CoordinateBreak
from .system_state import FrameData
from .tracing._dispatch import _advance_frame_after_surface, _coordinate_break_surface_frame, _identity_frame

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


def compute_frame_data(
    system: MultiOpticalSystem,
    *,
    device: torch.device | None = None,
    ignore_coordinate_breaks: bool = False,
) -> FrameData:
    """批量计算每个表面的追迹 frame。"""
    target_device = default_device(system) if device is None else device
    frame_rotation, frame_origin = _identity_frame(system.system_count, device=target_device)
    rotations: list[torch.Tensor] = []
    origins: list[torch.Tensor] = []

    for surface_index, surface in enumerate(system.surfaces):
        # CB 面本身仍使用父 frame 进入 dispatch，由 dispatch 内部应用 CB 变换。
        rotations.append(frame_rotation)
        origins.append(frame_origin)

        if isinstance(surface, CoordinateBreak) and not ignore_coordinate_breaks:
            surface_rotation, surface_origin = _coordinate_break_surface_frame(
                system,
                surface,
                surface_index,
                parent_rotation=frame_rotation,
                parent_origin=frame_origin,
                device=target_device,
            )
        else:
            surface_rotation = frame_rotation
            surface_origin = frame_origin

        frame_rotation, frame_origin = _advance_frame_after_surface(
            system,
            surface_index,
            surface_rotation=surface_rotation,
            surface_origin=surface_origin,
            device=target_device,
        )

    return FrameData(
        rotations=torch.stack(rotations, dim=1),
        origins=torch.stack(origins, dim=1),
        device=target_device,
    )
