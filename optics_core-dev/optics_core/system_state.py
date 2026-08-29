from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class FrameData:
    """缓存每个表面的全局 frame。"""

    rotations: torch.Tensor
    origins: torch.Tensor
    device: torch.device

    def surface_z(self, surface_index: int) -> torch.Tensor:
        """返回指定表面的全局轴向位置。"""
        return self.origins[:, surface_index, 2]

    def design_slice(self, start: int, stop: int) -> FrameData:
        """返回连续 design 区间的轻量视图。"""
        return FrameData(
            rotations=self.rotations[start:stop],
            origins=self.origins[start:stop],
            device=self.device,
        )


@dataclass(slots=True)
class FirstOrderData:
    """缓存系统一阶光学量。"""

    effl: torch.Tensor
    working_f_number: torch.Tensor
    ttl: torch.Tensor
    image_plane_distance: torch.Tensor
    bfl: torch.Tensor
    valid: torch.Tensor
    entrance_pupil_z: torch.Tensor
    entrance_pupil_radius: torch.Tensor
    stop_radius: torch.Tensor
    exit_pupil_z: torch.Tensor
    exit_pupil_radius: torch.Tensor

    def design_slice(self, start: int, stop: int) -> FirstOrderData:
        """返回连续 design 区间的一阶量视图。"""
        return FirstOrderData(
            effl=self.effl[start:stop],
            working_f_number=self.working_f_number[start:stop],
            ttl=self.ttl[start:stop],
            image_plane_distance=self.image_plane_distance[start:stop],
            bfl=self.bfl[start:stop],
            valid=self.valid[start:stop],
            entrance_pupil_z=self.entrance_pupil_z[start:stop],
            entrance_pupil_radius=self.entrance_pupil_radius[start:stop],
            stop_radius=self.stop_radius[start:stop],
            exit_pupil_z=self.exit_pupil_z[start:stop],
            exit_pupil_radius=self.exit_pupil_radius[start:stop],
        )
