from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .rays import TraceOptions, TraceResult
from .sampling import PupilSampler
from .surfaces import ImageSurface

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


def _inside_circular_aperture(
    x: torch.Tensor,
    y: torch.Tensor,
    aperture_radius: torch.Tensor,
) -> torch.Tensor:
    """判断交点是否位于圆孔内，并为 FP64 边界保留微小裕量。"""
    radius_squared = aperture_radius * aperture_radius
    return x * x + y * y <= radius_squared * (1.0 + 2e-12)


@dataclass(slots=True)
class ClearApertureResult:
    """半口径净值计算结果。"""

    semi_diameter: torch.Tensor
    valid: torch.Tensor
    surface_indices: tuple[int, ...]
    trace_result: TraceResult | None = None

    def design_slice(self, start: int, stop: int) -> ClearApertureResult:
        """返回连续 design 区间的净口径视图。"""
        return ClearApertureResult(
            semi_diameter=self.semi_diameter[start:stop],
            valid=self.valid[start:stop],
            surface_indices=self.surface_indices,
            trace_result=None,
        )


def calculate_clear_apertures(
    system: MultiOpticalSystem,
    *,
    sampler: PupilSampler,
    include_image_surface: bool = False,
    keep_trace_result: bool = False,
) -> ClearApertureResult:
    """通过正向追迹交点统计每个面的半口径净值。"""

    # 计算追迹结果，要求记录交点以供后续半口径统计。
    result = system.trace(
        sampler=sampler,
        options=TraceOptions(record_intersections=True),
    )
    if not result.intersections:
        raise ValueError("clear aperture calculation requires recorded intersections.")

    valid = torch.as_tensor(result.valid, dtype=torch.bool)
    semi_diameters: list[torch.Tensor] = []
    valid_by_surface: list[torch.Tensor] = []
    surface_indices: list[int] = []

    # 遍历所有交点，统计每个面上有效交点的最大半径作为半口径净值。
    # intersections：每个元素对应一个 surface，包含所有入瞳采样点在该面上的交点信息。
    frame_data = system.frame_data
    for hit in result.intersections:
        surface = system.surfaces[hit.surface_index]
        if isinstance(surface, ImageSurface) and not include_image_surface:
            continue

        # SemiDiameter 定义在当前表面的局部坐标系中；有 Coordinate Break 时不能使用全局 x/y。
        position = torch.stack(
            [torch.as_tensor(component, dtype=torch.float64) for component in hit.position],
            dim=-1,
        )
        rotation = frame_data.rotations[:, hit.surface_index].to(device=position.device)
        origin = frame_data.origins[:, hit.surface_index].to(device=position.device)
        local_position = torch.einsum(
            "sij,s...j->s...i",
            rotation.transpose(-1, -2),
            position - origin[:, None, None, None, :],
        )
        radius = torch.sqrt(local_position[..., 0] * local_position[..., 0] + local_position[..., 1] * local_position[..., 1])
        hit_valid = valid & torch.isfinite(radius)
        radius = torch.where(hit_valid, radius, torch.zeros_like(radius))

        flattened_radius = radius.reshape(system.system_count, -1)
        flattened_valid = hit_valid.reshape(system.system_count, -1)
        semi_diameters.append(flattened_radius.max(dim=1).values)
        valid_by_surface.append(flattened_valid.any(dim=1))
        surface_indices.append(hit.surface_index)

    # 如果没有任何有效交点，返回空结果。
    if not semi_diameters:
        empty_shape = (system.system_count, 0)
        return ClearApertureResult(
            semi_diameter=torch.zeros(empty_shape, dtype=torch.float64),
            valid=torch.zeros(empty_shape, dtype=torch.bool),
            surface_indices=(),
            trace_result=result if keep_trace_result else None,
        )

    return ClearApertureResult(
        semi_diameter=torch.stack(semi_diameters, dim=1),
        valid=torch.stack(valid_by_surface, dim=1),
        surface_indices=tuple(surface_indices),
        trace_result=result if keep_trace_result else None,
    )
