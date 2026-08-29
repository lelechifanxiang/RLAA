from __future__ import annotations

from typing import Any

import torch

from tests.zemax.common import zp


def fetch_zemax_chief_referenced_opd(
    spec: Any,
    oss: Any,
    pupil_coordinates: torch.Tensor,
    *,
    field_index: int,
    wavelength_index: int,
) -> torch.Tensor:
    """使用 Zemax Batch Ray Trace 读取主光线参考 OPD，单位为 waves。"""
    coordinates = torch.as_tensor(pupil_coordinates, dtype=torch.float64)
    field_x, field_y = spec.field_points[field_index]
    edge_x = max((abs(x) for x, _ in spec.field_points), default=0.0)
    edge_y = max((abs(y) for _, y in spec.field_points), default=0.0)
    hx = field_x / edge_x if edge_x > 0.0 else 0.0
    hy = field_y / edge_y if edge_y > 0.0 else 0.0

    tool = oss.Tools.OpenBatchRayTrace()
    try:
        rays = tool.CreateNormUnpol(
            coordinates.shape[0],
            zp.constants.Tools.RayTrace.RaysType.Real,
            spec.image_surface_index,
        )
        opd_mode = zp.constants.Tools.RayTrace.OPDMode.CurrentAndChief
        for pupil_x, pupil_y in coordinates.tolist():
            rays.AddRay(
                wavelength_index + 1,
                hx,
                hy,
                pupil_x,
                pupil_y,
                opd_mode,
            )

        tool.RunAndWaitForCompletion()
        rays.StartReadingResults()
        opd_waves = torch.empty(coordinates.shape[0], dtype=torch.float64)
        for ray_index in range(coordinates.shape[0]):
            result = rays.ReadNextResult()
            if not result[0] or result[2] != 0 or result[3] != 0:
                raise ValueError(f"Zemax OPD ray trace failed: {result}")
            opd_waves[ray_index] = float(result[13])
        return opd_waves
    finally:
        tool.Close()
