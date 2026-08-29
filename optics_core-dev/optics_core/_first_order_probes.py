from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .rays import RayBundle, TraceResult

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


@dataclass(slots=True)
class FrontParaxialData:
    """前组近轴探测结果。"""

    effl: torch.Tensor
    bfl: torch.Tensor
    entrance_pupil_z: torch.Tensor
    front_pupil_magnification: torch.Tensor


@dataclass(slots=True)
class ExitPupilData:
    """后组出瞳探测结果。"""

    exit_pupil_z: torch.Tensor
    rear_pupil_magnification: torch.Tensor


def build_front_paraxial_probe_rays(
    system: MultiOpticalSystem,
    probe_height: torch.Tensor,
) -> RayBundle:
    """构造正负高度、正负斜率共 4 根前组近轴探测光线。"""
    zeros = torch.zeros_like(probe_height)
    x = torch.stack((probe_height, -probe_height, zeros, zeros), dim=-1)
    l = torch.stack((zeros, zeros, probe_height, -probe_height), dim=-1)
    empty = torch.zeros_like(x)
    return RayBundle(
        x=x,
        y=empty,
        z=empty,
        l=l,
        m=empty,
        n=torch.ones_like(x),
        wavelength_index=torch.full_like(x, system.wavelengths.primary_index, dtype=torch.int64),
    )


def solve_front_paraxial_data(
    result: TraceResult,
    *,
    stop_index: int,
    probe_height: torch.Tensor,
    last_surface_z: torch.Tensor,
) -> FrontParaxialData:
    """由 4 根前组近轴光线求焦距、后焦距和入瞳数据。"""
    initial_x = torch.stack((probe_height, -probe_height), dim=-1)
    slope = result.rays.l[:, :2] / result.rays.n[:, :2]
    candidate_effl = -initial_x / slope
    valid = result.valid[:, :2] & (torch.abs(slope) > 1e-12) & torch.isfinite(candidate_effl)
    valid_count = valid.sum(dim=-1)
    effl = torch.where(valid, candidate_effl, torch.zeros_like(candidate_effl)).sum(dim=-1)
    effl = effl / valid_count.clamp_min(1).to(dtype=torch.float64)
    effl = torch.where(valid_count > 0, effl, torch.full_like(effl, torch.nan))

    candidate_bfl = result.rays.z[:, :2] - result.rays.x[:, :2] / slope - last_surface_z.unsqueeze(-1)
    bfl_valid = valid & torch.isfinite(candidate_bfl)
    bfl_count = bfl_valid.sum(dim=-1)
    bfl = torch.where(bfl_valid, candidate_bfl, torch.zeros_like(candidate_bfl)).sum(dim=-1)
    bfl = bfl / bfl_count.clamp_min(1).to(dtype=torch.float64)
    bfl = torch.where(bfl_count > 0, bfl, torch.full_like(bfl, torch.nan))

    stop_x = result.intersections[stop_index].position[0]
    scale = 2.0 * probe_height
    matrix_a = (stop_x[:, 0] - stop_x[:, 1]) / scale
    matrix_b = (stop_x[:, 2] - stop_x[:, 3]) / scale
    return FrontParaxialData(
        effl=effl,
        bfl=bfl,
        entrance_pupil_z=matrix_b / matrix_a,
        front_pupil_magnification=1.0 / matrix_a,
    )


def build_marginal_probe_ray(
    system: MultiOpticalSystem,
    *,
    entrance_pupil_z: torch.Tensor,
    entrance_pupil_radius: torch.Tensor,
) -> RayBundle:
    """构造轴上物点到入瞳边缘的一根真实边缘光线。"""
    zeros = torch.zeros_like(entrance_pupil_radius)
    marginal_height = entrance_pupil_radius
    marginal_slope = zeros
    object_distance_mm = float(system.architecture.object_distance_mm)
    if math.isfinite(object_distance_mm):
        object_z = -object_distance_mm
        marginal_slope = entrance_pupil_radius / (entrance_pupil_z - object_z)
        marginal_height = marginal_slope * (-object_z)

    x = zeros.unsqueeze(-1)
    return RayBundle(
        x=x,
        y=marginal_height.unsqueeze(-1),
        z=torch.zeros_like(x),
        l=torch.zeros_like(x),
        m=marginal_slope.unsqueeze(-1),
        n=torch.ones_like(x),
        wavelength_index=torch.full_like(x, system.wavelengths.primary_index, dtype=torch.int64),
    )


def solve_working_f_number_from_marginal_ray(result: TraceResult) -> torch.Tensor:
    """由单根真实边缘光线的像方数值孔径计算工作 F/#。"""
    slope_l = result.rays.l[:, 0] / result.rays.n[:, 0]
    slope_m = result.rays.m[:, 0] / result.rays.n[:, 0]
    transverse_slope = torch.sqrt(slope_l * slope_l + slope_m * slope_m)
    transverse_direction = transverse_slope / torch.sqrt(1.0 + transverse_slope * transverse_slope)
    valid = result.valid[:, 0] & (transverse_direction > torch.finfo(torch.float64).eps)
    working_f_number = 0.5 / transverse_direction.clamp_min(torch.finfo(torch.float64).eps)
    return torch.where(valid, working_f_number, torch.full_like(working_f_number, torch.nan))


def build_exit_pupil_probe_rays(
    system: MultiOpticalSystem,
    stop_position: torch.Tensor,
    probe_height: torch.Tensor,
) -> RayBundle:
    """构造从 stop 面出发的两根出瞳近轴探测光线。"""
    system_count = system.system_count
    probe_slope = 1e-4
    x = torch.stack((probe_height, probe_height), dim=-1)
    zeros = torch.zeros_like(x)
    l = torch.tensor([0.0, probe_slope], dtype=torch.float64, device=x.device).reshape(1, 2)
    return RayBundle(
        x=x,
        y=zeros,
        z=stop_position.reshape(system_count, 1).expand(-1, 2),
        l=l.expand(system_count, -1),
        m=zeros,
        n=torch.ones_like(x),
        wavelength_index=torch.full_like(x, system.wavelengths.primary_index, dtype=torch.int64),
    )


def solve_exit_pupil_from_probes(
    result: TraceResult,
    *,
    stop_position: torch.Tensor,
    probe_height: torch.Tensor,
) -> ExitPupilData:
    """由两根近轴探测光线反解出瞳位置和后组瞳放大率。"""
    x = result.rays.x
    z = result.rays.z
    slope = result.rays.l / result.rays.n
    denominator = slope[:, 0] - slope[:, 1]
    finite = torch.abs(denominator) > 1e-12
    safe_denominator = torch.where(finite, denominator, torch.ones_like(denominator))
    exit_pupil_z = torch.where(
        finite,
        z[:, 0] + (x[:, 1] - x[:, 0]) / safe_denominator,
        stop_position,
    )
    image_height0 = x[:, 0] + slope[:, 0] * (exit_pupil_z - z[:, 0])
    image_height1 = x[:, 1] + slope[:, 1] * (exit_pupil_z - z[:, 1])
    image_height = 0.5 * (image_height0 + image_height1)
    rear_pupil_magnification = torch.where(
        finite,
        image_height / probe_height,
        torch.ones_like(probe_height),
    )
    return ExitPupilData(
        exit_pupil_z=exit_pupil_z,
        rear_pupil_magnification=rear_pupil_magnification,
    )
