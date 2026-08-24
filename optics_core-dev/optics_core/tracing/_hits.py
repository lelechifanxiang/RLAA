from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._parameter_access import surface_value
from ..apertures import _inside_circular_aperture
from ..geometries import StandardGeometry
from ..surfaces import Surface
from ..types import TraceDirection
from ._interactions import _travel_direction
from ._types import SurfaceHit

if TYPE_CHECKING:
    from ..system import MultiOpticalSystem


def _local_plane_hit(
    surface_index: int,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> SurfaceHit:
    """计算局部坐标系中 z=0 平面和光线的交点。"""
    surface_z = torch.zeros_like(z)
    return _plane_hit_at_z(
        surface_index,
        surface_z,
        x,
        y,
        z,
        l,
        m,
        n,
        direction=direction,
    )


def _plane_hit_at_z(
    surface_index: int,
    surface_z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> SurfaceHit:
    """在指定 z 平面上计算交点，供全局轴向平面和局部平面共用。"""
    travel_l, travel_m, travel_n = _travel_direction(l, m, n, direction=direction)

    zeros = torch.zeros_like(z)
    dz = surface_z - z
    on_plane = torch.isclose(dz, zeros)
    parallel = torch.isclose(travel_n, zeros)
    safe_travel_n = torch.where(parallel, torch.ones_like(travel_n), travel_n)
    path = dz / safe_travel_n
    valid = on_plane | (~parallel)
    path = torch.where(on_plane, zeros, path)
    safe_path = torch.where(valid, path, zeros)

    hit_x = x + travel_l * safe_path
    hit_y = y + travel_m * safe_path
    hit_z = torch.where(valid, surface_z, torch.full_like(surface_z, torch.nan))
    hit_x = torch.where(valid, hit_x, torch.full_like(hit_x, torch.nan))
    hit_y = torch.where(valid, hit_y, torch.full_like(hit_y, torch.nan))

    normal_x = torch.zeros_like(hit_x)
    normal_y = torch.zeros_like(hit_y)
    normal_z = torch.ones_like(hit_z)
    axial_offset = torch.zeros_like(hit_z)
    return SurfaceHit(
        surface_index=surface_index,
        position=(hit_x, hit_y, hit_z),
        normal=(normal_x, normal_y, normal_z),
        valid=valid,
        axial_offset=axial_offset,
    )


def _split_local_vector(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return vector[..., 0], vector[..., 1], vector[..., 2]


def _apply_surface_aperture(
    system: MultiOpticalSystem,
    surface: Surface,
    surface_index: int,
    hit: SurfaceHit,
) -> SurfaceHit:
    """按 Surface Aperture 裁剪局部交点。"""
    if surface.aperture_type == "none":
        return hit

    hit_x, hit_y, hit_z = hit.position
    aperture_radius = surface_value(
        system,
        surface_index,
        "semi_diameter",
        surface.semi_diameter,
        batch_ndim=hit_x.ndim - 1,
        device=hit_x.device,
    )
    valid = hit.valid & _inside_circular_aperture(hit_x, hit_y, aperture_radius)
    nan_tensor = torch.full_like(hit_x, torch.nan)
    normal = None
    if hit.normal is not None:
        normal = tuple(torch.where(valid, component, nan_tensor) for component in hit.normal)
    return SurfaceHit(
        surface_index=surface_index,
        position=tuple(torch.where(valid, component, nan_tensor) for component in (hit_x, hit_y, hit_z)),
        normal=normal,
        valid=valid,
        axial_offset=torch.where(valid, hit.axial_offset, nan_tensor),
    )


def _standard_geometry_intersect(
    geometry: StandardGeometry,
    surface: Surface,
    system: MultiOpticalSystem,
    surface_index: int,
    local_origin: torch.Tensor,
    local_direction: torch.Tensor,
) -> torch.Tensor:
    if len(getattr(geometry, "coefficients", ())) > 0:
        raise NotImplementedError("EvenAsphereGeometry with non-empty coefficients is not implemented yet.")

    ox, oy, oz = _split_local_vector(local_origin)
    dx, dy, dz = _split_local_vector(local_direction)
    batch_ndim = ox.ndim - 1
    radius = surface_value(
        system,
        surface_index,
        "geometry.radius",
        geometry.radius,
        batch_ndim=batch_ndim,
        device=ox.device,
    )
    conic = surface_value(
        system,
        surface_index,
        "geometry.conic",
        geometry.conic,
        batch_ndim=batch_ndim,
        device=ox.device,
    )

    zeros = torch.zeros_like(oz)
    nan_tensor = torch.full_like(oz, torch.nan)
    plane = torch.isclose(radius, torch.zeros_like(radius))

    parallel = torch.isclose(dz, zeros)
    safe_dz = torch.where(parallel, torch.ones_like(dz), dz)
    plane_path = -oz / safe_dz
    plane_path = torch.where(~parallel, plane_path, nan_tensor)
    plane_path = torch.where(torch.isclose(oz, zeros), zeros, plane_path)

    conic_factor = 1.0 + conic
    quadratic_a = dx * dx + dy * dy + conic_factor * dz * dz
    half_b = ox * dx + oy * dy + (conic_factor * oz - radius) * dz
    quadratic_c = ox * ox + oy * oy + conic_factor * oz * oz - 2.0 * radius * oz
    discriminant = half_b * half_b - quadratic_a * quadratic_c

    degenerate = torch.isclose(quadratic_a, torch.zeros_like(quadratic_a))
    safe_a = torch.where(degenerate, torch.ones_like(quadratic_a), quadratic_a)
    sqrt_discriminant = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    root_near = (-half_b - sqrt_discriminant) / safe_a
    root_far = (-half_b + sqrt_discriminant) / safe_a
    # 顺序追迹由表面序号指定目标面，路径可为负，选择离当前位置最近的交点。
    near_is_closer = torch.abs(root_near) <= torch.abs(root_far)
    standard_path = torch.where(near_is_closer, root_near, root_far)
    standard_path = torch.where((discriminant >= 0.0) & (~degenerate), standard_path, nan_tensor)

    return torch.where(plane, plane_path, standard_path)


def _standard_geometry_normal(
    geometry: StandardGeometry,
    surface: Surface,
    system: MultiOpticalSystem,
    surface_index: int,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(getattr(geometry, "coefficients", ())) > 0:
        raise NotImplementedError("EvenAsphereGeometry with non-empty coefficients is not implemented yet.")

    batch_ndim = x.ndim - 1
    radius = surface_value(
        system,
        surface_index,
        "geometry.radius",
        geometry.radius,
        batch_ndim=batch_ndim,
        device=x.device,
    )
    conic = surface_value(
        system,
        surface_index,
        "geometry.conic",
        geometry.conic,
        batch_ndim=batch_ndim,
        device=x.device,
    )
    plane = torch.isclose(radius, torch.zeros_like(radius))
    safe_radius = torch.where(plane, torch.ones_like(radius), radius)

    rho_sq = x * x + y * y
    sqrt_term = torch.sqrt(torch.clamp_min(1.0 - (1.0 + conic) * rho_sq / (safe_radius * safe_radius), 0.0))
    safe_sqrt = torch.where(sqrt_term > 0.0, sqrt_term, torch.ones_like(sqrt_term))
    dz_dx = x / (safe_radius * safe_sqrt)
    dz_dy = y / (safe_radius * safe_sqrt)
    nx = -dz_dx
    ny = -dz_dy
    nz = torch.ones_like(nx)
    norm = torch.sqrt(nx * nx + ny * ny + nz * nz)
    normal_x = nx / norm
    normal_y = ny / norm
    normal_z = nz / norm

    zeros = torch.zeros_like(normal_x)
    ones = torch.ones_like(normal_z)
    return (
        torch.where(plane, zeros, normal_x),
        torch.where(plane, zeros, normal_y),
        torch.where(plane, ones, normal_z),
    )


def _local_sag_hit(
    system: MultiOpticalSystem,
    surface: Surface,
    surface_index: int,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> SurfaceHit:
    """计算局部坐标系中 sag 面和光线的交点。"""
    surface_z = torch.zeros_like(z)
    return _sag_surface_hit_at_z(
        system,
        surface,
        surface_index,
        surface_z,
        x,
        y,
        z,
        l,
        m,
        n,
        direction=direction,
    )


def _sag_surface_hit_at_z(
    system: MultiOpticalSystem,
    surface: Surface,
    surface_index: int,
    surface_z: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    *,
    direction: TraceDirection,
) -> SurfaceHit:
    """在指定顶点 z 位置计算 sag 面交点，供全局和局部 frame 共用。"""
    travel_l, travel_m, travel_n = _travel_direction(l, m, n, direction=direction)

    local_origin = torch.stack((x, y, z - surface_z), dim=-1)
    local_direction = torch.stack((travel_l, travel_m, travel_n), dim=-1)
    geometry = surface.geometry
    if isinstance(geometry, StandardGeometry):
        path = _standard_geometry_intersect(
            geometry,
            surface,
            system,
            surface_index,
            local_origin,
            local_direction,
        )
    else:
        path = geometry.intersect(local_origin, local_direction)
    valid = torch.isfinite(path)
    safe_path = torch.where(valid, path, torch.zeros_like(path))

    local_hit_x = x + travel_l * safe_path
    local_hit_y = y + travel_m * safe_path
    local_hit_z = z - surface_z + travel_n * safe_path
    hit_z = surface_z + local_hit_z

    if isinstance(geometry, StandardGeometry):
        normal_x, normal_y, normal_z = _standard_geometry_normal(
            geometry,
            surface,
            system,
            surface_index,
            local_hit_x,
            local_hit_y,
        )
    else:
        normal_x, normal_y, normal_z = geometry.normal(local_hit_x, local_hit_y)
    nan_tensor = torch.full_like(local_hit_x, torch.nan)
    hit_x = torch.where(valid, local_hit_x, nan_tensor)
    hit_y = torch.where(valid, local_hit_y, nan_tensor)
    hit_z = torch.where(valid, hit_z, nan_tensor)
    normal_x = torch.where(valid, normal_x, nan_tensor)
    normal_y = torch.where(valid, normal_y, nan_tensor)
    normal_z = torch.where(valid, normal_z, nan_tensor)
    axial_offset = torch.where(valid, local_hit_z, nan_tensor)

    return SurfaceHit(
        surface_index=surface_index,
        position=(hit_x, hit_y, hit_z),
        normal=(normal_x, normal_y, normal_z),
        valid=valid,
        axial_offset=axial_offset,
    )
