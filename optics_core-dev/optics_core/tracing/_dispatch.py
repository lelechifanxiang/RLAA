from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._material_batch import BatchedMaterialData
from .._parameter_access import surface_value
from ..surfaces import (
    CoordinateBreak,
    EvenAsphereSurface,
    ImageSurface,
    ObjectSurface,
    ParaxialSurface,
    SphereSurface,
    Surface,
)
from ..types import TraceDirection
from ._hits import _apply_surface_aperture, _local_plane_hit, _local_sag_hit
from ._interactions import _apply_surface_interaction
from ._types import RayState, SurfaceHit, SurfaceTraceStep

if TYPE_CHECKING:
    from ..system import MultiOpticalSystem


def _trace_surface_with_frame(
    system: MultiOpticalSystem,
    surface_index: int,
    ray: RayState,
    material_data: BatchedMaterialData,
    wavelength_index: torch.Tensor,
    *,
    direction: TraceDirection,
    frame_rotation: torch.Tensor,
    frame_origin: torch.Tensor,
    ignore_coordinate_breaks: bool = False,
    ignore_apertures: bool = False,
) -> SurfaceTraceStep:
    """在显式局部 frame 中追迹一个表面，并返回表面后的光线状态。"""
    surface = system.surfaces[surface_index]

    # 普通表面使用当前 frame；CB 面会在专用分支中决定是否切换到 post frame。
    if isinstance(surface, CoordinateBreak):
        local_ray = _global_to_local_ray(frame_rotation, frame_origin, ray)
        return _trace_coordinate_break_surface(
            system,
            surface,
            surface_index,
            ray,
            local_ray,
            direction=direction,
            frame_rotation=frame_rotation,
            frame_origin=frame_origin,
            ignore_coordinate_breaks=ignore_coordinate_breaks,
        )

    surface_rotation, surface_origin = _ordinary_surface_frame(
        system,
        surface,
        surface_index,
        parent_rotation=frame_rotation,
        parent_origin=frame_origin,
        device=ray.position[0].device,
    )
    local_ray = _global_to_local_ray(surface_rotation, surface_origin, ray)

    if isinstance(surface, ParaxialSurface):
        return _trace_paraxial_surface(
            system,
            surface,
            surface_index,
            local_ray,
            direction=direction,
            frame_rotation=surface_rotation,
            frame_origin=surface_origin,
            ignore_apertures=ignore_apertures,
        )

    # 普通光学面：先在局部 frame 求交，再转回全局 frame 应用折反射。
    local_hit = _intersect_local_surface(
        system,
        surface,
        surface_index,
        local_ray,
        direction=direction,
        ignore_apertures=ignore_apertures,
    )
    global_hit = _local_hit_to_global(local_hit, surface_rotation, surface_origin)
    out_l, out_m, out_n = _apply_surface_interaction(
        surface,
        surface_index,
        global_hit,
        material_data,
        wavelength_index,
        *ray.direction,
        direction=direction,
    )
    return SurfaceTraceStep(
        hit=global_hit,
        outgoing_ray=RayState(position=global_hit.position, direction=(out_l, out_m, out_n)),
    )


def _trace_coordinate_break_surface(
    system: MultiOpticalSystem,
    surface: CoordinateBreak,
    surface_index: int,
    ray: RayState,
    local_ray: RayState,
    *,
    direction: TraceDirection,
    frame_rotation: torch.Tensor,
    frame_origin: torch.Tensor,
    ignore_coordinate_breaks: bool,
) -> SurfaceTraceStep:
    """追迹坐标间断面，并按选项决定是否应用 CB frame。"""
    if ignore_coordinate_breaks:
        # 一阶近轴计算按 Zemax 语义忽略坐标间断，只把该面当作当前 frame 下的通过平面。
        local_hit = _local_plane_hit(
            surface_index,
            *local_ray.position,
            *local_ray.direction,
            direction=direction,
        )
        global_hit = _local_hit_to_global(local_hit, frame_rotation, frame_origin)
        return SurfaceTraceStep(
            hit=global_hit,
            outgoing_ray=RayState(position=global_hit.position, direction=ray.direction),
        )

    # 正常 CB 先生成 post frame，再在 post frame 的 z=0 平面求交。
    post_rotation, post_origin = _coordinate_break_surface_frame(
        system,
        surface,
        surface_index,
        parent_rotation=frame_rotation,
        parent_origin=frame_origin,
        device=ray.position[0].device,
    )
    post_local_ray = _global_to_local_ray(post_rotation, post_origin, ray)
    local_hit = _local_plane_hit(
        surface_index,
        *post_local_ray.position,
        *post_local_ray.direction,
        direction=direction,
    )
    global_hit = _local_hit_to_global(local_hit, post_rotation, post_origin)
    return SurfaceTraceStep(
        hit=global_hit,
        outgoing_ray=RayState(position=global_hit.position, direction=ray.direction),
    )


def _trace_paraxial_surface(
    system: MultiOpticalSystem,
    surface: ParaxialSurface,
    surface_index: int,
    local_ray: RayState,
    *,
    direction: TraceDirection,
    frame_rotation: torch.Tensor,
    frame_origin: torch.Tensor,
    ignore_apertures: bool,
) -> SurfaceTraceStep:
    """追迹近轴面：平面求交后在局部方向余弦上施加薄透镜偏折。"""
    local_l, local_m, local_n = local_ray.direction
    local_hit = _local_plane_hit(
        surface_index,
        *local_ray.position,
        local_l,
        local_m,
        local_n,
        direction=direction,
    )
    if not ignore_apertures:
        local_hit = _apply_surface_aperture(system, surface, surface_index, local_hit)
    global_hit = _local_hit_to_global(local_hit, frame_rotation, frame_origin)

    # 近轴面只改变局部方向余弦，不改变交点位置。
    hit_local_x, hit_local_y, _ = local_hit.position
    focal_length = surface_value(
        system,
        surface_index,
        "geometry.focal_length",
        surface.geometry.focal_length,
        batch_ndim=hit_local_x.ndim - 1,
        device=hit_local_x.device,
    )
    if direction == "forward":
        local_l = local_l - hit_local_x / focal_length
        local_m = local_m - hit_local_y / focal_length
    else:
        local_l = local_l + hit_local_x / focal_length
        local_m = local_m + hit_local_y / focal_length
    out_l, out_m, out_n = _local_to_global_direction(frame_rotation, local_l, local_m, local_n)
    return SurfaceTraceStep(
        hit=global_hit,
        outgoing_ray=RayState(position=global_hit.position, direction=(out_l, out_m, out_n)),
    )


def _intersect_local_surface(
    system: MultiOpticalSystem,
    surface: Surface,
    surface_index: int,
    local_ray: RayState,
    *,
    direction: TraceDirection,
    ignore_apertures: bool,
) -> SurfaceHit:
    """根据表面类型选择局部求交核。"""
    if isinstance(surface, (ObjectSurface, ImageSurface)):
        hit = _local_plane_hit(
            surface_index,
            *local_ray.position,
            *local_ray.direction,
            direction=direction,
        )
    elif isinstance(surface, (SphereSurface, EvenAsphereSurface)):
        hit = _local_sag_hit(
            system,
            surface,
            surface_index,
            *local_ray.position,
            *local_ray.direction,
            direction=direction,
        )
    else:
        raise NotImplementedError(
            f"SequentialSurfaceRayTracer does not support surface geometry at surface {surface_index} "
            f"(surface={type(surface).__name__!r}, geometry={type(surface.geometry).__name__!r})."
        )
    return hit if ignore_apertures else _apply_surface_aperture(system, surface, surface_index, hit)


def _local_hit_to_global(
    local_hit: SurfaceHit,
    frame_rotation: torch.Tensor,
    frame_origin: torch.Tensor,
) -> SurfaceHit:
    """把局部交点转换为全局交点。"""
    return _surface_hit_local_to_global(local_hit, frame_rotation, frame_origin)


def _identity_frame(
    system_count: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成批量恒等 frame。"""
    rotation = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0).repeat(system_count, 1, 1)
    origin = torch.zeros((system_count, 3), dtype=torch.float64, device=device)
    return rotation, origin


def _advance_frame_after_surface(
    system: MultiOpticalSystem,
    surface_index: int,
    *,
    surface_rotation: torch.Tensor,
    surface_origin: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿当前 surface 的局部 z 轴推进到下一面的顶点 frame。"""
    thickness = surface_value(
        system,
        surface_index,
        "gap.thickness",
        system.surfaces[surface_index].gap.thickness,
        batch_ndim=0,
        device=device,
    ).reshape(system.system_count, 1)
    next_origin = surface_origin + surface_rotation[:, :, 2] * thickness
    return surface_rotation, next_origin


def _global_to_local_ray(
    rotation: torch.Tensor,
    origin: torch.Tensor,
    ray: RayState,
) -> RayState:
    """把全局光线状态转换到当前局部 frame。"""
    return RayState(
        position=_global_to_local_position(rotation, origin, *ray.position),
        direction=_global_to_local_direction(rotation, *ray.direction),
    )


def _global_to_local_position(
    rotation: torch.Tensor,
    origin: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把全局位置转换到当前局部 frame。"""
    centered = torch.stack((x, y, z), dim=-1) - _expand_origin(origin, x.ndim - 1)
    local = torch.einsum("sji,s...j->s...i", rotation, centered)
    return local[..., 0], local[..., 1], local[..., 2]


def _global_to_local_direction(
    rotation: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把全局方向余弦转换到当前局部 frame。"""
    local = torch.einsum("sji,s...j->s...i", rotation, torch.stack((l, m, n), dim=-1))
    return local[..., 0], local[..., 1], local[..., 2]


def _local_to_global_position(
    rotation: torch.Tensor,
    origin: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把局部位置转换回全局坐标。"""
    global_position = torch.einsum("sij,s...j->s...i", rotation, torch.stack((x, y, z), dim=-1))
    global_position = global_position + _expand_origin(origin, x.ndim - 1)
    return global_position[..., 0], global_position[..., 1], global_position[..., 2]


def _local_to_global_direction(
    rotation: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把局部方向余弦转换回全局坐标。"""
    global_direction = torch.einsum("sij,s...j->s...i", rotation, torch.stack((l, m, n), dim=-1))
    return global_direction[..., 0], global_direction[..., 1], global_direction[..., 2]


def _expand_origin(origin: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    """把 frame 原点扩展到光线批量维度。"""
    return origin.reshape((origin.shape[0],) + (1,) * batch_ndim + (3,))


def _surface_hit_local_to_global(
    local_hit: SurfaceHit,
    rotation: torch.Tensor,
    origin: torch.Tensor,
) -> SurfaceHit:
    """把局部交点快照转换成全局交点快照。"""
    hit_x, hit_y, hit_z = _local_to_global_position(rotation, origin, *local_hit.position)
    normal = None
    if local_hit.normal is not None:
        normal_x, normal_y, normal_z = _local_to_global_direction(rotation, *local_hit.normal)
        normal = (normal_x, normal_y, normal_z)
    return SurfaceHit(
        surface_index=local_hit.surface_index,
        position=(hit_x, hit_y, hit_z),
        normal=normal,
        valid=local_hit.valid,
        axial_offset=local_hit.axial_offset,
    )


def _ordinary_surface_frame(
    system: MultiOpticalSystem,
    surface: Surface,
    surface_index: int,
    *,
    parent_rotation: torch.Tensor,
    parent_origin: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a local frame for a single ordinary surface.

    Unlike a Coordinate Break, this transform is used only while intersecting
    the current surface; the parent frame remains active for later surfaces.
    """
    decenter_x = surface_value(
        system, surface_index, "frame.x", surface.frame.x, batch_ndim=0, device=device
    ).reshape(system.system_count)
    decenter_y = surface_value(
        system, surface_index, "frame.y", surface.frame.y, batch_ndim=0, device=device
    ).reshape(system.system_count)
    tilt_x = torch.deg2rad(
        surface_value(system, surface_index, "frame.rx", surface.frame.rx, batch_ndim=0, device=device)
        .reshape(system.system_count)
    )
    tilt_y = torch.deg2rad(
        surface_value(system, surface_index, "frame.ry", surface.frame.ry, batch_ndim=0, device=device)
        .reshape(system.system_count)
    )
    local_rotation = _rotation_x(tilt_x) @ _rotation_y(tilt_y)
    decenter = torch.stack((decenter_x, decenter_y, torch.zeros_like(decenter_x)), dim=-1)
    rotated_decenter = torch.einsum("sij,sj->si", local_rotation, decenter)
    local_origin = parent_origin + torch.einsum(
        "sij,sj->si", parent_rotation, rotated_decenter
    )
    surface_rotation = parent_rotation @ local_rotation
    return surface_rotation, local_origin


def _coordinate_break_surface_frame(
    system: MultiOpticalSystem,
    surface: CoordinateBreak,
    surface_index: int,
    *,
    parent_rotation: torch.Tensor,
    parent_origin: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """根据坐标间断面的偏心、倾斜和 order 标志生成新 frame。"""
    decenter_x = surface_value(
        system,
        surface_index,
        "frame.x",
        surface.frame.x,
        batch_ndim=0,
        device=device,
    ).reshape(system.system_count)
    decenter_y = surface_value(
        system,
        surface_index,
        "frame.y",
        surface.frame.y,
        batch_ndim=0,
        device=device,
    ).reshape(system.system_count)
    tilt_x = torch.deg2rad(
        surface_value(
            system, surface_index, "frame.rx", surface.frame.rx, batch_ndim=0, device=device
        ).reshape(system.system_count)
    )
    tilt_y = torch.deg2rad(
        surface_value(
            system, surface_index, "frame.ry", surface.frame.ry, batch_ndim=0, device=device
        ).reshape(system.system_count)
    )
    tilt_z = torch.deg2rad(
        surface_value(
            system, surface_index, "frame.rz", surface.frame.rz, batch_ndim=0, device=device
        ).reshape(system.system_count)
    )
    order_flag = surface_value(
        system,
        surface_index,
        "order_flag",
        surface.order_flag,
        batch_ndim=0,
        device=device,
    ).reshape(system.system_count)

    # order=0 使用 XYZ，order!=0 使用反序 ZYX。
    rotation_xyz = _rotation_x(tilt_x) @ _rotation_y(tilt_y) @ _rotation_z(tilt_z)
    rotation_zyx = _rotation_z(tilt_z) @ _rotation_y(tilt_y) @ _rotation_x(tilt_x)
    reverse_order = torch.abs(order_flag) > 0.5
    local_rotation = torch.where(
        reverse_order.reshape(system.system_count, 1, 1),
        rotation_zyx,
        rotation_xyz,
    )

    decenter = torch.stack((decenter_x, decenter_y, torch.zeros_like(decenter_x)), dim=-1)
    rotated_decenter = torch.einsum("sij,sj->si", local_rotation, decenter)
    local_origin = torch.where(reverse_order.reshape(system.system_count, 1), rotated_decenter, decenter)

    surface_rotation = parent_rotation @ local_rotation
    surface_origin = parent_origin + torch.einsum("sij,sj->si", parent_rotation, local_origin)
    return surface_rotation, surface_origin


def _rotation_x(angle: torch.Tensor) -> torch.Tensor:
    """生成绕 x 轴旋转的批量矩阵。"""
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cos_angle, -sin_angle), dim=-1),
            torch.stack((zeros, sin_angle, cos_angle), dim=-1),
        ),
        dim=-2,
    )


def _rotation_y(angle: torch.Tensor) -> torch.Tensor:
    """生成绕 y 轴旋转的批量矩阵。"""
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cos_angle, zeros, sin_angle), dim=-1),
            torch.stack((zeros, ones, zeros), dim=-1),
            torch.stack((-sin_angle, zeros, cos_angle), dim=-1),
        ),
        dim=-2,
    )


def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
    """生成绕 z 轴旋转的批量矩阵。"""
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cos_angle, -sin_angle, zeros), dim=-1),
            torch.stack((sin_angle, cos_angle, zeros), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ),
        dim=-2,
    )
