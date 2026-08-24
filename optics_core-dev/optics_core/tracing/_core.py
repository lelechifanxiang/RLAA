from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._material_batch import BatchedMaterialData
from ._dispatch import (
    _trace_surface_with_frame,
)
from ._types import RayState, SurfaceHit
from ._sampled_rays import build_input_rays_from_sample
from ..frames import compute_frame_data
from ..rays import (
    RayAimingResult,
    RayBundle,
    SurfaceIntersection,
    SurfaceStateSelection,
    SurfaceTraceHistory,
    TraceOptions,
    TraceResult,
)
from ..sampling import PupilSampler, RayAimer, SquarePupilSampler
from ..surfaces import CoordinateBreak
from ..types import TraceDirection

if TYPE_CHECKING:
    from ..system import MultiOpticalSystem


class SequentialSurfaceRayTracer:
    """基于 SurfaceSequence 的最小顺序追迹器。"""

    def trace(
        self,
        system: MultiOpticalSystem,
        rays: RayBundle,
        options: TraceOptions | None = None,
    ) -> TraceResult:
        """对显式光线按 surface 顺序追迹；续追时传入上一段终态并指定下一面。"""
        options = options or TraceOptions()
        self._validate_options(options)
        return self._trace_surfaces(system, rays, options=options)

    def _trace_surfaces(
        self,
        system: MultiOpticalSystem,
        rays: RayBundle,
        *,
        options: TraceOptions,
    ) -> TraceResult:
        """多个面追迹接口，按 `options.direction` 决定正向或反向。

        forward: 从 start_surface 到 stop_surface 顺序推进。
        backward: 从 start_surface 到 stop_surface 逆序回退。
        """
        # 0. 数据本地化
        system_count = system.system_count
        direction = options.direction
        ray = RayState(
            position=(rays.x, rays.y, rays.z),
            direction=(rays.l, rays.m, rays.n),
        )
        ray_device = ray.position[0].device
        wavelength_index = rays.wavelength_index

        # 0.1 确认材料准备好
        material_data = system._material_data
        if material_data is None:
            raise ValueError("trace requires system.prepare() before run().")

        # 0.2 计算待追迹表面编号
        start_surface, stop_surface = self._surface_bounds(system, options, direction)
        surface_indices = tuple(self._surface_indices(start_surface, stop_surface, direction))

        # 0.3 反向追迹且存在坐标间断时，抛出异常
        contains_coordinate_break = any(
            isinstance(system.surfaces[surface_index], CoordinateBreak)
            for surface_index in surface_indices
        )
        if contains_coordinate_break and direction == "backward" and not options.ignore_coordinate_breaks:
            raise NotImplementedError("反向追迹穿越坐标间断尚未实现，请启用 ignore_coordinate_breaks 选项或避免在反向追迹路径上使用坐标间断。")

        # 0.4 如果有数据导出，初始化数据结构
        intersections: list[SurfaceIntersection] = []
        recorded_surface_set = _resolve_recorded_surface_set(options.record_surface_states, surface_indices)
        recorded_steps = [] if recorded_surface_set is not None else None

        # 0.5 初始化opl
        opl = self._initial_opl(rays, options, device=ray_device)


        # 1. 计算追迹所需的 frame 数据
        # TODO: 可优化 ，不管是否开启ignore coordinate break，都计算一遍
        if options.ignore_coordinate_breaks:
            frame_data = compute_frame_data(system, device=ray_device, ignore_coordinate_breaks=True)
        else:
            frame_data = system.frame_data
            if frame_data is None or frame_data.device != ray_device:
                frame_data = compute_frame_data(system, device=ray_device)
        frame_rotation = frame_data.rotations[:, start_surface]
        frame_origin = frame_data.origins[:, start_surface]

        # 2. 逐面追迹
        for trace_index, surface_index in enumerate(surface_indices):
            # 2.1 记录追迹前的位置
            previous_position = ray.position
            previous_direction = ray.direction

            # 2.2 执行追迹
            step = _trace_surface_with_frame(
                system,
                surface_index,
                ray,
                material_data,
                wavelength_index,
                direction=direction,
                frame_rotation=frame_rotation,
                frame_origin=frame_origin,
                ignore_coordinate_breaks=options.ignore_coordinate_breaks,
                ignore_apertures=options.ignore_apertures,
            )
            ray = step.outgoing_ray

            # 2.3 记录opl
            if opl is not None:
                opl = _accumulate_optical_path(
                    material_data,
                    surface_index,
                    previous_position,
                    ray.position,
                    previous_direction,
                    wavelength_index,
                    opl,
                    direction=direction,
                )

            # 2.4 记录交点
            if options.record_intersections:
                intersections.append(_surface_intersection(step.hit))
            
            # 2.5 记录表面追迹历史
            if recorded_surface_set is not None and surface_index in recorded_surface_set:
                state_valid = _ray_state_valid(ray)
                state_opl = None if opl is None else torch.where(
                    state_valid, opl, torch.full_like(opl, torch.nan)
                )
                recorded_steps.append((surface_index, ray, state_opl, state_valid))

            # 2.6 坐标系更新
            if trace_index + 1 < len(surface_indices):
                next_surface_index = surface_indices[trace_index + 1]
                frame_rotation = frame_data.rotations[:, next_surface_index]
                frame_origin = frame_data.origins[:, next_surface_index]

        # 3. 取出追迹结果，构造输出结构体
        x, y, z = ray.position
        l, m, n = ray.direction
        valid = _ray_state_valid(ray)
        if opl is not None:
            opl = torch.where(valid, opl, torch.full_like(opl, torch.nan))

        final_rays = RayBundle(
            x=x,
            y=y,
            z=z,
            l=l,
            m=m,
            n=n,
            wavelength_index=rays.wavelength_index,
            intensity=rays.intensity,
            opl=opl if options.record_opd else rays.opl,
            metadata={**rays.metadata, "trace_mode": f"sequential_surface_{direction}"},
        )
        surface_history = None if recorded_steps is None else _stack_surface_history(recorded_steps)
        cache = {
            "mode": direction,
            "surface_count": len(system.surfaces),
            "system_count": system_count,
            "batch_shape": tuple(rays.x.shape),
        }
        return TraceResult(
            rays=final_rays,
            valid=valid,
            intersections=tuple(intersections),
            surface_history=surface_history,
            cache=cache,
        )

    def batched_trace(
        self,
        system: MultiOpticalSystem,
        *,
        sampler: PupilSampler | None = None,
        aimer: RayAimer | None = None,
        options: TraceOptions | None = None,
    ) -> TraceResult:
        """
        光线采样+批量光线追迹
        design x field x wavelength 维度执行采样追迹。
        """
        fields = list(system.fields)
        wavelengths = list(system.wavelengths)
        if len(fields) == 0:
            raise ValueError("At least one field is required for sampled tracing.")
        if len(wavelengths) == 0:
            raise ValueError("At least one wavelength is required for sampled tracing.")

        sampler = sampler or SquarePupilSampler(nx=3, ny=3)
        sample = sampler.sample()

        if aimer is None:
            first_order_data = system.first_order_data
            if first_order_data is None:
                raise ValueError("sampled tracing requires system.prepare() before using the default entrance pupil.")
            entrance_pupil = RayAimingResult(
                entrance_pupil_z=first_order_data.entrance_pupil_z,
                entrance_pupil_radius=first_order_data.entrance_pupil_radius,
            )
        else:
            entrance_pupil = aimer.aim(system, fields, wavelengths, sample)
        rays = build_input_rays_from_sample(
            system,
            fields,
            range(len(wavelengths)),
            sample,
            entrance_pupil,
        )
        return self.trace(system, rays, options=options)

    def _surface_bounds(
        self,
        system: MultiOpticalSystem,
        options: TraceOptions,
        direction: TraceDirection,
    ) -> tuple[int, int]:
        """解析不同追迹方向下的 surface 区间。"""
        surface_count = len(system.surfaces)
        if surface_count == 0:
            raise ValueError("At least one surface is required for tracing.")

        if direction == "forward":
            start_surface = int(options.start_surface)
            stop_surface = surface_count - 1 if options.stop_surface is None else int(options.stop_surface)
            if start_surface < 0 or stop_surface >= surface_count or start_surface > stop_surface:
                raise ValueError("Invalid forward tracing surface range.")
            return start_surface, stop_surface

        start_surface = int(options.start_surface)
        if options.stop_surface is None and start_surface == 0:
            start_surface = surface_count - 1
        stop_surface = 0 if options.stop_surface is None else int(options.stop_surface)
        if start_surface >= surface_count or stop_surface < 0 or start_surface < stop_surface:
            raise ValueError("Invalid backward tracing surface range.")
        return start_surface, stop_surface

    def _surface_indices(
        self,
        start_surface: int,
        stop_surface: int,
        direction: TraceDirection,
    ) -> range:
        """按方向生成逐面追迹索引。"""
        if direction == "forward":
            return range(start_surface, stop_surface + 1)
        return range(start_surface, stop_surface - 1, -1)

    def _validate_options(self, options: TraceOptions) -> None:
        """对当前真实 tracer 尚未兑现的选项显式失败。"""
        if options.record_ray_angles:
            raise NotImplementedError("SequentialSurfaceRayTracer does not implement record_ray_angles yet.")
        if options.warm_start is not None:
            raise NotImplementedError("SequentialSurfaceRayTracer does not implement warm_start yet.")

    def _initial_opl(
        self,
        rays: RayBundle,
        options: TraceOptions,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        """根据追迹选项初始化绝对光程。"""
        if not options.record_opd:
            return None
        if rays.opl is None:
            return torch.zeros_like(rays.x, dtype=torch.float64, device=device)
        return torch.as_tensor(rays.opl, dtype=torch.float64, device=device)


def _resolve_recorded_surface_set(
    selection: SurfaceStateSelection | None,
    traced_surface_indices: tuple[int, ...],
) -> set[int] | None:
    """解析追迹选项中指定的表面状态记录集合。"""
    if selection is None:
        return None
    selected = set(traced_surface_indices) if selection == "all" else set(selection)
    if not selected or not selected.issubset(traced_surface_indices):
        raise ValueError("record_surface_states must select surfaces in the current trace range.")
    return selected


def _ray_state_valid(ray: RayState) -> torch.Tensor:
    """计算光线位置和方向共同的有效掩码。"""
    components = (*ray.position, *ray.direction)
    valid = torch.isfinite(components[0])
    for component in components[1:]:
        valid &= torch.isfinite(component)
    return valid


def _surface_intersection(hit: SurfaceHit) -> SurfaceIntersection:
    """从内部交点生成公共几何快照。"""
    normal = None if hit.normal is None else tuple(component.clone() for component in hit.normal)
    return SurfaceIntersection(
        surface_index=hit.surface_index,
        position=tuple(component.clone() for component in hit.position),
        normal=normal,
    )


def _stack_surface_history(
    steps: list[tuple[int, RayState, torch.Tensor | None, torch.Tensor]],
) -> SurfaceTraceHistory:
    """将逐面追迹的光线状态堆叠为表面追迹历史。"""
    surface_indices, states, opl_values, valid = zip(*steps)
    positions = tuple(torch.stack([state.position[index] for state in states], dim=1) for index in range(3))
    directions = tuple(torch.stack([state.direction[index] for state in states], dim=1) for index in range(3))
    opl = None if opl_values[0] is None else torch.stack(opl_values, dim=1)
    return SurfaceTraceHistory(
        surface_indices=tuple(surface_indices),
        x=positions[0],
        y=positions[1],
        z=positions[2],
        l=directions[0],
        m=directions[1],
        n=directions[2],
        opl=opl,
        valid=torch.stack(valid, dim=1),
    )


def _accumulate_optical_path(
    material_data: BatchedMaterialData,
    surface_index: int,
    previous_position: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    current_position: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    previous_direction: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    wavelength_index: torch.Tensor,
    opl: torch.Tensor,
    *,
    direction: TraceDirection,
) -> torch.Tensor:
    """累计上一位置到当前交点之间的有符号绝对光程，单位 mm。"""
    previous_x, previous_y, previous_z = previous_position
    current_x, current_y, current_z = current_position
    direction_l, direction_m, direction_n = previous_direction
    dx = current_x - previous_x
    dy = current_y - previous_y
    dz = current_z - previous_z

    if direction == "forward":
        travel_l, travel_m, travel_n = direction_l, direction_m, direction_n
    else:
        travel_l, travel_m, travel_n = -direction_l, -direction_m, -direction_n
    direction_norm = torch.sqrt(travel_l * travel_l + travel_m * travel_m + travel_n * travel_n)
    safe_norm = torch.where(direction_norm > 0.0, direction_norm, torch.ones_like(direction_norm))
    signed_distance = (dx * travel_l + dy * travel_m + dz * travel_n) / safe_norm

    incident_material, _ = material_data.surface_indices(surface_index, direction=direction)
    refractive_index = material_data.refractive_index(incident_material, wavelength_index)
    segment_opl = refractive_index * signed_distance
    return torch.where(torch.isfinite(segment_opl), opl + segment_opl, torch.full_like(opl, torch.nan))
