from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ._first_order_probes import (
    ExitPupilData,
    FrontParaxialData,
    build_exit_pupil_probe_rays,
    build_front_paraxial_probe_rays,
    build_marginal_probe_ray,
    solve_exit_pupil_from_probes,
    solve_front_paraxial_data,
    solve_working_f_number_from_marginal_ray,
)
from ._parameter_access import surface_value
from ._runtime import default_device
from .rays import TraceOptions
from .surfaces import EvenAsphereSurface, ParaxialSurface, SphereSurface
from .system_state import FirstOrderData, FrameData

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


@dataclass(slots=True)
class ResolvedAperture:
    """由系统孔径定义解析得到的实际半径。"""

    entrance_pupil_radius: torch.Tensor
    stop_radius: torch.Tensor


def _fixed_stop_radius(system: MultiOpticalSystem, *, device: torch.device) -> torch.Tensor:
    stop_index = system.surfaces.stop_index
    if stop_index is None:
        raise ValueError("A stop surface is required for first-order calculations.")
    surface = system.surfaces[stop_index]
    if surface.semi_diameter is None:
        raise ValueError("float_by_stop_size requires a fixed stop semi-diameter.")
    return surface_value(
        system,
        stop_index,
        "semi_diameter",
        surface.semi_diameter,
        batch_ndim=0,
        device=device,
    ).reshape(system.system_count)


def resolve_system_aperture(
    system: MultiOpticalSystem,
    *,
    effl: torch.Tensor,
    front_pupil_magnification: torch.Tensor,
) -> ResolvedAperture:
    """将系统孔径定义统一换算为入瞳半径和 stop 半径。"""
    if system.aperture is None:
        raise ValueError("A system aperture is required for first-order calculations.")

    aperture_kind = system.aperture.kind
    aperture_value = torch.as_tensor(system.aperture.value, dtype=torch.float64, device=effl.device)
    aperture_value = aperture_value.expand_as(effl)
    if aperture_kind == "entrance_pupil_diameter":
        entrance_pupil_radius = aperture_value / 2.0
        stop_radius = entrance_pupil_radius / front_pupil_magnification.abs().clamp_min(1e-12)
    elif aperture_kind == "image_f_number":
        if torch.any(aperture_value <= 0.0):
            raise ValueError("image_f_number must be greater than zero.")
        entrance_pupil_radius = effl.abs() / (2.0 * aperture_value)
        stop_radius = entrance_pupil_radius / front_pupil_magnification.abs().clamp_min(1e-12)
    elif aperture_kind == "float_by_stop_size":
        stop_radius = _fixed_stop_radius(system, device=effl.device)
        entrance_pupil_radius = front_pupil_magnification.abs() * stop_radius
    else:
        raise NotImplementedError(f"Unsupported aperture kind {aperture_kind!r}.")

    return ResolvedAperture(
        entrance_pupil_radius=entrance_pupil_radius,
        stop_radius=stop_radius,
    )


# ------------------------------------ 一阶参数计算 ------------------------------------


class _FirstOrderCalculator:
    """一阶参数计算器，集中管理共享上下文和计算流程。"""

    def __init__(
        self,
        system: MultiOpticalSystem,
        *,
        device: torch.device | None = None,
        frame_data: FrameData | None = None,
    ) -> None:
        if system.tracer is None:
            raise ValueError("A tracer is required for first-order calculations.")

        self.system = system
        self.device = default_device(system) if device is None else device
        self.frame_data = frame_data if frame_data is not None else self._compute_first_order_frame_data()
        stop_index = system.surfaces.stop_index
        if stop_index is None:
            raise ValueError("A stop surface is required for first-order calculations.")
        self.stop_index = stop_index
        self.stop_position = self.frame_data.surface_z(stop_index).to(device=self.device)
        self.last_refractive_surface_index = max(
            index
            for index, surface in enumerate(system.surfaces)
            if isinstance(surface, (SphereSurface, EvenAsphereSurface, ParaxialSurface))
        )
        self.first_surface_z = self.frame_data.surface_z(0).to(device=self.device)
        self.last_surface_z = self.frame_data.surface_z(self.last_refractive_surface_index).to(device=self.device)
        self.image_surface_z = self.frame_data.surface_z(len(system.surfaces) - 1).to(device=self.device)

    def run(self) -> FirstOrderData:
        """计算并返回当前系统的一阶数据。"""
        ttl = self.total_track_length()
        image_plane_distance = self.image_surface_z - self.last_surface_z
        probe_height = self._probe_height()
        front = self._solve_front_paraxial(probe_height)
        aperture = resolve_system_aperture(
            self.system,
            effl=front.effl,
            front_pupil_magnification=front.front_pupil_magnification,
        )
        working_f_number = self._trace_working_f_number(
            front,
            aperture,
        )
        exit_pupil = self._solve_exit_pupil(probe_height)
        exit_pupil_radius = exit_pupil.rear_pupil_magnification.abs() * aperture.stop_radius

        values = torch.stack(
            (ttl, front.effl, working_f_number, image_plane_distance, front.bfl),
            dim=-1,
        )
        valid = torch.isfinite(values).all(dim=-1)
        ttl, effl, working_f_number, image_plane_distance, bfl = (
            value.masked_fill(~valid, torch.nan)
            for value in (ttl, front.effl, working_f_number, image_plane_distance, front.bfl)
        )

        return FirstOrderData(
            effl=effl,
            working_f_number=working_f_number,
            ttl=ttl,
            image_plane_distance=image_plane_distance,
            bfl=bfl,
            valid=valid,
            entrance_pupil_z=front.entrance_pupil_z,
            entrance_pupil_radius=aperture.entrance_pupil_radius,
            stop_radius=aperture.stop_radius,
            exit_pupil_z=exit_pupil.exit_pupil_z,
            exit_pupil_radius=exit_pupil_radius,
        )

    def _solve_front_paraxial(self, probe_height: torch.Tensor) -> FrontParaxialData:
        """追迹 4 根前组近轴光线并求解一阶几何量。"""
        probe_rays = build_front_paraxial_probe_rays(self.system, probe_height)
        result = self.system.tracer.trace(
            self.system,
            probe_rays,
            options=TraceOptions(
                stop_surface=max(self.stop_index, self.last_refractive_surface_index),
                record_intersections=True,
                record_opd=False,
                ignore_coordinate_breaks=True,
                ignore_apertures=True,
            ),
        )
        return solve_front_paraxial_data(
            result,
            stop_index=self.stop_index,
            probe_height=probe_height,
            last_surface_z=self.last_surface_z,
        )

    def _trace_working_f_number(
        self,
        front: FrontParaxialData,
        aperture: ResolvedAperture,
    ) -> torch.Tensor:
        """追迹一根真实边缘光线并求工作 F/#。"""
        probe_ray = build_marginal_probe_ray(
            self.system,
            entrance_pupil_z=front.entrance_pupil_z,
            entrance_pupil_radius=aperture.entrance_pupil_radius,
        )
        # Zemax System Data 的一阶边缘光线不受处方固定半口径裁剪。
        result = self.system.tracer.trace(
            self.system,
            probe_ray,
            options=TraceOptions(
                record_intersections=False,
                record_opd=False,
                ignore_coordinate_breaks=True,
                ignore_apertures=True,
            ),
        )
        return solve_working_f_number_from_marginal_ray(result)

    def total_track_length(self) -> torch.Tensor:
        """读取忽略坐标断裂后的一阶总长。"""
        return self.image_surface_z - self.first_surface_z

    def _solve_exit_pupil(
        self,
        probe_height: torch.Tensor,
    ) -> ExitPupilData:
        """计算出瞳位置和后组近轴放大率。"""
        if self.stop_index >= len(self.system.surfaces) - 1:
            return ExitPupilData(
                exit_pupil_z=self.stop_position,
                rear_pupil_magnification=torch.ones(
                    (self.system.system_count,),
                    dtype=torch.float64,
                    device=self.device,
                ),
            )

        # 从 stop 向像方正向追迹两根近轴探测光线。
        probe_rays = build_exit_pupil_probe_rays(self.system, self.stop_position, probe_height)
        result = self.system.tracer.trace(
            self.system,
            probe_rays,
            options=TraceOptions(
                start_surface=self.stop_index,
                stop_surface=len(self.system.surfaces) - 1,
                direction="forward",
                record_intersections=False,
                record_opd=False,
                ignore_coordinate_breaks=True,
                ignore_apertures=True,
            ),
        )
        return solve_exit_pupil_from_probes(
            result,
            stop_position=self.stop_position,
            probe_height=probe_height,
        )

    def _probe_height(self) -> torch.Tensor:
        """生成统一的近轴探测高度。"""
        return torch.full(
            (self.system.system_count,),
            1e-6,
            dtype=torch.float64,
            device=self.device,
        )

    def _compute_first_order_frame_data(self) -> FrameData:
        """计算一阶参数专用 frame，按 Zemax 习惯忽略坐标断裂。"""
        from .frames import compute_frame_data

        return compute_frame_data(self.system, device=self.device, ignore_coordinate_breaks=True)


def compute_first_order_data(
    system: MultiOpticalSystem,
    *,
    device: torch.device | None = None,
) -> FirstOrderData:
    """计算当前系统的一阶数据。"""
    return _FirstOrderCalculator(system, device=device).run()
