from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import torch

from .apertures import calculate_clear_apertures
from .materials import AIR, Material
from .rays import RayAimingResult, TraceOptions, TraceResult
from .sampling import ExplicitPupilSampler, PupilSampler
from .surfaces import CoordinateBreak, EvenAsphereSurface, ImageSurface, ObjectSurface, ParaxialSurface, SphereSurface
from .tracing._sampled_rays import build_input_rays_from_sample

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plotting_config import configure_matplotlib_chinese

configure_matplotlib_chinese()

if TYPE_CHECKING:
    from .analysis import Layout2DResult, Layout2DSettings
    from .system import MultiOpticalSystem
    from .system_specs import FieldPoint


LAYOUT_TRACE_PUPIL_COORDINATES: tuple[tuple[float, float], ...] = (
    (0.0, -1.0),
    (0.0, -2.0 / 3.0),
    (0.0, -1.0 / 3.0),
    (0.0, 0.0),
    (0.0, 1.0 / 3.0),
    (0.0, 2.0 / 3.0),
    (0.0, 1.0),
)
LAYOUT_CLEAR_APERTURE_PUPIL_COORDINATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)
LAYOUT_FIELD_COLORS: tuple[str, ...] = (
    "#0000FF",
    "#00AA00",
    "#FF0000",
    "#FFD700",
    "#FF00FF",
    "#00C8C8",
)
SURFACE_PROFILE_SAMPLE_COUNT = 257


def filter_layout_fields(system: MultiOpticalSystem) -> tuple[list[int], list[FieldPoint], str | None]:
    """筛选 yz layout 可显示的 x=0 视场。"""
    filtered_indices: list[int] = []
    filtered_fields: list[FieldPoint] = []
    removed_count = 0
    for field_index, field in enumerate(system.fields):
        if abs(float(field.x)) <= 1e-12:
            filtered_indices.append(field_index)
            filtered_fields.append(field)
        else:
            removed_count += 1

    if not filtered_fields:
        raise ValueError("2D layout 仅支持 x=0 的视场，但当前系统中不存在满足条件的视场。")
    if removed_count == 0:
        return filtered_indices, filtered_fields, None
    return filtered_indices, filtered_fields, f"已过滤 {removed_count} 个 x!=0 的视场，仅显示 yz 平面可见视场。"


def build_layout_sampler() -> PupilSampler:
    """构造 yz 截面的固定 7 光线采样器。"""
    return ExplicitPupilSampler(
        pupil_coordinates=torch.tensor(LAYOUT_TRACE_PUPIL_COORDINATES, dtype=torch.float64),
    )


def _build_clear_aperture_sampler() -> PupilSampler:
    return ExplicitPupilSampler(
        pupil_coordinates=torch.tensor(LAYOUT_CLEAR_APERTURE_PUPIL_COORDINATES, dtype=torch.float64),
    )


def _resolve_layout_entrance_pupil(
    system: MultiOpticalSystem
) -> RayAimingResult:
    """复用 first_order_data"""
    first_order_data = system.first_order_data
    if (
        first_order_data is not None
        and getattr(first_order_data, "entrance_pupil_z", None) is not None
        and getattr(first_order_data, "entrance_pupil_radius", None) is not None
    ):
        return RayAimingResult(
            entrance_pupil_z=first_order_data.entrance_pupil_z,
            entrance_pupil_radius=first_order_data.entrance_pupil_radius,
            cache={"source": "system.first_order_data"},
        )

    raise NotImplementedError("当前系统缺乏一阶数据，请先执行 first_order 分析以获得入瞳信息。")


def trace_layout_rays(
    system: MultiOpticalSystem,
    fields: list[FieldPoint],
) -> TraceResult:
    """对筛选后的视场执行主波长 layout 光线追迹。"""
    if system.tracer is None:
        raise ValueError("layout_2d requires system.tracer.")

    sampler = build_layout_sampler()
    sample = sampler.sample()
    entrance_pupil = _resolve_layout_entrance_pupil(system)
    rays = build_input_rays_from_sample(
        system,
        fields,
        (system.wavelengths.primary_index,),
        sample,
        entrance_pupil,
    )
    return system.tracer.trace(
        system,
        rays,
        options=TraceOptions(record_intersections=True),
    )


def _surface_draw_radius(
    system: MultiOpticalSystem,
    surface_index: int,
    clear_aperture_by_surface: dict[int, float],
    trace_result: TraceResult,
) -> float:
    semi_diameter = system.surfaces[surface_index].semi_diameter
    if semi_diameter is not None:
        radius = float(torch.as_tensor(semi_diameter, dtype=torch.float64).reshape(-1)[0].item())
        if radius > 0.0:
            return radius

    clear_aperture_radius = clear_aperture_by_surface.get(surface_index)
    if clear_aperture_radius is not None and clear_aperture_radius > 0.0:
        return clear_aperture_radius

    for hit in trace_result.intersections:
        if hit.surface_index != surface_index:
            continue
        y = torch.as_tensor(hit.position[1], dtype=torch.float64)
        finite_y = torch.where(torch.isfinite(y), torch.abs(y), torch.zeros_like(y))
        radius = float(torch.max(finite_y).item())
        if radius > 0.0:
            return radius
    return 0.0


def sample_surface_profile(
    system: MultiOpticalSystem,
    surface_index: int,
    semi_diameter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """采样单个表面的 yz 轮廓。"""
    surface = system.surfaces[surface_index]
    if isinstance(surface, CoordinateBreak):
        raise NotImplementedError("2D layout 暂不支持 CoordinateBreak。")
    if isinstance(surface, EvenAsphereSurface) and surface.geometry.coefficients:
        raise NotImplementedError("2D layout 暂不支持带高次系数的非球面绘制。")
    if not isinstance(surface, (ObjectSurface, ImageSurface, ParaxialSurface, SphereSurface, EvenAsphereSurface)):
        raise NotImplementedError(f"2D layout 暂不支持表面类型 {type(surface).__name__!r}。")

    z0 = system.frame_data.surface_z(surface_index).reshape(-1)
    if z0.numel() != 1:
        raise ValueError("2D layout 仅支持单个 design_view。")
    y = torch.linspace(
        -semi_diameter,
        semi_diameter,
        SURFACE_PROFILE_SAMPLE_COUNT,
        dtype=torch.float64,
        device=z0.device
    )
    x = torch.zeros_like(y)
    sag = surface.geometry.sag(x, y)
    z = sag + z0[0]
    return z.detach().cpu(), y.detach().cpu()


def _field_color(field_index: int) -> str:
    return LAYOUT_FIELD_COLORS[field_index % len(LAYOUT_FIELD_COLORS)]


def _field_legend_label(field: FieldPoint) -> str:
    return f"field x={float(field.x):.1f}, y={float(field.y):.1f}"


def _collect_layout_ray_path(
    trace_result: TraceResult,
    *,
    field_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_segments: list[torch.Tensor] = []
    y_segments: list[torch.Tensor] = []
    for hit in trace_result.intersections:
        z_segments.append(
            torch.as_tensor(hit.position[2][0, field_index, 0], dtype=torch.float64).detach().cpu().reshape(1, -1)
        )
        y_segments.append(
            torch.as_tensor(hit.position[1][0, field_index, 0], dtype=torch.float64).detach().cpu().reshape(1, -1)
        )
    if not z_segments or not y_segments:
        raise ValueError("layout_2d requires recorded surface intersections.")
    return torch.cat(z_segments, dim=0), torch.cat(y_segments, dim=0)


def _is_optical_medium(medium: Material | None) -> bool:
    return medium is not None and medium.name != AIR.name


def _connected_element_surface_groups(
    system: MultiOpticalSystem,
) -> list[list[int]]:
    groups: list[list[int]] = []
    current_group: list[int] = []
    inside_element = False

    for surface_index, surface in enumerate(system.surfaces):
        if not inside_element and _is_optical_medium(surface.gap.medium):
            current_group = [surface_index]
            inside_element = True
            continue

        if not inside_element:
            continue

        current_group.append(surface_index)
        if not _is_optical_medium(surface.gap.medium):
            groups.append(current_group)
            current_group = []
            inside_element = False

    return groups


def _surface_edge_z(
    system: MultiOpticalSystem,
    surface_index: int,
    semi_diameter: float,
) -> float:
    surface = system.surfaces[surface_index]
    z0 = system.frame_data.surface_z(surface_index).reshape(-1)
    y = torch.tensor([semi_diameter], dtype=torch.float64, device=z0.device)
    x = torch.zeros_like(y)
    sag = surface.geometry.sag(x, y)
    if z0.numel() != 1:
        raise ValueError("2D layout 仅支持单个 design_view。")
    return float((sag + z0[0])[0].item())


def _draw_element_edges(
    axes,
    system: MultiOpticalSystem,
    surface_draw_radius: dict[int, float],
) -> None:
    for surface_group in _connected_element_surface_groups(system):
        radii = [surface_draw_radius.get(surface_index, 0.0) for surface_index in surface_group]
        if not radii or max(radii) <= 0.0:
            continue

        edge_radius = max(radii)
        edge_z_by_surface = {
            surface_index: _surface_edge_z(system, surface_index, radius)
            for surface_index, radius in zip(surface_group, radii)
            if radius > 0.0
        }
        if len(edge_z_by_surface) < 2:
            continue

        for surface_index, radius in zip(surface_group, radii):
            if radius <= 0.0 or radius >= edge_radius:
                continue
            edge_z = edge_z_by_surface[surface_index]
            axes.plot([edge_z, edge_z], [radius, edge_radius], color="black", linewidth=1.2)
            axes.plot([edge_z, edge_z], [-edge_radius, -radius], color="black", linewidth=1.2)

        edge_z_values = list(edge_z_by_surface.values())
        axes.plot([min(edge_z_values), max(edge_z_values)], [edge_radius, edge_radius], color="black", linewidth=1.2)
        axes.plot([min(edge_z_values), max(edge_z_values)], [-edge_radius, -edge_radius], color="black", linewidth=1.2)


def plot_layout_2d(
    system: MultiOpticalSystem,
    trace_result: TraceResult,
    clear_aperture_result: Any,
    filtered_fields: list[FieldPoint],
    *,
    save_path: str | Path | None = None,
):
    """绘制 yz 平面的 layout 图。"""
    figure, axes = plt.subplots(figsize=(10.0, 5.5), dpi=150)
    clear_aperture_by_surface = {
        surface_index: float(clear_aperture_result.semi_diameter[0, aperture_index].item())
        for aperture_index, surface_index in enumerate(clear_aperture_result.surface_indices)
    }
    surface_draw_radius: dict[int, float] = {}

    # 绘制表面轮廓
    for surface_index, _surface in enumerate(system.surfaces):
        semi_diameter = _surface_draw_radius(
            system,
            surface_index,
            clear_aperture_by_surface,
            trace_result,
        )
        if semi_diameter <= 0.0:
            continue
        surface_draw_radius[surface_index] = semi_diameter
        z_profile, y_profile = sample_surface_profile(system, surface_index, semi_diameter)
        axes.plot(z_profile.tolist(), y_profile.tolist(), color="black", linewidth=1.2)

    _draw_element_edges(axes, system, surface_draw_radius)

    # 绘制各视场的 7 根光线，同一视场保持同色
    for field_index, field in enumerate(filtered_fields):
        field_color = _field_color(field_index)
        field_label = _field_legend_label(field)
        z_path, y_path = _collect_layout_ray_path(
            trace_result,
            field_index=field_index,
        )
        for ray_index in range(z_path.shape[1]):
            axes.plot(
                z_path[:, ray_index].tolist(),
                y_path[:, ray_index].tolist(),
                color=field_color,
                linewidth=1.0,
                label=field_label if ray_index == 0 else None,
            )

    axes.set_xlabel("z (mm)")
    axes.set_ylabel("y (mm)")
    axes.set_title(system.name or "layout_2d")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, linewidth=0.3, alpha=0.35)
    if filtered_fields:
        axes.legend()
    figure.tight_layout()

    resolved_save_path: str | None = None
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        resolved_save_path = str(output_path)
    return figure, axes, resolved_save_path


def run_layout_2d(system: MultiOpticalSystem, settings: Layout2DSettings) -> Layout2DResult:
    """执行单个 design_view 的 yz layout 分析。"""
    if system.system_count != 1:
        raise ValueError("2D layout 当前只支持单个 design_view，请先调用 system.design_view(i)。")

    # 筛选出 x=0 的视场进行追迹
    filtered_field_indices, filtered_fields, message = filter_layout_fields(system)

    # 复用准备态净口径，缺失时再即时计算。
    clear_aperture_result = system.clear_aperture_data
    if clear_aperture_result is None:
        clear_aperture_result = calculate_clear_apertures(
            system,
            sampler=_build_clear_aperture_sampler(),
            keep_trace_result=False,
        )

    # 执行7光束追迹，用于layout绘制
    trace_result = trace_layout_rays(system, filtered_fields)

    # 绘图
    figure, axes, save_path = plot_layout_2d(
        system,
        trace_result,
        clear_aperture_result,
        filtered_fields,
        save_path=settings.save_path,
    )

    from .analysis import Layout2DResult

    return Layout2DResult(
        filtered_field_indices=tuple(filtered_field_indices),
        filtered_field_points=tuple((float(field.x), float(field.y)) for field in filtered_fields),
        trace_result=trace_result,
        clear_aperture_result=clear_aperture_result,
        figure=figure,
        axes=axes,
        save_path=save_path,
        message=message,
    )
