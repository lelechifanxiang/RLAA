from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import torch

from .analysis import SpotDiagramResult, SpotDiagramSettings
from .rays import RayAimingResult, TraceOptions, TraceResult
from .sampling import (
    HexapolarPupilSampler,
    PupilSampler,
    SamplingResult,
    _append_reference_chief_ray,
)
from .surfaces import CoordinateBreak
from .tracing._sampled_rays import build_input_rays_from_sample

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plotting_config import configure_matplotlib_chinese

configure_matplotlib_chinese()

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


SPOT_WAVE_COLORS: tuple[str, ...] = (
    "#0000FF",
    "#00AA00",
    "#FF0000",
    "#FFD700",
    "#FF00FF",
    "#00C8C8",
)


def run_spot_diagram(
    system: MultiOpticalSystem,
    settings: SpotDiagramSettings,
) -> SpotDiagramResult:
    """执行 spot diagram 数值分析主流程。"""
    if system.tracer is None:
        raise ValueError("spot_diagram requires system.tracer.")

    sampler = build_spot_diagram_sampler(settings)
    sample = sampler.sample()
    trace_result = trace_spot_diagram_rays(system, sample)
    spot_data = extract_spot_data(system, trace_result, sample)
    rms_radius_um, geo_radius_um = compute_spot_metrics(system, spot_data)
    valid_count = spot_data["valid_points"].sum(dim=-1)
    valid_fraction = valid_count.to(dtype=torch.float64) / sample.sample_ray_count

    figure = None
    axes = None
    save_path = None
    scatter_points = None
    if settings.save_path is not None:
        figure, axes, save_path = plot_spot_diagram(
            system,
            spot_data,
            rms_radius_um=torch.as_tensor(rms_radius_um, dtype=torch.float64).detach().cpu(),
            geo_radius_um=torch.as_tensor(geo_radius_um, dtype=torch.float64).detach().cpu(),
            save_path=settings.save_path,
        )
        scatter_points = build_scatter_point_payload(system, spot_data)

    return SpotDiagramResult(
        rms_radius_um=rms_radius_um,
        geo_radius_um=geo_radius_um,
        valid_count=valid_count,
        valid_fraction=valid_fraction,
        field_points=tuple((float(field.x), float(field.y)) for field in system.fields),
        figure=figure,
        axes=axes,
        save_path=save_path,
        scatter_points=scatter_points,
    )


def build_spot_diagram_sampler(settings: SpotDiagramSettings) -> PupilSampler:
    """根据 settings 构造 spot diagram 采样器。"""
    pattern = settings.pattern
    ray_density = int(settings.ray_density)
    if ray_density < 3:
        raise ValueError("spot diagram ray_density must be greater than or equal to 3.")

    if pattern == "square":
        return SquareSpotDiagramPupilSampler(ray_density=ray_density)
    if pattern == "hexapolar":
        return HexapolarPupilSampler(rings=ray_density)
    raise ValueError(f"Unsupported spot diagram pattern: {pattern!r}.")


class SquareSpotDiagramPupilSampler(PupilSampler):
    """按 Zemax Standard Spot 的 square 语义生成裁圆网格。"""

    pattern = "square"

    def __init__(self, *, ray_density: int) -> None:
        self.ray_density = int(ray_density)

    def sample(self) -> SamplingResult:
        grid_count = 2 * self.ray_density + 1
        axis = torch.linspace(-1.0, 1.0, grid_count, dtype=torch.float64)
        px_grid, py_grid = torch.meshgrid(axis, axis, indexing="ij")
        coordinates = torch.stack((px_grid.reshape(-1), py_grid.reshape(-1)), dim=-1)

        # Zemax Standard Spot 的 square 语义等价于先生成规则方格，再裁圆保留单位 pupil 内部点。
        inside_pupil = coordinates[:, 0] * coordinates[:, 0] + coordinates[:, 1] * coordinates[:, 1] <= 1.0 + 1e-12
        coordinates = coordinates[inside_pupil]

        ray_count = coordinates.shape[0]
        return _append_reference_chief_ray(
            coordinates,
            torch.full((ray_count,), 1.0 / ray_count, dtype=torch.float64),
            pattern=self.pattern,
        )


def trace_spot_diagram_rays(
    system: MultiOpticalSystem,
    sample: SamplingResult,
) -> TraceResult:
    """组装并追迹 spot diagram 所需的批量光线。"""
    if system.tracer is None:
        raise ValueError("spot_diagram requires system.tracer.")

    fields = list(system.fields)
    wavelengths = list(system.wavelengths)
    if len(fields) == 0:
        raise ValueError("spot diagram requires at least one field.")
    if len(wavelengths) == 0:
        raise ValueError("spot diagram requires at least one wavelength.")
    if sample.pupil_coordinates is None:
        raise ValueError("spot diagram sampler must provide pupil coordinates.")
    if sample.chief_ray_index is None:
        raise ValueError("spot diagram sampler must provide chief_ray_index.")

    first_order_data = system.first_order_data
    if first_order_data is None:
        if any(isinstance(surface, CoordinateBreak) for surface in system.surfaces):
            raise NotImplementedError("Coordinate Break spot diagram 仍需单独对齐入瞳瞄准。")
        raise ValueError("spot_diagram requires system.prepare() before tracing spot rays.")

    rays = build_input_rays_from_sample(
        system,
        fields,
        range(len(wavelengths)),
        sample,
        RayAimingResult(
            entrance_pupil_z=first_order_data.entrance_pupil_z,
            entrance_pupil_radius=first_order_data.entrance_pupil_radius,
        ),
    )
    return system.tracer.trace(
        system,
        rays,
        options=TraceOptions(record_intersections=False),
    )


def extract_spot_data(
    system: MultiOpticalSystem,
    trace_result: TraceResult,
    sample: SamplingResult,
) -> dict[str, Any]:
    """提取与 chief ray 对齐后的 spot 散点和有效掩码。"""
    x = torch.as_tensor(trace_result.rays.x, dtype=torch.float64)
    y = torch.as_tensor(trace_result.rays.y, dtype=torch.float64, device=x.device)
    valid = torch.as_tensor(trace_result.valid, dtype=torch.bool, device=x.device)

    if x.ndim != 4 or y.ndim != 4 or valid.ndim != 4:
        raise ValueError("spot diagram trace result must have shape (design, field, wavelength, ray).")

    chief_ray_index = sample.chief_ray_index

    primary_wavelength_index = system.wavelengths.primary_index
    chief_valid = valid[:, :, primary_wavelength_index, chief_ray_index]
    chief_x_mm = x[:, :, primary_wavelength_index, chief_ray_index]
    chief_y_mm = y[:, :, primary_wavelength_index, chief_ray_index]
    chief_x = chief_x_mm.unsqueeze(-1).unsqueeze(-1)
    chief_y = chief_y_mm.unsqueeze(-1).unsqueeze(-1)
    dx = x - chief_x
    dy = y - chief_y
    valid_points = valid & chief_valid.unsqueeze(-1).unsqueeze(-1)
    valid_points[..., int(sample.sample_ray_count):] = False
    return {
        "dx_mm": dx,
        "dy_mm": dy,
        "chief_x_mm": chief_x_mm,
        "chief_y_mm": chief_y_mm,
        "valid_points": valid_points,
    }


def compute_spot_metrics(
    system: MultiOpticalSystem,
    spot_data: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Zemax Standard Spot 语义统计 RMS radius 与 GEO radius。"""
    dx = torch.as_tensor(spot_data["dx_mm"], dtype=torch.float64)
    dy = torch.as_tensor(spot_data["dy_mm"], dtype=torch.float64, device=dx.device)
    valid_points = torch.as_tensor(spot_data["valid_points"], dtype=torch.bool, device=dx.device)

    radius_squared_mm = dx * dx + dy * dy
    radius_mm = torch.sqrt(radius_squared_mm)
    rms_radius_um = _compute_rms_radius_um(
        system,
        radius_squared_mm=radius_squared_mm,
        valid_points=valid_points,
    )
    geo_radius_um = _compute_geo_radius_um(
        radius_mm=radius_mm,
        valid_points=valid_points,
    )
    return rms_radius_um, geo_radius_um


def plot_spot_diagram(
    system: MultiOpticalSystem,
    spot_data: dict[str, Any],
    *,
    rms_radius_um: torch.Tensor,
    geo_radius_um: torch.Tensor,
    save_path: str | Path,
):
    """绘制并导出 spot diagram 图片。"""
    if system.system_count != 1:
        raise ValueError("spot diagram image export currently only supports a single design_view.")

    dx = torch.as_tensor(spot_data["dx_mm"], dtype=torch.float64)
    dy = torch.as_tensor(spot_data["dy_mm"], dtype=torch.float64, device=dx.device)
    chief_x_mm = torch.as_tensor(spot_data["chief_x_mm"], dtype=torch.float64).detach().cpu()
    chief_y_mm = torch.as_tensor(spot_data["chief_y_mm"], dtype=torch.float64).detach().cpu()
    valid_points = torch.as_tensor(spot_data["valid_points"], dtype=torch.bool, device=dx.device)
    field_count = dx.shape[1]
    wavelength_count = dx.shape[2]

    finite_radius = torch.sqrt(dx * dx + dy * dy)
    masked_radius = torch.where(valid_points, finite_radius, torch.zeros_like(finite_radius))
    max_radius_mm = float(torch.max(masked_radius).item())
    axis_limit_mm = max(max_radius_mm * 1.1, 1e-6)

    figure, axes = plt.subplots(1, field_count, figsize=(5.2 * field_count, 5.6), dpi=150, squeeze=False)
    axes_row = axes[0]
    for field_index, axis in enumerate(axes_row):
        for wavelength_index in range(wavelength_count):
            valid_now = valid_points[0, field_index, wavelength_index]
            if not torch.any(valid_now).item():
                continue
            draw_x = dx[0, field_index, wavelength_index, valid_now].detach().cpu().numpy()
            draw_y = dy[0, field_index, wavelength_index, valid_now].detach().cpu().numpy()
            axis.scatter(
                draw_x,
                draw_y,
                s=4.0,
                c=_spot_wave_color(wavelength_index),
                linewidths=0.1,
                alpha=0.9,
                label=_spot_wave_label(system, wavelength_index) if field_index == 0 else None,
            )

        field = system.fields[field_index]
        axis.set_title(
            "OBJ: {:.2f}, {:.2f} (deg)\nIMA: {:.3f}, {:.3f} mm\nRMS: {:.3f} um\nGEO: {:.3f} um".format(
                float(field.x),
                float(field.y),
                float(chief_x_mm[0, field_index].item()),
                float(chief_y_mm[0, field_index].item()),
                float(rms_radius_um[0, field_index].item()),
                float(geo_radius_um[0, field_index].item()),
            )
        )
        axis.set_xlim(-axis_limit_mm, axis_limit_mm)
        axis.set_ylim(-axis_limit_mm, axis_limit_mm)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linewidth=0.3, alpha=0.35)
        axis.set_xlabel("x (mm)")
        if field_index == 0:
            axis.set_ylabel("y (mm)")
            axis.legend(loc="upper right", fontsize=8)

    # figure.suptitle(system.name or "spot_diagram")
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    return figure, axes_row, str(output_path)


def build_scatter_point_payload(system: MultiOpticalSystem, spot_data: dict[str, Any]) -> dict[str, Any]:
    """导出按视场、波长组织的原始散点。"""
    if system.system_count != 1:
        raise ValueError("spot diagram scatter export currently only supports a single design_view.")

    dx = torch.as_tensor(spot_data["dx_mm"], dtype=torch.float64)
    dy = torch.as_tensor(spot_data["dy_mm"], dtype=torch.float64, device=dx.device)
    valid_points = torch.as_tensor(spot_data["valid_points"], dtype=torch.bool, device=dx.device)

    payload: dict[str, Any] = {}
    for field_index, field in enumerate(system.fields):
        wave_payload: dict[str, Any] = {}
        for wavelength_index, wavelength in enumerate(system.wavelengths):
            valid_now = valid_points[0, field_index, wavelength_index]
            wave_payload[f"wave_{wavelength_index}"] = {
                "wavelength_um": float(wavelength.value_um),
                "color": _spot_wave_color(wavelength_index),
                "x_mm": dx[0, field_index, wavelength_index, valid_now].detach().cpu().tolist(),
                "y_mm": dy[0, field_index, wavelength_index, valid_now].detach().cpu().tolist(),
            }
        payload[f"field_{field_index}"] = {
            "field_point": (float(field.x), float(field.y)),
            "waves": wave_payload,
        }
    return payload


def _spot_wave_color(wavelength_index: int) -> str:
    return SPOT_WAVE_COLORS[wavelength_index % len(SPOT_WAVE_COLORS)]


def _spot_wave_label(system: MultiOpticalSystem, wavelength_index: int) -> str:
    wavelength = system.wavelengths[wavelength_index]
    return f"{float(wavelength.value_um):.4f} um"


def _compute_rms_radius_um(
    system: MultiOpticalSystem,
    *,
    radius_squared_mm: torch.Tensor,
    valid_points: torch.Tensor,
) -> torch.Tensor:
    wavelength_weights = system._material_data.wavelength_weights.reshape(1, 1, -1)
    valid_counts = valid_points.sum(dim=-1)
    squared_sum = torch.where(valid_points, radius_squared_mm, torch.zeros_like(radius_squared_mm)).sum(dim=-1)
    mean_squared_by_wave = squared_sum / valid_counts.clamp_min(1).to(dtype=torch.float64)
    effective_weights = wavelength_weights * (valid_counts > 0).to(dtype=torch.float64)
    weighted_sum = (mean_squared_by_wave * effective_weights).sum(dim=-1)
    total_weight = effective_weights.sum(dim=-1)

    rms_radius_mm = torch.full_like(weighted_sum, torch.nan)
    valid_metric = total_weight > 0
    rms_radius_mm = torch.where(
        valid_metric,
        torch.sqrt(weighted_sum / total_weight.clamp_min(torch.finfo(torch.float64).eps)),
        rms_radius_mm,
    )
    return rms_radius_mm * 1000.0


def _compute_geo_radius_um(
    *,
    radius_mm: torch.Tensor,
    valid_points: torch.Tensor,
) -> torch.Tensor:
    masked_radius = torch.where(valid_points, radius_mm, torch.full_like(radius_mm, float("-inf")))
    geo_radius_mm = torch.amax(masked_radius, dim=(-2, -1))
    has_valid_point = torch.any(valid_points, dim=(-2, -1))
    geo_radius_mm = torch.where(has_valid_point, geo_radius_mm, torch.full_like(geo_radius_mm, torch.nan))
    return geo_radius_mm * 1000.0
