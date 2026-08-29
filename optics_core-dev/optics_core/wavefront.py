from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import matplotlib
import torch

from .analysis import WavefrontResult, WavefrontSettings
from .rays import RayAimingResult, TraceOptions, TraceResult
from .sampling import SamplingResult
from .tracing._sampled_rays import build_input_rays_from_sample

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .system import MultiOpticalSystem
    from .system_specs import FieldPoint


@dataclass(slots=True)
class WavefrontBatch:
    """单个 design minibatch 的 Wavefront Map 结果。"""

    opd: torch.Tensor
    rms_wavefront: torch.Tensor
    valid_mask: torch.Tensor
    valid_count: torch.Tensor
    valid_fraction: torch.Tensor
    pupil_x: torch.Tensor
    pupil_y: torch.Tensor
    field_indices: tuple[int, ...]
    wavelength_indices: tuple[int, ...]
    sample_count: int


@dataclass(slots=True)
class ImageWaveData:
    """PSF 与 Wavefront Map 共用的像面光线波前数据。"""

    image_points: torch.Tensor
    ray_directions: torch.Tensor
    opl: torch.Tensor
    wavelength_mm: torch.Tensor
    chief_points: torch.Tensor
    valid_points: torch.Tensor
    pupil_coordinates: torch.Tensor
    pupil_weights: torch.Tensor
    ordinary_ray_count: int
    chief_ray_index: int


class WavefrontDesignBatchIterator(Iterator[tuple[int, int, WavefrontBatch]]):
    """按 design minibatch 计算 Wavefront Map。"""

    def __init__(
        self,
        system: MultiOpticalSystem,
        *,
        field_indices: tuple[int, ...],
        wavelength_indices: tuple[int, ...],
        sample_count: int,
    ) -> None:
        self.system = system
        self.field_indices = field_indices
        self.wavelength_indices = wavelength_indices
        self.sample_count = int(sample_count)
        self.design_batch_size = system.system_count
        self.initial_design_batch_size = self.design_batch_size
        self.minibatch_count = 0
        self._start = 0

    def __iter__(self) -> WavefrontDesignBatchIterator:
        return self

    def __next__(self) -> tuple[int, int, WavefrontBatch]:
        if self._start >= self.system.system_count:
            raise StopIteration

        while True:
            stop = min(self._start + self.design_batch_size, self.system.system_count)
            batch_system = self.system.design_batch_view(self._start, stop)
            retry_after_oom = False
            try:
                batch = compute_wavefront_batch(
                    batch_system,
                    field_indices=self.field_indices,
                    wavelength_indices=self.wavelength_indices,
                    sample_count=self.sample_count,
                )
            except torch.OutOfMemoryError as exc:
                if self.design_batch_size == 1:
                    raise RuntimeError(
                        "Wavefront Map CUDA out of memory: "
                        f"design_count={self.system.system_count}, field_count={len(self.field_indices)}, "
                        f"wavelength_count={len(self.wavelength_indices)}, sample_count={self.sample_count}."
                    ) from exc
                self.design_batch_size = max(1, self.design_batch_size // 2)
                retry_after_oom = True

            if retry_after_oom:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            start = self._start
            self._start = stop
            self.minibatch_count += 1
            return start, stop, batch


def run_wavefront(system: MultiOpticalSystem, settings: WavefrontSettings) -> WavefrontResult:
    """执行 Wavefront Map 分析。"""
    if system.tracer is None:
        raise ValueError("wavefront requires system.tracer.")
    if system.first_order_data is None:
        raise ValueError("wavefront requires system.prepare() before run().")

    field_indices = _resolve_field_indices(system, settings.field_indices)
    wavelength_indices = _resolve_wavelength_indices(system, settings.wavelength_indices)
    sample_count = int(settings.sample_count)
    if settings.save_path is not None:
        if system.system_count != 1 or len(field_indices) != 1 or len(wavelength_indices) != 1:
            raise ValueError("wavefront image export requires single design, single field, and single wavelength.")

    batches = iter_wavefront_design_batches(
        system,
        field_indices=field_indices,
        wavelength_indices=wavelength_indices,
        sample_count=sample_count,
    )
    opd_batches = []
    rms_batches = []
    valid_mask_batches = []
    valid_count_batches = []
    valid_fraction_batches = []
    pupil_x = None
    pupil_y = None
    for _, _, batch in batches:
        opd_batches.append(batch.opd.detach().cpu())
        rms_batches.append(batch.rms_wavefront.detach().cpu())
        valid_mask_batches.append(batch.valid_mask.detach().cpu())
        valid_count_batches.append(batch.valid_count.detach().cpu())
        valid_fraction_batches.append(batch.valid_fraction.detach().cpu())
        pupil_x = batch.pupil_x.detach().cpu()
        pupil_y = batch.pupil_y.detach().cpu()

    opd = torch.cat(opd_batches, dim=0)
    rms_wavefront = torch.cat(rms_batches, dim=0)
    valid_mask = torch.cat(valid_mask_batches, dim=0)
    valid_count = torch.cat(valid_count_batches, dim=0)
    valid_fraction = torch.cat(valid_fraction_batches, dim=0)

    figure = None
    axes = None
    save_path = None
    if settings.save_path is not None:
        figure, axes, save_path = plot_wavefront(
            system,
            opd,
            rms_wavefront=rms_wavefront,
            pupil_x=pupil_x,
            pupil_y=pupil_y,
            field_indices=field_indices,
            wavelength_indices=wavelength_indices,
            save_path=settings.save_path,
        )

    return WavefrontResult(
        opd=opd,
        rms_wavefront=rms_wavefront,
        valid_mask=valid_mask,
        valid_count=valid_count,
        valid_fraction=valid_fraction,
        pupil_x=pupil_x,
        pupil_y=pupil_y,
        field_indices=field_indices,
        wavelength_indices=wavelength_indices,
        sample_count=sample_count,
        figure=figure,
        axes=axes,
        save_path=save_path,
        detected_design_batch_size=batches.initial_design_batch_size,
        design_batch_size=batches.design_batch_size,
        minibatch_count=batches.minibatch_count,
    )


def iter_wavefront_design_batches(
    system: MultiOpticalSystem,
    *,
    field_indices: tuple[int, ...],
    wavelength_indices: tuple[int, ...],
    sample_count: int,
) -> WavefrontDesignBatchIterator:
    """创建 Wavefront Map design minibatch 迭代器。"""
    return WavefrontDesignBatchIterator(
        system,
        field_indices=field_indices,
        wavelength_indices=wavelength_indices,
        sample_count=int(sample_count),
    )


def compute_wavefront_batch(
    system: MultiOpticalSystem,
    *,
    field_indices: tuple[int, ...],
    wavelength_indices: tuple[int, ...],
    sample_count: int,
) -> WavefrontBatch:
    """一次追迹计算多个视场、多个单波长的 Wavefront Map。"""
    if system.tracer is None:
        raise ValueError("wavefront requires system.tracer.")
    if system.first_order_data is None:
        raise ValueError("wavefront requires system.prepare() before run().")

    fields = _select_fields(system, field_indices)
    sample = sample_zemax_wavefront_pupil(int(sample_count))
    trace_result = trace_pupil_to_image(system, fields, wavelength_indices, sample)
    wave_data = extract_image_wave_data(system, trace_result, sample)
    opd, rms_wavefront = _compute_wavefront_opd(system, wave_data)

    pupil_x = wave_data.pupil_coordinates[: wave_data.ordinary_ray_count, 0].reshape(int(sample_count), int(sample_count))
    pupil_y = wave_data.pupil_coordinates[: wave_data.ordinary_ray_count, 1].reshape(int(sample_count), int(sample_count))
    valid_mask = wave_data.valid_points[..., : wave_data.ordinary_ray_count].reshape(
        system.system_count, len(field_indices), len(wavelength_indices), int(sample_count), int(sample_count)
    )
    valid_count = valid_mask.sum(dim=(-2, -1))
    pupil_count = (pupil_x * pupil_x + pupil_y * pupil_y <= 1.0 + 1e-12).sum()

    return WavefrontBatch(
        opd=opd,
        rms_wavefront=rms_wavefront,
        valid_mask=valid_mask,
        valid_count=valid_count,
        valid_fraction=valid_count.to(dtype=torch.float64) / pupil_count,
        pupil_x=pupil_x,
        pupil_y=pupil_y,
        field_indices=field_indices,
        wavelength_indices=wavelength_indices,
        sample_count=int(sample_count),
    )


def sample_zemax_wavefront_pupil(sample_count: int) -> SamplingResult:
    """生成与 Zemax Wavefront Map 对齐的方形 pupil 网格，并追加 reference chief ray。"""
    count = int(sample_count)
    if count <= 1:
        raise ValueError("sample_count must be greater than 1.")

    # Zemax 的 DataGrid 显示范围为 [-1, 1]，实际追迹网格向负方向偏移半个显示步长，
    # 从而让偶数采样率的 (N/2, N/2) 点严格落在 pupil 原点。
    center_offset = 1.0 / (count - 1)
    axis_x = torch.linspace(-1.0, 1.0, count, dtype=torch.float64) - center_offset
    axis_y = torch.linspace(-1.0, 1.0, count, dtype=torch.float64) - center_offset
    y_grid, x_grid = torch.meshgrid(axis_y, axis_x, indexing="ij")
    coordinates = torch.stack((x_grid.reshape(-1), y_grid.reshape(-1)), dim=-1)
    weights = torch.ones(coordinates.shape[0] + 1, dtype=torch.float64)
    coordinates = torch.cat((coordinates, torch.zeros((1, 2), dtype=torch.float64)), dim=0)
    weights[-1] = 0.0
    return SamplingResult(
        pupil_coordinates=coordinates,
        weights=weights,
        pattern="zemax_wavefront",
        chief_ray_index=coordinates.shape[0] - 1,
        sample_ray_count=coordinates.shape[0] - 1,
    )


def trace_pupil_to_image(
    system: MultiOpticalSystem,
    fields: Sequence[FieldPoint],
    wavelength_indices: Sequence[int],
    sample: SamplingResult,
) -> TraceResult:
    """采样入瞳并追迹到像面。"""
    first_order_data = system.first_order_data
    if first_order_data is None:
        raise ValueError("wavefront requires system.prepare() before tracing rays.")

    rays = build_input_rays_from_sample(
        system,
        fields,
        wavelength_indices,
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


def extract_image_wave_data(
    system: MultiOpticalSystem,
    trace_result: TraceResult,
    sample: SamplingResult,
) -> ImageWaveData:
    """从像面追迹结果中提取 PSF / Wavefront 共用数据。"""
    if trace_result.rays.opl is None:
        raise ValueError("image wave data requires traced OPL.")

    x = torch.as_tensor(trace_result.rays.x, dtype=torch.float64)
    device = x.device
    y = torch.as_tensor(trace_result.rays.y, dtype=torch.float64, device=device)
    z = torch.as_tensor(trace_result.rays.z, dtype=torch.float64, device=device)
    l = torch.as_tensor(trace_result.rays.l, dtype=torch.float64, device=device)
    m = torch.as_tensor(trace_result.rays.m, dtype=torch.float64, device=device)
    n = torch.as_tensor(trace_result.rays.n, dtype=torch.float64, device=device)
    opl = torch.as_tensor(trace_result.rays.opl, dtype=torch.float64, device=device)
    valid = torch.as_tensor(trace_result.valid, dtype=torch.bool, device=device)
    wavelength_index = torch.as_tensor(trace_result.rays.wavelength_index, dtype=torch.int64, device=device)

    if x.ndim != 4:
        raise ValueError("image wave data trace result must have shape (design, field, wavelength, ray).")

    ordinary_ray_count = int(sample.sample_ray_count)
    chief_ray_index = int(sample.chief_ray_index)
    pupil_coordinates = torch.as_tensor(sample.pupil_coordinates, dtype=torch.float64, device=device)
    pupil_weights = torch.as_tensor(sample.weights, dtype=torch.float64, device=device)

    ordinary_coordinates = pupil_coordinates[:ordinary_ray_count]
    inside_pupil = torch.sum(ordinary_coordinates * ordinary_coordinates, dim=-1) <= 1.0 + 1e-12
    valid_points = valid.clone()
    valid_points &= valid[..., chief_ray_index].unsqueeze(-1)
    valid_points[..., ordinary_ray_count:] = False
    valid_points[..., :ordinary_ray_count] &= inside_pupil.reshape(1, 1, 1, -1)

    image_points = torch.stack((x, y, z), dim=-1)
    ray_directions = normalized(torch.stack((l, m, n), dim=-1))
    chief_points = image_points[..., chief_ray_index, :]
    wavelength_mm = system._material_data.wavelength_um[wavelength_index[..., 0]] * 1.0e-3

    return ImageWaveData(
        image_points=image_points,
        ray_directions=ray_directions,
        opl=opl,
        wavelength_mm=wavelength_mm,
        chief_points=chief_points,
        valid_points=valid_points,
        pupil_coordinates=pupil_coordinates,
        pupil_weights=pupil_weights,
        ordinary_ray_count=ordinary_ray_count,
        chief_ray_index=chief_ray_index,
    )


def _compute_wavefront_opd(
    system: MultiOpticalSystem,
    wave_data: ImageWaveData,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按焦系统参考球或无焦系统参考平面计算 OPD，输出单位 waves。"""
    if system.first_order_data is None:
        raise ValueError("wavefront requires system.prepare() before OPD calculation.")

    image_points = wave_data.image_points
    ray_directions = wave_data.ray_directions
    opl = wave_data.opl
    chief_points = wave_data.chief_points
    valid_points = wave_data.valid_points
    ordinary_ray_count = wave_data.ordinary_ray_count
    design_count, field_count, wavelength_count, ray_count = image_points.shape[:4]
    flattened_count = field_count * wavelength_count
    flattened_image_points = image_points.reshape(design_count, flattened_count, ray_count, 3)
    flattened_ray_directions = ray_directions.reshape(design_count, flattened_count, ray_count, 3)
    flattened_opl = opl.reshape(design_count, flattened_count, ray_count)
    flattened_chief_points = chief_points.reshape(design_count, flattened_count, 3)
    flattened_valid = valid_points.reshape(design_count, flattened_count, ray_count)
    exit_pupil_z = system.first_order_data.exit_pupil_z.reshape(design_count, 1).expand(design_count, flattened_count)

    if system.architecture.afocal_image_space:
        _, opd_mm_flat = exit_pupil_planar_opd(
            image_points=flattened_image_points,
            ray_directions=flattened_ray_directions,
            opl=flattened_opl,
            chief_points=flattened_chief_points,
            chief_directions=flattened_ray_directions[:, :, wave_data.chief_ray_index],
            exit_pupil_z=exit_pupil_z,
            valid_points=flattened_valid,
        )
    else:
        _, opd_mm_flat = exit_pupil_referenced_opd(
            image_points=flattened_image_points,
            ray_directions=flattened_ray_directions,
            opl=flattened_opl,
            chief_points=flattened_chief_points,
            exit_pupil_z=exit_pupil_z,
            valid_points=flattened_valid,
        )
    sample_count = int(round(ordinary_ray_count**0.5))
    reference_ray_index = (sample_count // 2) * sample_count + sample_count // 2
    opd_mm_all = opd_mm_flat.reshape(design_count, field_count, wavelength_count, ray_count)
    piston_removed_opd_mm = opd_mm_all[..., :ordinary_ray_count]
    reference_opd_mm = opd_mm_all[..., reference_ray_index].unsqueeze(-1)
    opd_mm = piston_removed_opd_mm - reference_opd_mm
    wavelength_mm = wave_data.wavelength_mm[..., None].clamp_min(torch.finfo(torch.float64).eps)
    opd_waves = opd_mm / wavelength_mm
    piston_removed_opd_waves = piston_removed_opd_mm / wavelength_mm
    ordinary_valid = valid_points[..., :ordinary_ray_count]
    opd_waves = torch.where(ordinary_valid, opd_waves, torch.zeros_like(opd_waves))

    valid_weight = ordinary_valid.to(dtype=torch.float64)
    valid_count = valid_weight.sum(dim=-1)
    valid_piston_removed_opd = torch.where(
        ordinary_valid,
        piston_removed_opd_waves,
        torch.zeros_like(piston_removed_opd_waves),
    )
    rms_wavefront = torch.sqrt(
        torch.sum(valid_piston_removed_opd * valid_piston_removed_opd, dim=-1)
        / valid_count.clamp_min(1.0)
    )
    rms_wavefront = torch.where(valid_count > 0, rms_wavefront, torch.full_like(rms_wavefront, torch.nan))
    return (
        opd_waves.reshape(design_count, field_count, wavelength_count, sample_count, sample_count),
        rms_wavefront,
    )


def exit_pupil_referenced_opd(
    *,
    image_points: torch.Tensor,
    ray_directions: torch.Tensor,
    opl: torch.Tensor,
    chief_points: torch.Tensor,
    exit_pupil_z: torch.Tensor,
    valid_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """反向求交出瞳参考球面，返回交点和去活塞后的出瞳 OPD。"""
    backward_direction = -ray_directions
    delta = image_points - chief_points[:, :, None, :]
    axial_exit_center = torch.stack(
        (
            torch.zeros_like(exit_pupil_z),
            torch.zeros_like(exit_pupil_z),
            exit_pupil_z,
        ),
        dim=-1,
    )
    sphere_radius = torch.linalg.vector_norm(chief_points - axial_exit_center, dim=-1)

    quadratic_a = torch.sum(backward_direction * backward_direction, dim=-1)
    quadratic_b = 2.0 * torch.sum(backward_direction * delta, dim=-1)
    quadratic_c = torch.sum(delta * delta, dim=-1) - sphere_radius[:, :, None] * sphere_radius[:, :, None]
    discriminant = torch.clamp(quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c, min=0.0)
    t_back = (-quadratic_b + torch.sqrt(discriminant)) / (2.0 * quadratic_a.clamp_min(torch.finfo(torch.float64).eps))

    exit_pupil_points = image_points + t_back[..., None] * backward_direction
    opd_exit = t_back - opl
    valid_weight = valid_points.to(dtype=torch.float64)
    valid_count = valid_weight.sum(dim=-1).clamp_min(1.0)
    piston = (torch.where(valid_points, opd_exit, torch.zeros_like(opd_exit)).sum(dim=-1) / valid_count).unsqueeze(-1)
    return exit_pupil_points, opd_exit - piston


def exit_pupil_planar_opd(
    *,
    image_points: torch.Tensor,
    ray_directions: torch.Tensor,
    opl: torch.Tensor,
    chief_points: torch.Tensor,
    chief_directions: torch.Tensor,
    exit_pupil_z: torch.Tensor,
    valid_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把无焦像空间光线反向追迹到垂直主光线的出瞳参考平面。"""
    plane_normal = normalized(chief_directions)
    backward_chief = -plane_normal
    chief_to_plane = (exit_pupil_z - chief_points[..., 2]) / backward_chief[..., 2]
    plane_center = chief_points + chief_to_plane[..., None] * backward_chief

    backward_direction = -ray_directions
    denominator = torch.sum(backward_direction * plane_normal[:, :, None, :], dim=-1)
    numerator = torch.sum(
        (plane_center[:, :, None, :] - image_points) * plane_normal[:, :, None, :],
        dim=-1,
    )
    t_back = numerator / denominator
    exit_pupil_points = image_points + t_back[..., None] * backward_direction
    opd_exit = t_back - opl

    valid_weight = valid_points.to(dtype=torch.float64)
    valid_count = valid_weight.sum(dim=-1).clamp_min(1.0)
    piston = (torch.where(valid_points, opd_exit, torch.zeros_like(opd_exit)).sum(dim=-1) / valid_count).unsqueeze(-1)
    return exit_pupil_points, opd_exit - piston


def normalized(vector: torch.Tensor) -> torch.Tensor:
    """归一化方向余弦。"""
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float64).eps)
    return vector / norm


def _resolve_field_indices(
    system: MultiOpticalSystem,
    field_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    """解析公开 field_indices 参数。"""
    fields = list(system.fields)
    if not fields:
        raise ValueError("wavefront requires at least one field.")
    if field_indices is None:
        return tuple(range(len(fields)))
    resolved = tuple(int(index) for index in field_indices)
    for index in resolved:
        if index < 0 or index >= len(fields):
            raise ValueError("field index is out of range.")
    return resolved


def _resolve_wavelength_indices(
    system: MultiOpticalSystem,
    wavelength_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    """解析公开 wavelength_indices 参数，禁止全波长混合索引 -1。"""
    wavelengths = list(system.wavelengths)
    if not wavelengths:
        raise ValueError("wavefront requires at least one wavelength.")
    if wavelength_indices is None:
        return (int(system.wavelengths.primary_index),)
    resolved = tuple(int(index) for index in wavelength_indices)
    for index in resolved:
        if index == -1:
            raise ValueError("Wavefront Map 不支持 wavelength index -1；请显式传入具体波长索引。")
        if index < 0 or index >= len(wavelengths):
            raise ValueError("wavelength index is out of range.")
    return resolved


def _select_fields(
    system: MultiOpticalSystem,
    field_indices: tuple[int, ...],
) -> list[FieldPoint]:
    """按索引选择视场。"""
    return [system.fields[index] for index in field_indices]


def plot_wavefront(
    system: MultiOpticalSystem,
    opd: torch.Tensor,
    *,
    rms_wavefront: torch.Tensor,
    pupil_x: torch.Tensor,
    pupil_y: torch.Tensor,
    field_indices: tuple[int, ...],
    wavelength_indices: tuple[int, ...],
    save_path: str | Path,
):
    """将单设计、单视场、单波长 Wavefront Map 导出为图片。"""
    if system.system_count != 1 or len(field_indices) != 1 or len(wavelength_indices) != 1:
        raise ValueError("wavefront image export requires single design, single field, and single wavelength.")

    opd_cpu = torch.as_tensor(opd, dtype=torch.float64).detach().cpu()
    rms_cpu = torch.as_tensor(rms_wavefront, dtype=torch.float64).detach().cpu()
    pupil_x_cpu = torch.as_tensor(pupil_x, dtype=torch.float64).detach().cpu()
    pupil_y_cpu = torch.as_tensor(pupil_y, dtype=torch.float64).detach().cpu()
    image = opd_cpu[0, 0, 0]
    inside_pupil = pupil_x_cpu * pupil_x_cpu + pupil_y_cpu * pupil_y_cpu <= 1.0 + 1e-12
    valid_values = image[inside_pupil]
    vmin = float(valid_values.min().item())
    vmax = float(valid_values.max().item())
    display_image = torch.where(inside_pupil, image, torch.full_like(image, vmin))

    field_index = field_indices[0]
    wavelength_index = wavelength_indices[0]
    wavelength_um = float(system.wavelengths[wavelength_index].value_um)
    title = (
        f"design=0, field={field_index}, wavelength={wavelength_um:.4f} um\n"
        f"RMS={float(rms_cpu[0, 0, 0].item()):.6g} waves"
    )
    extent = (
        float(pupil_x_cpu.min().item()),
        float(pupil_x_cpu.max().item()),
        float(pupil_y_cpu.min().item()),
        float(pupil_y_cpu.max().item()),
    )

    figure, axis = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    plotted = axis.imshow(
        display_image.numpy(),
        cmap="jet",
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_xlabel("Normalized pupil x")
    axis.set_ylabel("Normalized pupil y")
    axis.set_aspect("equal")
    figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04, label="OPD (waves)")

    figure.tight_layout()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    return figure, axis, str(output_path)
