from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import matplotlib
import torch

from .analysis import PSFResult, PSFSettings
from .sampling import SquarePupilSampler
from .wavefront import ImageWaveData, extract_image_wave_data, trace_pupil_to_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plotting_config import configure_matplotlib_chinese

configure_matplotlib_chinese()

if TYPE_CHECKING:
    from .system import MultiOpticalSystem
    from .system_specs import FieldPoint, Wavelength


@dataclass(slots=True)
class HuygensPSFBatch:
    """惠更斯 PSF 批量计算结果。"""

    psf: torch.Tensor
    strehl_ratio: torch.Tensor | None
    psf_by_wavelength: torch.Tensor | None
    strehl_by_wavelength: torch.Tensor | None
    pixel_pitch_um: torch.Tensor
    field_indices: tuple[int, ...]
    wavelength_indices: tuple[int, ...]


class HuygensPSFDesignBatchIterator(Iterator[tuple[int, int, HuygensPSFBatch]]):
    """按自动 design batch size 顺序计算惠更斯 PSF。"""

    def __init__(
        self,
        system: MultiOpticalSystem,
        *,
        field_indices: tuple[int, ...],
        wavelength_index: int | None,
        pupil_sample_count: int,
        image_sample_count: int,
        image_delta_um: float,
        compute_ideal_psf: bool = True,
    ) -> None:
        self.system = system
        self.field_indices = field_indices
        self.wavelength_index = wavelength_index
        self.pupil_sample_count = pupil_sample_count
        self.image_sample_count = image_sample_count
        self.image_delta_um = image_delta_um
        self.compute_ideal_psf = compute_ideal_psf
        self.design_batch_size = resolve_huygens_design_batch_size(
            system,
            field_indices=field_indices,
            wavelength_index=wavelength_index,
            pupil_sample_count=pupil_sample_count,
            image_sample_count=image_sample_count,
            image_delta_um=image_delta_um,
            compute_ideal_psf=compute_ideal_psf,
        )
        self.initial_design_batch_size = self.design_batch_size
        self.minibatch_count = 0
        self._start = 0

    def __iter__(self) -> HuygensPSFDesignBatchIterator:
        return self

    def __next__(self) -> tuple[int, int, HuygensPSFBatch]:
        if self._start >= self.system.system_count:
            raise StopIteration

        while True:
            stop = min(self._start + self.design_batch_size, self.system.system_count)
            batch_system = self.system.design_batch_view(self._start, stop)
            retry_after_oom = False
            try:
                batch = compute_huygens_psf_batch(
                    batch_system,
                    field_indices=self.field_indices,
                    wavelength_index=self.wavelength_index,
                    pupil_sample_count=self.pupil_sample_count,
                    image_sample_count=self.image_sample_count,
                    image_delta_um=self.image_delta_um,
                    compute_ideal_psf=self.compute_ideal_psf,
                )
            except torch.OutOfMemoryError as exc:
                if self.design_batch_size == 1:
                    raise RuntimeError(
                        _huygens_oom_message(
                            self.system,
                            field_indices=self.field_indices,
                            wavelength_index=self.wavelength_index,
                            pupil_sample_count=self.pupil_sample_count,
                            image_sample_count=self.image_sample_count,
                            design_batch_size=1,
                        )
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


def run_huygens_psf(system: MultiOpticalSystem, settings: PSFSettings) -> PSFResult:
    """执行最小惠更斯 PSF 分析。"""
    if system.tracer is None:
        raise ValueError("huygens_psf requires system.tracer.")
    if system.first_order_data is None:
        raise ValueError("huygens_psf requires system.prepare() before run().")
    if settings.save_path is not None and system.system_count != 1:
        raise ValueError("huygens_psf image export currently only supports a single design_view.")

    field_index = _select_field_index(system, settings.field_index)
    batches = iter_huygens_psf_design_batches(
        system,
        field_indices=(field_index,),
        wavelength_index=settings.wavelength_index,
        pupil_sample_count=int(settings.pupil_sample_count),
        image_sample_count=int(settings.image_sample_count),
        image_delta_um=float(settings.image_delta_um),
    )
    psf_batches = []
    strehl_batches = []
    psf_by_wavelength_batches = []
    strehl_by_wavelength_batches = []
    pixel_pitch_batches = []
    wavelength_indices: tuple[int, ...] = ()
    for _, _, batch in batches:
        if (
            batch.strehl_ratio is None
            or batch.psf_by_wavelength is None
            or batch.strehl_by_wavelength is None
        ):
            raise RuntimeError("huygens_psf expected Strehl and per-wavelength PSF results.")
        psf_batches.append(batch.psf.detach().cpu())
        strehl_batches.append(batch.strehl_ratio.detach().cpu())
        psf_by_wavelength_batches.append(batch.psf_by_wavelength.detach().cpu())
        strehl_by_wavelength_batches.append(batch.strehl_by_wavelength.detach().cpu())
        pixel_pitch_batches.append(batch.pixel_pitch_um.detach().cpu())
        wavelength_indices = batch.wavelength_indices

    psf = torch.cat(psf_batches, dim=0)
    strehl_ratio = torch.cat(strehl_batches, dim=0)
    psf_by_wavelength = torch.cat(psf_by_wavelength_batches, dim=0)
    strehl_by_wavelength = torch.cat(strehl_by_wavelength_batches, dim=0)
    pixel_pitch_um = torch.cat(pixel_pitch_batches, dim=0)

    # 绘图
    figure = None
    axes = None
    save_path = None
    if settings.save_path is not None:
        figure, axes, save_path = plot_huygens_psf(
            system,
            psf,
            strehl_ratio=strehl_ratio,
            wavelength_indices=wavelength_indices,
            pixel_pitch_um=float(pixel_pitch_um[0].item()),
            save_path=settings.save_path,
        )

    return PSFResult(
        psf=psf,
        strehl_ratio=strehl_ratio,
        psf_by_wavelength=psf_by_wavelength[:, 0],
        strehl_by_wavelength=strehl_by_wavelength[:, 0],
        pixel_pitch_um=pixel_pitch_um,
        field_index=field_index,
        wavelength_indices=wavelength_indices,
        figure=figure,
        axes=axes,
        save_path=save_path,
        detected_design_batch_size=batches.initial_design_batch_size,
        design_batch_size=batches.design_batch_size,
        minibatch_count=batches.minibatch_count,
    )


def iter_huygens_psf_design_batches(
    system: MultiOpticalSystem,
    *,
    field_indices: tuple[int, ...],
    wavelength_index: int | None,
    pupil_sample_count: int,
    image_sample_count: int,
    image_delta_um: float,
    compute_ideal_psf: bool = True,
) -> HuygensPSFDesignBatchIterator:
    """创建共享的惠更斯 PSF design minibatch 迭代器。"""
    return HuygensPSFDesignBatchIterator(
        system,
        field_indices=field_indices,
        wavelength_index=wavelength_index,
        pupil_sample_count=pupil_sample_count,
        image_sample_count=image_sample_count,
        image_delta_um=image_delta_um,
        compute_ideal_psf=compute_ideal_psf,
    )


def resolve_huygens_design_batch_size(
    system: MultiOpticalSystem,
    *,
    field_indices: tuple[int, ...],
    wavelength_index: int | None,
    pupil_sample_count: int,
    image_sample_count: int,
    image_delta_um: float,
    compute_ideal_psf: bool = True,
) -> int:
    """通过单设计 PSF 峰值显存估算当前分析的 design batch size。"""
    first_order_data = system.first_order_data
    if first_order_data is None:
        raise ValueError("huygens_psf requires system.prepare() before resolving design batch size.")
    device = first_order_data.working_f_number.device
    if device.type != "cuda":
        return system.system_count

    probe_system = system.design_batch_view(0, 1)
    try:
        warmup = compute_huygens_psf_batch(
            probe_system,
            field_indices=field_indices,
            wavelength_index=wavelength_index,
            pupil_sample_count=pupil_sample_count,
            image_sample_count=image_sample_count,
            image_delta_um=image_delta_um,
            compute_ideal_psf=compute_ideal_psf,
        )
        del warmup
        torch.cuda.synchronize(device)

        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        probe = compute_huygens_psf_batch(
            probe_system,
            field_indices=field_indices,
            wavelength_index=wavelength_index,
            pupil_sample_count=pupil_sample_count,
            image_sample_count=image_sample_count,
            image_delta_um=image_delta_um,
            compute_ideal_psf=compute_ideal_psf,
        )
        torch.cuda.synchronize(device)
        peak_increment = max(
            1,
            torch.cuda.max_memory_allocated(device) - baseline_allocated,
        )
        del probe
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            _huygens_oom_message(
                system,
                field_indices=field_indices,
                wavelength_index=wavelength_index,
                pupil_sample_count=pupil_sample_count,
                image_sample_count=image_sample_count,
                design_batch_size=1,
            )
        ) from exc

    free_memory, _ = torch.cuda.mem_get_info(device)
    reusable_cached_memory = (
        torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    available_memory = free_memory + reusable_cached_memory
    candidate = int(float(available_memory) * 0.75 // peak_increment)
    return min(system.system_count, max(1, candidate))


def _huygens_oom_message(
    system: MultiOpticalSystem,
    *,
    field_indices: tuple[int, ...],
    wavelength_index: int | None,
    pupil_sample_count: int,
    image_sample_count: int,
    design_batch_size: int,
) -> str:
    """构造单设计仍无法执行时的显存错误信息。"""
    wavelength_count = len(system.wavelengths) if wavelength_index == -1 else 1
    memory_description = ""
    first_order_data = system.first_order_data
    if first_order_data is not None and first_order_data.working_f_number.device.type == "cuda":
        free_memory, total_memory = torch.cuda.mem_get_info(first_order_data.working_f_number.device)
        memory_description = (
            f", CUDA free={free_memory / 2**30:.3f} GiB"
            f", total={total_memory / 2**30:.3f} GiB"
        )
    return (
        "Huygens PSF CUDA out of memory: "
        f"design_count={system.system_count}, field_count={len(field_indices)}, "
        f"wavelength_count={wavelength_count}, pupil_sample_count={pupil_sample_count}, "
        f"image_sample_count={image_sample_count}, design_batch_size={design_batch_size}"
        f"{memory_description}."
    )


def compute_huygens_psf_batch(
    system: MultiOpticalSystem,
    *,
    field_indices: tuple[int, ...],
    wavelength_index: int | None,
    pupil_sample_count: int,
    image_sample_count: int,
    image_delta_um: float,
    compute_ideal_psf: bool = True,
) -> HuygensPSFBatch:
    """一次追迹计算多个视场的单色或混合惠更斯 PSF。"""
    if system.tracer is None:
        raise ValueError("huygens_psf requires system.tracer.")
    if system.first_order_data is None:
        raise ValueError("huygens_psf requires system.prepare() before run().")

    # 1. 选择视场和波长
    fields = _select_fields(system, field_indices)
    wavelengths, wavelength_indices = _select_wavelengths(system, wavelength_index)
    pixel_pitch_um = _resolve_image_delta_um(
        system,
        image_delta_um=image_delta_um,
        pupil_sample_count=pupil_sample_count,
        wavelengths=wavelengths,
    )

    # 2. 一次追迹全部视场和波长
    sample = SquarePupilSampler(
        nx=pupil_sample_count,
        ny=pupil_sample_count,
    ).sample()
    trace_result = trace_pupil_to_image(system, fields, wavelength_indices, sample)
    wave_data = extract_image_wave_data(system, trace_result, sample)

    # 3. 在各视场共同像面网格上计算逐波长 PSF
    psf_by_wavelength, ideal_psf_by_wavelength = compute_huygens_psf(
        system,
        wave_data,
        image_sample_count=image_sample_count,
        image_delta_um=pixel_pitch_um,
        compute_ideal_psf=compute_ideal_psf,
    )
    if compute_ideal_psf:
        if ideal_psf_by_wavelength is None:
            raise RuntimeError("huygens_psf expected ideal PSF results.")
        strehl_by_wavelength = _strehl_from_psfs(
            psf_by_wavelength,
            ideal_psf_by_wavelength,
        )
    else:
        strehl_by_wavelength = None

    # 4. 单色直接返回，全波长先混合 PSF
    psf, strehl_ratio = _reduce_huygens_psf(
        system,
        psf_by_wavelength=psf_by_wavelength,
        strehl_by_wavelength=strehl_by_wavelength,
    )
    return HuygensPSFBatch(
        psf=psf,
        strehl_ratio=strehl_ratio,
        psf_by_wavelength=psf_by_wavelength if compute_ideal_psf else None,
        strehl_by_wavelength=strehl_by_wavelength,
        pixel_pitch_um=pixel_pitch_um,
        field_indices=field_indices,
        wavelength_indices=wavelength_indices,
    )


def compute_huygens_psf(
    system: MultiOpticalSystem,
    wave_data: ImageWaveData,
    *,
    image_sample_count: int,
    image_delta_um: torch.Tensor,
    compute_ideal_psf: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """在各视场统一物理像面网格上计算逐波长惠更斯 PSF。"""
    if image_sample_count < 2:
        raise ValueError("image_sample_count must be greater than or equal to 2.")

    if system.first_order_data is None:
        raise ValueError("huygens_psf requires system.prepare() before psf calculation.")

    device = wave_data.image_points.device
    image_points = wave_data.image_points
    ray_directions = wave_data.ray_directions
    opl = wave_data.opl
    chief_points = wave_data.chief_points
    valid_points = wave_data.valid_points
    pupil_weights = wave_data.pupil_weights
    wavelength_mm = wave_data.wavelength_mm

    # 1. 单波长使用自身像点，全波长使用主波长像点建立共同网格
    reference_chief_points = _reference_chief_points(
        chief_points,
        primary_wavelength_index=int(system.wavelengths.primary_index),
    )

    image_x, image_y, image_z = _image_grid(
        reference_chief_points,
        image_sample_count=image_sample_count,
        image_delta_um=image_delta_um,
        image_plane_rotation=system.frame_data.rotations[:, -1],
        device=device,
    )

    # 2. 按视场分块积分，避免多视场高采样率产生过大的复数相位临时张量。
    psf_by_field = []
    ideal_psf_by_field = []
    for field_index in range(image_points.shape[1]):
        field_psf, field_ideal_psf = _huygens_integral(
            image_points=image_points[:, field_index : field_index + 1],
            ray_directions=ray_directions[:, field_index : field_index + 1],
            opl=opl[:, field_index : field_index + 1],
            chief_points=chief_points[:, field_index : field_index + 1],
            image_x=image_x[:, field_index : field_index + 1],
            image_y=image_y[:, field_index : field_index + 1],
            image_z=image_z[:, field_index : field_index + 1],
            wavelength_mm=wavelength_mm[:, field_index : field_index + 1],
            valid_points=valid_points[:, field_index : field_index + 1],
            pupil_weights=pupil_weights,
            compute_ideal_psf=compute_ideal_psf,
        )
        psf_by_field.append(field_psf)
        if compute_ideal_psf:
            if field_ideal_psf is None:
                raise RuntimeError("huygens_psf expected ideal PSF results.")
            ideal_psf_by_field.append(field_ideal_psf)
    psf_by_wavelength = torch.cat(psf_by_field, dim=1)
    if not compute_ideal_psf:
        return psf_by_wavelength, None
    return psf_by_wavelength, torch.cat(ideal_psf_by_field, dim=1)


def _reference_chief_points(
    chief_points: torch.Tensor,
    *,
    primary_wavelength_index: int,
) -> torch.Tensor:
    """选择像面网格参考点；多波长统一使用主波长主光线像点。"""
    local_wavelength_index = 0 if chief_points.shape[2] == 1 else primary_wavelength_index
    return chief_points[:, :, local_wavelength_index]


def _strehl_from_psfs(
    psf_by_wavelength: torch.Tensor,
    ideal_psf_by_wavelength: torch.Tensor,
) -> torch.Tensor:
    """计算逐波长 Strehl ratio。"""
    peak = torch.amax(psf_by_wavelength, dim=(-2, -1))
    ideal_peak = torch.amax(ideal_psf_by_wavelength, dim=(-2, -1))
    valid_peak = ideal_peak > torch.finfo(torch.float64).eps
    strehl_ratio = peak / ideal_peak.clamp_min(torch.finfo(torch.float64).eps)
    return torch.where(valid_peak, strehl_ratio, torch.full_like(strehl_ratio, torch.nan))


def _reduce_huygens_psf(
    system: MultiOpticalSystem,
    *,
    psf_by_wavelength: torch.Tensor,
    strehl_by_wavelength: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """将逐波长结果规整为单波长或多色最终输出。"""
    if psf_by_wavelength.shape[2] == 1:
        strehl_ratio = None if strehl_by_wavelength is None else strehl_by_wavelength[:, :, 0]
        return psf_by_wavelength[:, :, 0], strehl_ratio

    weights = system._material_data.wavelength_weights
    weights = weights / weights.sum().clamp_min(torch.finfo(torch.float64).eps)
    mixed_psf = torch.sum(
        psf_by_wavelength * weights.reshape(1, 1, -1, 1, 1),
        dim=2,
    )
    if strehl_by_wavelength is None:
        return mixed_psf, None
    mixed_strehl = torch.amax(mixed_psf, dim=(-2, -1))
    return mixed_psf, mixed_strehl


def _select_field_index(system: MultiOpticalSystem, field_index: int) -> int:
    """检查单个视场索引。"""
    fields = list(system.fields)
    if not fields:
        raise ValueError("huygens_psf requires at least one field.")
    field_index = int(field_index)
    if field_index < 0 or field_index >= len(fields):
        raise ValueError("field_index is out of range.")
    return field_index


def _select_fields(
    system: MultiOpticalSystem,
    field_indices: tuple[int, ...],
) -> list[FieldPoint]:
    """按索引选择多个视场。"""
    return [system.fields[_select_field_index(system, field_index)] for field_index in field_indices]


def _select_wavelengths(
    system: MultiOpticalSystem,
    wavelength_index: int | None,
) -> tuple[list[Wavelength], tuple[int, ...]]:
    """解析 PSF 使用的波长；-1 表示全部波长。"""
    wavelengths = list(system.wavelengths)
    if not wavelengths:
        raise ValueError("huygens_psf requires at least one wavelength.")

    if wavelength_index is None:
        indices = (int(system.wavelengths.primary_index),)
    elif int(wavelength_index) == -1:
        indices = tuple(range(len(wavelengths)))
    else:
        requested_index = int(wavelength_index)
        if requested_index < 0 or requested_index >= len(wavelengths):
            raise ValueError("wavelength_index is out of range.")
        indices = (requested_index,)
    return [wavelengths[index] for index in indices], indices


def _resolve_image_delta_um(
    system: MultiOpticalSystem,
    *,
    image_delta_um: float,
    pupil_sample_count: int,
    wavelengths: list[Wavelength],
) -> torch.Tensor:
    """解析像面采样间隔；零值按 Zemax 默认公式自动计算。

    Zemax 默认公式为：
    Image Delta = wavelength * Working F/# / sqrt(pupil sample count)

    全波长分析使用系统定义波长中的最长波长。
    """
    requested_delta_um = float(image_delta_um)
    if requested_delta_um < 0.0:
        raise ValueError("image_delta_um must be greater than or equal to zero.")

    first_order_data = system.first_order_data
    if first_order_data is None:
        raise ValueError("huygens_psf requires system.prepare() before resolving image delta.")
    device = first_order_data.working_f_number.device
    if requested_delta_um > 0.0:
        return torch.full(
            (system.system_count,),
            requested_delta_um,
            dtype=torch.float64,
            device=device,
        )

    wavelength_um = max(float(wavelength.value_um) for wavelength in wavelengths)
    if pupil_sample_count <= 0:
        raise ValueError("pupil_sample_count must be greater than zero.")
    return (
        first_order_data.working_f_number
        * wavelength_um
        / torch.sqrt(torch.tensor(float(pupil_sample_count), dtype=torch.float64, device=device))
    )


def _image_grid(
    reference_chief_points: torch.Tensor,
    *,
    image_sample_count: int,
    image_delta_um: torch.Tensor,
    image_plane_rotation: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """沿局部像面 x/y 基向量生成方形采样网格。"""
    if reference_chief_points.ndim != 3 or reference_chief_points.shape[-1] != 3:
        raise ValueError("reference_chief_points must have shape (design, field, 3).")
    image_plane_rotation = torch.as_tensor(
        image_plane_rotation,
        dtype=torch.float64,
        device=device,
    )
    pixel_pitch_mm = torch.as_tensor(
        image_delta_um,
        dtype=torch.float64,
        device=device,
    ).reshape(-1, 1) * 1.0e-3
    axis_index = (
        torch.arange(image_sample_count, dtype=torch.float64, device=device)
        - image_sample_count // 2
    )
    axis = pixel_pitch_mm * axis_index.reshape(1, -1)
    local_x_basis = image_plane_rotation[:, :, 0]
    local_y_basis = image_plane_rotation[:, :, 1]
    grid = (
        reference_chief_points[:, :, None, None, None, :]
        + axis[:, None, None, None, :, None] * local_x_basis[:, None, None, None, None, :]
        + axis[:, None, None, :, None, None] * local_y_basis[:, None, None, None, None, :]
    )
    return grid[..., 0], grid[..., 1], grid[..., 2]


def _huygens_integral(
    *,
    image_points: torch.Tensor,
    ray_directions: torch.Tensor,
    opl: torch.Tensor,
    chief_points: torch.Tensor,
    image_x: torch.Tensor,
    image_y: torch.Tensor,
    image_z: torch.Tensor,
    wavelength_mm: torch.Tensor,
    valid_points: torch.Tensor,
    pupil_weights: torch.Tensor,
    compute_ideal_psf: bool = True,
    ray_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """使用像面光线数据执行局部平面波形式的惠更斯复振幅叠加。

    对第 j 根光线，它在像面网格点 P 上的相位由两部分组成：
    phase_j(P) = k * (OPL_j + dot(d_j, P - P_image_j) - piston)

    其中：
    k         = 2*pi / wavelength_mm
    OPL_j     = 第 j 根光线到像面的累计光程
    d_j       = 第 j 根光线在像面的单位方向向量
    P_image_j = 第 j 根光线的像面交点
    piston    = 有效光瞳上的加权公共光程，仅用于改善相位数值精度

    Args:
        ray_chunk_size: 分块处理的光线数量，降低显存峰值（默认256，实测771MB峰值）
    """
    num_rays = image_points.shape[3]

    # 预计算：构建像面网格和偏移
    image_grid = torch.stack((image_x, image_y, image_z), dim=-1)
    grid_offset = image_grid - chief_points[:, :, :, None, None, :]

    # 预计算：光程和权重
    ray_to_chief = chief_points[:, :, :, None, :] - image_points
    path_at_chief = opl + torch.sum(ray_directions * ray_to_chief, dim=-1)
    integration_weight = torch.where(valid_points, pupil_weights.reshape(1, 1, 1, -1), 0.0)
    weight_sum = integration_weight.sum(dim=-1, keepdim=True)
    integration_weight = torch.where(
        weight_sum > 0.0,
        integration_weight / weight_sum.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(integration_weight),
    )
    valid_path_at_chief = torch.where(valid_points, path_at_chief, torch.zeros_like(path_at_chief))
    piston = torch.sum(integration_weight * valid_path_at_chief, dim=-1, keepdim=True)
    relative_path_at_chief = path_at_chief - piston
    wave_number = 2.0 * torch.pi / wavelength_mm

    # 分块计算复振幅（降低显存峰值）
    grid_h = image_y.shape[-2] if image_y.dim() > 4 else 1
    grid_w = image_x.shape[-1]
    amplitude = torch.zeros(
        (image_points.shape[0], image_points.shape[1], image_points.shape[2], grid_h, grid_w),
        dtype=torch.complex128,
        device=image_points.device,
    )
    if compute_ideal_psf:
        ideal_amplitude = torch.zeros_like(amplitude)

    for start_idx in range(0, num_rays, ray_chunk_size):
        end_idx = min(start_idx + ray_chunk_size, num_rays)
        ray_chunk = slice(start_idx, end_idx)

        # 当前块的光线数据
        chunk_directions = ray_directions[:, :, :, ray_chunk, :]
        chunk_path = relative_path_at_chief[:, :, :, ray_chunk]
        chunk_weight = integration_weight[:, :, :, ray_chunk, None, None]

        # 计算相位倾斜（这是显存瓶颈）
        phase_tilt = torch.sum(
            chunk_directions[:, :, :, :, None, None, :] * grid_offset[:, :, :, None, :, :, :],
            dim=-1,
        )

        # 计算实际PSF的复指数核
        phase = wave_number[:, :, :, None, None, None] * (
            chunk_path[:, :, :, :, None, None] + phase_tilt
        )
        kernel = torch.complex(torch.cos(phase), torch.sin(phase))
        kernel = kernel * chunk_weight
        amplitude += kernel.sum(dim=3)

        # 计算理想PSF
        if compute_ideal_psf:
            ideal_phase = wave_number[:, :, :, None, None, None] * phase_tilt
            ideal_kernel = torch.complex(torch.cos(ideal_phase), torch.sin(ideal_phase))
            ideal_kernel = ideal_kernel * chunk_weight
            ideal_amplitude += ideal_kernel.sum(dim=3)

    # 计算PSF强度
    psf = torch.real(amplitude * torch.conj(amplitude))
    if not compute_ideal_psf:
        return psf, None

    ideal_psf = torch.real(ideal_amplitude * torch.conj(ideal_amplitude))
    return psf, ideal_psf


def plot_huygens_psf(
    system: MultiOpticalSystem,
    psf: torch.Tensor,
    *,
    strehl_ratio: torch.Tensor,
    wavelength_indices: tuple[int, ...],
    pixel_pitch_um: float,
    save_path: str | Path,
):
    """将单设计惠更斯 PSF 绘制为一张二维强度图并导出。"""
    if system.system_count != 1:
        raise ValueError("huygens_psf image export currently only supports a single design_view.")

    psf_cpu = torch.as_tensor(psf, dtype=torch.float64).detach().cpu()
    strehl_cpu = torch.as_tensor(strehl_ratio, dtype=torch.float64).detach().cpu()
    image_sample_count = psf_cpu.shape[-1]
    first_center_um = -float(image_sample_count // 2) * float(pixel_pitch_um)
    last_center_um = float(image_sample_count - 1 - image_sample_count // 2) * float(pixel_pitch_um)
    half_pixel_um = 0.5 * float(pixel_pitch_um)
    extent = (
        first_center_um - half_pixel_um,
        last_center_um + half_pixel_um,
        first_center_um - half_pixel_um,
        last_center_um + half_pixel_um,
    )

    if len(wavelength_indices) == 1:
        wavelength_index = wavelength_indices[0]
        intensity = psf_cpu[0, 0]
        display_strehl = float(strehl_cpu[0, 0].item())
        wavelength_um = float(system.wavelengths[wavelength_index].value_um)
        title = f"design=0, wavelength={wavelength_um:.4f} um\nStrehl={display_strehl:.4f}"
    else:
        intensity = psf_cpu[0, 0]
        display_strehl = float(strehl_cpu[0, 0].item())
        title = f"design=0, all wavelengths\nStrehl={display_strehl:.4f}"

    figure, axis = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    image = axis.imshow(
        intensity.numpy(),
        cmap="jet",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        vmin=0.0,
        vmax=float(intensity.max().item()),
    )
    axis.set_title(title)
    axis.set_xlabel("x (um)")
    axis.set_ylabel("y (um)")
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Intensity")

    figure.tight_layout()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    return figure, axis, str(output_path)
