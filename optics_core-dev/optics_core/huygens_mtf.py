from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import torch
import torch.nn.functional as functional

from .analysis import MTFResult, MTFSettings
from .huygens_psf import iter_huygens_psf_design_batches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plotting_config import configure_matplotlib_chinese

configure_matplotlib_chinese()

if TYPE_CHECKING:
    from .system import MultiOpticalSystem


MTF_FIELD_COLORS: tuple[str, ...] = (
    "#0000FF",
    "#00AA00",
    "#FF0000",
    "#FFD700",
    "#FF00FF",
    "#00C8C8",
)


def run_huygens_mtf(system: MultiOpticalSystem, settings: MTFSettings) -> MTFResult:
    """执行多视场惠更斯 MTF 分析。"""
    if settings.save_path is not None and system.system_count != 1:
        raise ValueError("huygens_mtf image export currently only supports a single design_view.")

    # 1. 自动按 design 分批计算全部目标视场的最终 PSF
    field_indices = _resolve_field_indices(system, settings.field_indices)
    first_order_data = system.first_order_data
    if first_order_data is None:
        raise ValueError("huygens_mtf requires system.prepare() before run().")
    frequencies_lp_per_mm = _resolve_frequencies(
        settings.frequencies_lp_per_mm,
        device=first_order_data.working_f_number.device,
    )
    batches = iter_huygens_psf_design_batches(
        system,
        field_indices=field_indices,
        wavelength_index=settings.wavelength_index,
        pupil_sample_count=int(settings.pupil_sample_count),
        image_sample_count=int(settings.image_sample_count),
        image_delta_um=float(settings.image_delta_um),
        compute_ideal_psf=False,
    )

    # 2. 每批立即计算 MTF，保留 GPU 结果并在最后统一回收 CPU
    sagittal_batches = []
    tangential_batches = []
    pixel_pitch_batches = []
    wavelength_indices: tuple[int, ...] = ()
    for _, _, psf_batch in batches:
        batch_sagittal, batch_tangential = compute_huygens_mtf(
            psf_batch.psf,
            pixel_pitch_um=psf_batch.pixel_pitch_um,
            frequencies_lp_per_mm=frequencies_lp_per_mm,
        )
        sagittal_batches.append(batch_sagittal.detach())
        tangential_batches.append(batch_tangential.detach())
        pixel_pitch_batches.append(psf_batch.pixel_pitch_um.detach())
        wavelength_indices = psf_batch.wavelength_indices

    sagittal = torch.cat(sagittal_batches, dim=0).cpu()
    tangential = torch.cat(tangential_batches, dim=0).cpu()
    pixel_pitch_um = torch.cat(pixel_pitch_batches, dim=0).cpu()
    frequencies_cpu = frequencies_lp_per_mm.detach().cpu()

    # 3. 绘制单设计多视场曲线
    figure = None
    axes = None
    save_path = None
    if settings.save_path is not None:
        figure, axes, save_path = plot_huygens_mtf(
            system,
            frequencies_cpu,
            sagittal,
            tangential,
            field_indices=field_indices,
            wavelength_indices=wavelength_indices,
            save_path=settings.save_path,
        )

    return MTFResult(
        frequencies_lp_per_mm=frequencies_cpu,
        sagittal=sagittal,
        tangential=tangential,
        pixel_pitch_um=pixel_pitch_um,
        field_indices=field_indices,
        wavelength_indices=wavelength_indices,
        figure=figure,
        axes=axes,
        save_path=save_path,
        detected_design_batch_size=batches.initial_design_batch_size,
        design_batch_size=batches.design_batch_size,
        minibatch_count=batches.minibatch_count,
    )


def compute_huygens_mtf(
    psf: torch.Tensor,
    *,
    pixel_pitch_um: torch.Tensor,
    frequencies_lp_per_mm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由二维 PSF 投影和 FFT 计算 Sagittal/Tangential MTF。"""

    # 1. 数据本地化和状态检查
    psf = torch.as_tensor(psf, dtype=torch.float64)
    if psf.ndim != 4:
        raise ValueError("psf must have shape (design, field, image_y, image_x).")
    device = psf.device
    pixel_pitch_um = torch.as_tensor(pixel_pitch_um, dtype=torch.float64, device=device)
    frequencies_lp_per_mm = torch.as_tensor(
        frequencies_lp_per_mm,
        dtype=torch.float64,
        device=device,
    )
    if pixel_pitch_um.shape != (psf.shape[0],):
        raise ValueError("pixel_pitch_um must have shape (design,).")
    if frequencies_lp_per_mm.ndim != 1:
        raise ValueError("frequencies_lp_per_mm must be one-dimensional.")
    if torch.any(frequencies_lp_per_mm < 0.0):
        raise ValueError("frequencies_lp_per_mm must be non-negative.")
    if frequencies_lp_per_mm.numel() > 1 and torch.any(
        frequencies_lp_per_mm[1:] < frequencies_lp_per_mm[:-1]
    ):
        raise ValueError("frequencies_lp_per_mm must be sorted.")
    nyquist_lp_per_mm = 500.0 / pixel_pitch_um
    if frequencies_lp_per_mm.numel() > 0 and frequencies_lp_per_mm[-1] > torch.min(nyquist_lp_per_mm):
        raise ValueError("requested MTF frequency exceeds the Nyquist frequency.")

    # 2. 子午和弧矢分别积分计算otf
    # S 对应像面 x 频率，T 对应像面 y 频率。
    sagittal_lsf = psf.sum(dim=-2)
    tangential_lsf = psf.sum(dim=-1)

    # 3. 计算子午和弧矢OTF
    sagittal_otf, native_frequencies = _normalized_otf(
        sagittal_lsf,
        pixel_pitch_um=pixel_pitch_um,
    )
    tangential_otf, _ = _normalized_otf(
        tangential_lsf,
        pixel_pitch_um=pixel_pitch_um,
    )

    # 4. 插值到期望的MTF频率点
    sagittal = torch.abs(
        _interpolate_complex_otf(
            sagittal_otf,
            native_frequencies=native_frequencies,
            target_frequencies=frequencies_lp_per_mm,
        )
    )
    tangential = torch.abs(
        _interpolate_complex_otf(
            tangential_otf,
            native_frequencies=native_frequencies,
            target_frequencies=frequencies_lp_per_mm,
        )
    )
    zero_frequency = frequencies_lp_per_mm == 0.0
    sagittal[..., zero_frequency] = 1.0
    tangential[..., zero_frequency] = 1.0
    return sagittal, tangential


def _normalized_otf(
    lsf: torch.Tensor,
    *,
    pixel_pitch_um: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对中心化 LSF 补零并计算归一化复 OTF。"""
    sample_count = lsf.shape[-1]
    fft_sample_count = 1 << (sample_count * 8 - 1).bit_length()
    left_padding = fft_sample_count // 2 - sample_count // 2
    right_padding = fft_sample_count - sample_count - left_padding
    padded_lsf = functional.pad(lsf, (left_padding, right_padding))
    shifted_lsf = torch.fft.ifftshift(padded_lsf, dim=-1)
    otf = torch.fft.rfft(shifted_lsf, dim=-1)
    otf = otf / otf[..., :1]

    frequency_index = torch.arange(
        otf.shape[-1],
        dtype=torch.float64,
        device=lsf.device,
    )
    native_frequencies = (
        frequency_index.reshape(1, -1)
        * 1000.0
        / (float(fft_sample_count) * pixel_pitch_um.reshape(-1, 1))
    )
    return otf, native_frequencies


def _interpolate_complex_otf(
    otf: torch.Tensor,
    *,
    native_frequencies: torch.Tensor,
    target_frequencies: torch.Tensor,
) -> torch.Tensor:
    """对复 OTF 的实部和虚部分别执行线性插值。"""
    design_count, field_count, _ = otf.shape
    targets = target_frequencies.reshape(1, -1).expand(design_count, -1).contiguous()
    upper = torch.searchsorted(native_frequencies, targets).clamp(
        min=1,
        max=native_frequencies.shape[-1] - 1,
    )
    lower = upper - 1
    lower_frequency = torch.gather(native_frequencies, -1, lower)
    upper_frequency = torch.gather(native_frequencies, -1, upper)
    fraction = (targets - lower_frequency) / (upper_frequency - lower_frequency)

    lower_index = lower[:, None, :].expand(-1, field_count, -1)
    upper_index = upper[:, None, :].expand(-1, field_count, -1)
    lower_otf = torch.gather(otf, -1, lower_index)
    upper_otf = torch.gather(otf, -1, upper_index)
    return lower_otf + (upper_otf - lower_otf) * fraction[:, None, :]


def _resolve_field_indices(
    system: MultiOpticalSystem,
    field_indices,
) -> tuple[int, ...]:
    """解析 MTF 使用的视场索引。"""
    if field_indices is None:
        return tuple(range(len(system.fields)))
    return tuple(int(field_index) for field_index in field_indices)


def _resolve_frequencies(frequencies, *, device: torch.device) -> torch.Tensor:
    """解析目标频率；空设置默认返回 0 至 300 lp/mm。"""
    if not frequencies:
        return torch.arange(301, dtype=torch.float64, device=device)
    return torch.tensor(tuple(float(value) for value in frequencies), dtype=torch.float64, device=device)


def plot_huygens_mtf(
    system: MultiOpticalSystem,
    frequencies_lp_per_mm: torch.Tensor,
    sagittal: torch.Tensor,
    tangential: torch.Tensor,
    *,
    field_indices: tuple[int, ...],
    wavelength_indices: tuple[int, ...],
    save_path: str | Path,
):
    """绘制单设计多视场 Sagittal/Tangential MTF。"""
    if system.system_count != 1:
        raise ValueError("huygens_mtf image export currently only supports a single design_view.")

    frequencies_cpu = frequencies_lp_per_mm.detach().cpu()
    sagittal_cpu = sagittal.detach().cpu()
    tangential_cpu = tangential.detach().cpu()
    figure, axis = plt.subplots(figsize=(6.2, 4.8), dpi=150)
    for local_field_index, field_index in enumerate(field_indices):
        color = MTF_FIELD_COLORS[local_field_index % len(MTF_FIELD_COLORS)]
        axis.plot(
            frequencies_cpu.numpy(),
            tangential_cpu[0, local_field_index].numpy(),
            color=color,
            linestyle="-",
            label=f"field={field_index} T",
        )
        axis.plot(
            frequencies_cpu.numpy(),
            sagittal_cpu[0, local_field_index].numpy(),
            color=color,
            linestyle="--",
            label=f"field={field_index} S",
        )

    wavelength_title = (
        f"wavelength={float(system.wavelengths[wavelength_indices[0]].value_um):.4f} um"
        if len(wavelength_indices) == 1
        else "all wavelengths"
    )
    axis.set_title(f"Huygens MTF, {wavelength_title}")
    axis.set_xlabel("Spatial Frequency (lp/mm)")
    axis.set_ylabel("Modulation")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(0.0, frequencies_cpu.numpy()[-1])
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    return figure, axis, str(output_path)
