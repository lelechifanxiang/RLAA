from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

import optics_core as oc
from optics_core.huygens_mtf import compute_huygens_mtf
from optics_core.huygens_psf import compute_huygens_psf_batch
from tests.fixtures.systems import build_backward_paraxial_system, build_multifield_multistructure_system


def _frequency_tensor(values: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _gaussian_psf(*, sigma_x: float, sigma_y: float, sample_count: int = 33) -> torch.Tensor:
    axis = torch.arange(sample_count, dtype=torch.float64) - sample_count // 2
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    psf = torch.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
    return psf.reshape(1, 1, sample_count, sample_count)


def test_huygens_mtf_impulse_is_one() -> None:
    """单像素脉冲 PSF 的全部空间频率响应应为 1。"""
    psf = torch.zeros((1, 1, 17, 17), dtype=torch.float64)
    psf[0, 0, 8, 8] = 1.0
    frequencies = _frequency_tensor((0.0, 100.0, 300.0, 500.0))

    sagittal, tangential = compute_huygens_mtf(
        psf,
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )

    torch.testing.assert_close(sagittal, torch.ones_like(sagittal))
    torch.testing.assert_close(tangential, torch.ones_like(tangential))


def test_huygens_mtf_isotropic_gaussian_has_equal_directions() -> None:
    """各向同性高斯 PSF 的 S/T MTF 应相同。"""
    frequencies = _frequency_tensor((0.0, 50.0, 100.0, 200.0))
    sagittal, tangential = compute_huygens_mtf(
        _gaussian_psf(sigma_x=2.0, sigma_y=2.0),
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )

    torch.testing.assert_close(sagittal, tangential, atol=1e-12, rtol=0.0)


def test_huygens_mtf_anisotropic_gaussian_keeps_s_t_orientation() -> None:
    """x 方向更宽的 PSF 应具有更低的 Sagittal MTF。"""
    frequencies = _frequency_tensor((0.0, 50.0, 100.0))
    sagittal, tangential = compute_huygens_mtf(
        _gaussian_psf(sigma_x=4.0, sigma_y=1.0),
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )

    assert torch.all(sagittal[..., 1:] < tangential[..., 1:])


def test_huygens_mtf_is_invariant_to_psf_intensity_scale() -> None:
    """PSF 整体强度缩放应在 DC 归一化中约掉。"""
    psf = _gaussian_psf(sigma_x=3.0, sigma_y=1.5)
    frequencies = _frequency_tensor((0.0, 40.0, 80.0, 120.0))
    base = compute_huygens_mtf(
        psf,
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )
    scaled = compute_huygens_mtf(
        psf * 17.0,
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )

    torch.testing.assert_close(base[0], scaled[0])
    torch.testing.assert_close(base[1], scaled[1])


def test_huygens_mtf_projection_matches_2d_fft_axes() -> None:
    """一维投影 FFT 应与二维 PSF FFT 的中心频率轴一致。"""
    generator = torch.Generator().manual_seed(7)
    psf = torch.rand((1, 1, 9, 9), dtype=torch.float64, generator=generator)
    fft_sample_count = 128
    pixel_pitch_um = 0.5
    native_frequencies = (
        torch.arange(fft_sample_count // 2 + 1, dtype=torch.float64)
        * 1000.0
        / (fft_sample_count * pixel_pitch_um)
    )
    frequencies = native_frequencies[:20]

    sagittal, tangential = compute_huygens_mtf(
        psf,
        pixel_pitch_um=torch.tensor([pixel_pitch_um], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )

    left_padding = fft_sample_count // 2 - psf.shape[-1] // 2
    right_padding = fft_sample_count - psf.shape[-1] - left_padding
    padded = functional.pad(psf, (left_padding, right_padding, left_padding, right_padding))
    otf_2d = torch.fft.fft2(torch.fft.ifftshift(padded, dim=(-2, -1)))
    otf_2d = otf_2d / otf_2d[..., :1, :1]
    expected_sagittal = torch.abs(otf_2d[..., 0, :20])
    expected_tangential = torch.abs(otf_2d[..., :20, 0])

    torch.testing.assert_close(sagittal, expected_sagittal, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(tangential, expected_tangential, atol=1e-12, rtol=0.0)


def test_huygens_mtf_rejects_frequency_above_nyquist() -> None:
    """请求频率超过像面采样 Nyquist 时应直接报错。"""
    with pytest.raises(ValueError, match="Nyquist"):
        compute_huygens_mtf(
            _gaussian_psf(sigma_x=2.0, sigma_y=2.0),
            pixel_pitch_um=torch.tensor([2.0], dtype=torch.float64),
            frequencies_lp_per_mm=_frequency_tensor((0.0, 251.0)),
        )


def test_polychromatic_mtf_is_computed_after_psf_mixing() -> None:
    """多色 MTF 应由混合 PSF 计算，而不是单色 MTF 模值加权平均。"""
    psf_a = _gaussian_psf(sigma_x=1.0, sigma_y=2.0)
    psf_b = torch.roll(_gaussian_psf(sigma_x=4.0, sigma_y=1.0), shifts=5, dims=-1)
    frequencies = _frequency_tensor((0.0, 50.0, 100.0, 150.0))
    weights = torch.tensor([0.4, 0.6], dtype=torch.float64)
    mixed_psf = weights[0] * psf_a + weights[1] * psf_b

    mixed_sagittal, _ = compute_huygens_mtf(
        mixed_psf,
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )
    mtf_a, _ = compute_huygens_mtf(
        psf_a,
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )
    mtf_b, _ = compute_huygens_mtf(
        psf_b,
        pixel_pitch_um=torch.tensor([0.5], dtype=torch.float64),
        frequencies_lp_per_mm=frequencies,
    )

    weighted_mtf = weights[0] * mtf_a + weights[1] * mtf_b
    assert torch.max(torch.abs(mixed_sagittal - weighted_mtf)).item() > 1e-3


def test_huygens_mtf_consumes_mixed_psf_from_shared_batch() -> None:
    """全波长分析应直接消费 PSF 内核生成的混合 PSF。"""
    system = build_backward_paraxial_system().prepare()
    frequencies = _frequency_tensor((0.0, 25.0, 50.0))
    psf_batch = compute_huygens_psf_batch(
        system,
        field_indices=(0,),
        wavelength_index=-1,
        pupil_sample_count=5,
        image_sample_count=9,
        image_delta_um=0.5,
    )
    expected_sagittal, expected_tangential = compute_huygens_mtf(
        psf_batch.psf,
        pixel_pitch_um=psf_batch.pixel_pitch_um,
        frequencies_lp_per_mm=frequencies,
    )

    result = system.analysis.mtf(
        oc.MTFSettings(
            pupil_sample_count=5,
            image_sample_count=9,
            image_delta_um=0.5,
            frequencies_lp_per_mm=tuple(frequencies.tolist()),
            field_indices=(0,),
            wavelength_index=-1,
        )
    ).run()

    torch.testing.assert_close(torch.as_tensor(result.sagittal), expected_sagittal)
    torch.testing.assert_close(torch.as_tensor(result.tangential), expected_tangential)
    assert result.wavelength_indices == tuple(range(len(system.wavelengths)))


class _CountingTracer(oc.SequentialSurfaceRayTracer):
    """记录 MTF 分析发起的显式追迹次数。"""

    def __init__(self) -> None:
        self.trace_count = 0

    def trace(
        self,
        system: oc.MultiOpticalSystem,
        rays: oc.RayBundle,
        options: oc.TraceOptions | None = None,
    ) -> oc.TraceResult:
        self.trace_count += 1
        return super().trace(system, rays, options)


def test_huygens_mtf_traces_all_fields_once() -> None:
    """多视场 MTF 应只执行一次 PSF 光线追迹。"""
    system = build_backward_paraxial_system()
    tracer = _CountingTracer()
    system.set_tracer(tracer)
    system.prepare()
    tracer.trace_count = 0

    result = system.analysis.mtf(
        oc.MTFSettings(
            pupil_sample_count=6,
            image_sample_count=9,
            image_delta_um=0.5,
            frequencies_lp_per_mm=(0.0, 50.0),
            field_indices=(0, 1),
        )
    ).run()

    assert tracer.trace_count == 1
    assert tuple(torch.as_tensor(result.sagittal).shape) == (1, 2, 2)
    assert tuple(torch.as_tensor(result.tangential).shape) == (1, 2, 2)
    assert result.field_indices == (0, 1)


def test_huygens_mtf_supports_multi_design_batch() -> None:
    """多设计和多视场应保持 batch-first 输出。"""
    system = build_multifield_multistructure_system().prepare()
    result = system.analysis.mtf(
        oc.MTFSettings(
            pupil_sample_count=4,
            image_sample_count=7,
            image_delta_um=0.5,
            frequencies_lp_per_mm=(0.0, 25.0),
            field_indices=(0, 1),
        )
    ).run()

    assert tuple(torch.as_tensor(result.sagittal).shape) == (system.system_count, 2, 2)
    assert tuple(torch.as_tensor(result.tangential).shape) == (system.system_count, 2, 2)
    torch.testing.assert_close(
        torch.as_tensor(result.sagittal)[..., 0],
        torch.ones((system.system_count, 2), dtype=torch.float64),
    )


def test_huygens_mtf_exports_single_design_curves(tmp_path) -> None:
    """单设计 MTF 应能导出全部视场的 S/T 曲线。"""
    system = build_backward_paraxial_system().prepare()
    output_path = tmp_path / "huygens_mtf.png"

    result = system.analysis.mtf(
        oc.MTFSettings(
            pupil_sample_count=4,
            image_sample_count=7,
            image_delta_um=0.5,
            frequencies_lp_per_mm=(0.0, 25.0, 50.0),
            field_indices=(0, 1),
            save_path=str(output_path),
        )
    ).run()

    assert result.figure is not None
    assert result.axes is not None
    assert len(result.axes.lines) == 4
    assert [line.get_color() for line in result.axes.lines] == [
        "#0000FF",
        "#0000FF",
        "#00AA00",
        "#00AA00",
    ]
    assert [line.get_linestyle() for line in result.axes.lines] == ["-", "--", "-", "--"]
    assert result.save_path == str(output_path)
    assert output_path.exists()
