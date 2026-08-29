from __future__ import annotations

import pytest
import torch

import optics_core as oc
import optics_core.spot_diagram as spot_diagram_module
import optics_core.wavefront as wavefront_module
from optics_core.wavefront import exit_pupil_planar_opd, sample_zemax_wavefront_pupil
from tests.fixtures.systems import build_backward_paraxial_system, build_multifield_multistructure_system


def test_wavefront_uses_all_fields_and_primary_wavelength_by_default() -> None:
    """默认计算全部视场和主波长。"""
    system = build_backward_paraxial_system().prepare()

    result = system.analysis.wavefront(oc.WavefrontSettings(sample_count=8)).run()

    opd = torch.as_tensor(result.opd, dtype=torch.float64)
    rms = torch.as_tensor(result.rms_wavefront, dtype=torch.float64)
    print(f"Wavefront 默认 shape: opd={tuple(opd.shape)}, rms={tuple(rms.shape)}")

    assert tuple(opd.shape) == (system.system_count, len(system.fields), 1, 8, 8)
    assert tuple(rms.shape) == (system.system_count, len(system.fields), 1)
    assert result.field_indices == tuple(range(len(system.fields)))
    assert result.wavelength_indices == (system.wavelengths.primary_index,)
    assert result.sample_count == 8
    assert torch.isfinite(opd).all()
    assert torch.isfinite(rms).all()


def test_wavefront_can_select_multiple_fields_and_wavelengths() -> None:
    """Wavefront Map 支持多个视场、多个具体单波长并行。"""
    system = build_backward_paraxial_system().prepare()

    result = system.analysis.wavefront(
        oc.WavefrontSettings(
            field_indices=(0, 1),
            wavelength_indices=(0, 1),
            sample_count=6,
        )
    ).run()

    assert tuple(torch.as_tensor(result.opd).shape) == (system.system_count, 2, 2, 6, 6)
    assert result.field_indices == (0, 1)
    assert result.wavelength_indices == (0, 1)


def test_wavefront_rejects_all_wavelength_index() -> None:
    """Wavefront Map 不支持 wavelength_index=-1 的多色混合。"""
    system = build_backward_paraxial_system().prepare()

    with pytest.raises(ValueError, match="-1"):
        system.analysis.wavefront(oc.WavefrontSettings(wavelength_indices=(-1,))).run()


def test_wavefront_sample_count_controls_output_grid() -> None:
    """偶数采样网格应包含 Zemax 约定的 pupil 原点。"""
    sample = sample_zemax_wavefront_pupil(32)
    coordinates = torch.as_tensor(sample.pupil_coordinates, dtype=torch.float64)

    assert sample.sample_ray_count == 32 * 32
    assert sample.chief_ray_index == 32 * 32
    torch.testing.assert_close(coordinates[16 * 32 + 16], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(coordinates[sample.chief_ray_index], torch.zeros(2, dtype=torch.float64))


def test_wavefront_sets_invalid_pupil_area_to_zero_and_rms_uses_valid_points() -> None:
    """单位圆外 OPD 应填 0，RMS 只统计单位圆内采样点。"""
    system = build_backward_paraxial_system().prepare()

    result = system.analysis.wavefront(
        oc.WavefrontSettings(
            field_indices=(0,),
            wavelength_indices=(system.wavelengths.primary_index,),
            sample_count=8,
        )
    ).run()

    opd = torch.as_tensor(result.opd, dtype=torch.float64)
    pupil_x = torch.as_tensor(result.pupil_x, dtype=torch.float64)
    pupil_y = torch.as_tensor(result.pupil_y, dtype=torch.float64)
    inside = pupil_x * pupil_x + pupil_y * pupil_y <= 1.0 + 1e-12
    outside = ~inside
    valid_opd = opd[0, 0, 0][inside]
    expected_rms = torch.sqrt(torch.mean((valid_opd - valid_opd.mean()) ** 2))
    valid_mask = torch.as_tensor(result.valid_mask)[0, 0, 0]

    torch.testing.assert_close(opd[0, 0, 0][outside], torch.zeros_like(opd[0, 0, 0][outside]))
    torch.testing.assert_close(torch.as_tensor(result.rms_wavefront)[0, 0, 0], expected_rms)
    assert torch.equal(valid_mask, inside)
    assert torch.as_tensor(result.valid_count)[0, 0, 0] == inside.sum()
    assert torch.as_tensor(result.valid_fraction)[0, 0, 0] == 1.0


def test_wavefront_and_spot_report_all_invalid_samples(monkeypatch) -> None:
    """Wavefront 与 Spot 均应报告有效率，并以 NaN 表示无可用评价结果。"""
    system = build_backward_paraxial_system().prepare()
    original_wave_trace = wavefront_module.trace_pupil_to_image
    original_spot_trace = spot_diagram_module.trace_spot_diagram_rays

    def invalidate_wave(*args, **kwargs):
        result = original_wave_trace(*args, **kwargs)
        result.valid = torch.zeros_like(result.valid)
        return result

    def invalidate_spot(*args, **kwargs):
        result = original_spot_trace(*args, **kwargs)
        result.valid = torch.zeros_like(result.valid)
        return result

    monkeypatch.setattr(wavefront_module, "trace_pupil_to_image", invalidate_wave)
    wavefront = system.analysis.wavefront(
        oc.WavefrontSettings(field_indices=(0,), sample_count=4)
    ).run()
    assert not torch.as_tensor(wavefront.valid_mask).any()
    assert not torch.as_tensor(wavefront.valid_count).any()
    assert not torch.as_tensor(wavefront.valid_fraction).any()
    assert torch.isnan(torch.as_tensor(wavefront.rms_wavefront)).all()

    monkeypatch.setattr(spot_diagram_module, "trace_spot_diagram_rays", invalidate_spot)
    spot = system.analysis.spot_diagram(oc.SpotDiagramSettings(pattern="hexapolar", ray_density=3)).run()
    assert not torch.as_tensor(spot.valid_count).any()
    assert not torch.as_tensor(spot.valid_fraction).any()
    assert torch.isnan(torch.as_tensor(spot.rms_radius_um)).all()
    assert torch.isnan(torch.as_tensor(spot.geo_radius_um)).all()


def test_afocal_opd_uses_planar_exit_pupil_reference() -> None:
    """平行输出光线反向到出瞳参考平面后应得到零波前误差。"""
    image_points = torch.tensor(
        [[[[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]]]],
        dtype=torch.float64,
    )
    ray_directions = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]],
        dtype=torch.float64,
    )

    exit_points, opd = exit_pupil_planar_opd(
        image_points=image_points,
        ray_directions=ray_directions,
        opl=torch.full((1, 1, 2), 10.0, dtype=torch.float64),
        chief_points=image_points[:, :, 0],
        chief_directions=ray_directions[:, :, 0],
        exit_pupil_z=torch.zeros((1, 1), dtype=torch.float64),
        valid_points=torch.ones((1, 1, 2), dtype=torch.bool),
    )

    torch.testing.assert_close(exit_points[..., 2], torch.zeros((1, 1, 2), dtype=torch.float64))
    torch.testing.assert_close(opd, torch.zeros_like(opd))


def test_wavefront_multi_design_calculation_does_not_export_image(tmp_path) -> None:
    """多设计 Wavefront Map 可计算，但图片导出要求单设计。"""
    system = build_multifield_multistructure_system().prepare()

    result = system.analysis.wavefront(oc.WavefrontSettings(field_indices=(0,), sample_count=4)).run()
    assert torch.as_tensor(result.opd).shape[0] == system.system_count

    with pytest.raises(ValueError, match="single design"):
        system.analysis.wavefront(
            oc.WavefrontSettings(
                field_indices=(0,),
                sample_count=4,
                save_path=str(tmp_path / "multi_design_wavefront.png"),
            )
        ).run()


def test_wavefront_exports_single_image(tmp_path) -> None:
    """单设计、单视场、单波长支持导出 Wavefront Map 图片。"""
    system = build_backward_paraxial_system().prepare()
    output_path = tmp_path / "wavefront.png"

    result = system.analysis.wavefront(
        oc.WavefrontSettings(
            field_indices=(0,),
            wavelength_indices=(system.wavelengths.primary_index,),
            sample_count=8,
            save_path=str(output_path),
        )
    ).run()

    assert result.figure is not None
    assert result.axes is not None
    assert result.save_path == str(output_path)
    assert output_path.exists()
