from __future__ import annotations

import pytest
import torch

import optics_core as oc
from optics_core.huygens_psf import _image_grid, _reference_chief_points
from optics_core.rays import RayAimingResult
from optics_core.sampling import SamplingResult
from optics_core.tracing._sampled_rays import build_input_rays_from_sample
from tests.fixtures.systems import build_backward_paraxial_system, build_multifield_multistructure_system


def test_huygens_image_grid_places_origin_at_even_grid_center_index() -> None:
    """偶数像面网格的 N//2 索引应与主光线像点重合。"""
    chief_points = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float64)

    image_x, image_y, image_z = _image_grid(
        chief_points,
        image_sample_count=32,
        image_delta_um=torch.tensor([0.5], dtype=torch.float64),
        image_plane_rotation=torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
        device=chief_points.device,
    )

    torch.testing.assert_close(image_x[0, 0, 0, 16, 16], chief_points[0, 0, 0])
    torch.testing.assert_close(image_y[0, 0, 0, 16, 16], chief_points[0, 0, 1])
    torch.testing.assert_close(image_z[0, 0, 0, 16, 16], chief_points[0, 0, 2])
    assert image_x[0, 0, 0, 16, 0].item() == pytest.approx(1.0 - 8.0e-3)
    assert image_x[0, 0, 0, 16, -1].item() == pytest.approx(1.0 + 7.5e-3)


def test_huygens_image_grid_follows_local_image_plane_axes() -> None:
    """像面网格 x/y 轴应跟随 image surface 的局部 frame。"""
    chief_points = torch.zeros((1, 1, 3), dtype=torch.float64)
    rotation = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )

    image_x, image_y, image_z = _image_grid(
        chief_points,
        image_sample_count=3,
        image_delta_um=torch.tensor([1000.0], dtype=torch.float64),
        image_plane_rotation=rotation,
        device=chief_points.device,
    )

    # 局部 +x 映射到全局 +y，局部 +y 映射到全局 -x。
    torch.testing.assert_close(image_x[0, 0, 0, 1, 2], torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(image_y[0, 0, 0, 1, 2], torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(image_x[0, 0, 0, 2, 1], torch.tensor(-1.0, dtype=torch.float64))
    torch.testing.assert_close(image_y[0, 0, 0, 2, 1], torch.tensor(0.0, dtype=torch.float64))
    torch.testing.assert_close(image_z, torch.zeros_like(image_z))


def test_huygens_all_wavelengths_use_primary_chief_point_as_grid_reference() -> None:
    """全波长 PSF 应以主波长主光线像点建立共同像面网格。"""
    chief_points = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        dtype=torch.float64,
    )

    reference = _reference_chief_points(
        chief_points,
        primary_wavelength_index=1,
    )

    torch.testing.assert_close(reference, chief_points[:, :, 1])


def test_square_pupil_area_weights_reduce_boundary_cells() -> None:
    """圆周边界网格单元的积分权重应小于 pupil 内部单元。"""
    sample = oc.SquarePupilSampler(nx=32, ny=32).sample()
    coordinates = sample.pupil_coordinates[: sample.sample_ray_count]
    weights = torch.as_tensor(sample.weights, dtype=torch.float64)[: sample.sample_ray_count]
    center_index = torch.argmin(torch.sum(coordinates * coordinates, dim=-1))
    positive_boundary = weights[(weights > 0.0) & (weights < weights[center_index])]

    assert positive_boundary.numel() > 0
    assert weights.sum().item() == pytest.approx(1.0)


def test_sampled_rays_include_tilted_plane_wave_initial_opl() -> None:
    """非零视场输入平面上的 OPL 应包含倾斜平面波线性相位。"""
    system = build_backward_paraxial_system().prepare()
    sample = SamplingResult(
        pupil_coordinates=torch.tensor(((0.0, 0.0), (0.0, 0.5)), dtype=torch.float64),
        weights=torch.ones(2, dtype=torch.float64),
        pattern="fixed",
        chief_ray_index=0,
        sample_ray_count=2,
    )
    first_order = system.first_order_data

    rays = build_input_rays_from_sample(
        system,
        [system.fields[1]],
        [system.wavelengths.primary_index],
        sample,
        RayAimingResult(
            entrance_pupil_z=first_order.entrance_pupil_z,
            entrance_pupil_radius=first_order.entrance_pupil_radius,
        ),
    )

    direction_norm = torch.sqrt(rays.l * rays.l + rays.m * rays.m + rays.n * rays.n)
    expected_opl = (rays.l * rays.x + rays.m * rays.y + rays.n * rays.z) / direction_norm
    torch.testing.assert_close(rays.opl, expected_opl)
    assert not torch.allclose(rays.opl[..., 0], rays.opl[..., 1])


def test_sampled_rays_include_finite_object_spherical_wave_opl() -> None:
    """有限物距系统应从物点向入瞳发射球面波，而不是继续使用平面波。"""
    system = build_backward_paraxial_system()
    system.architecture.object_distance_mm = 100.0
    sample = SamplingResult(
        pupil_coordinates=torch.tensor(((0.0, 0.0), (0.0, 0.5)), dtype=torch.float64),
        weights=torch.ones(2, dtype=torch.float64),
        pattern="fixed",
        chief_ray_index=0,
        sample_ray_count=2,
    )

    rays = build_input_rays_from_sample(
        system,
        [system.fields[0]],
        range(len(system.wavelengths)),
        sample,
        RayAimingResult(
            entrance_pupil_z=torch.tensor([0.0], dtype=torch.float64),
            entrance_pupil_radius=torch.tensor([6.0], dtype=torch.float64),
        ),
    )

    assert rays.x.shape[2] == len(system.wavelengths)
    torch.testing.assert_close(rays.y[0, 0, 0], torch.tensor([0.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(rays.m[0, 0, 0], torch.tensor([0.0, 0.03], dtype=torch.float64))
    torch.testing.assert_close(
        rays.opl[0, 0, 0],
        torch.tensor([100.0, (100.0**2 + 3.0**2) ** 0.5], dtype=torch.float64),
    )


def test_huygens_psf_uses_primary_wavelength_by_default() -> None:
    """默认仅计算主波长 PSF。"""
    system = build_backward_paraxial_system().prepare()

    result = system.analysis.psf(
        oc.PSFSettings(
            pupil_sample_count=8,
            image_sample_count=6,
        )
    ).run()

    psf = torch.as_tensor(result.psf, dtype=torch.float64)
    strehl_ratio = torch.as_tensor(result.strehl_ratio, dtype=torch.float64)
    print(f"主波长 PSF shape: {tuple(psf.shape)}, strehl={strehl_ratio.tolist()}")

    assert tuple(psf.shape) == (system.system_count, 1, 6, 6)
    assert tuple(strehl_ratio.shape) == (system.system_count, 1)
    assert tuple(torch.as_tensor(result.psf_by_wavelength).shape) == tuple(psf.shape)
    assert tuple(torch.as_tensor(result.strehl_by_wavelength).shape) == tuple(strehl_ratio.shape)
    assert result.field_index == 0
    assert result.wavelength_indices == (system.wavelengths.primary_index,)
    expected_image_delta_um = (
        float(system.wavelengths.primary.value_um)
        * float(system.first_order_data.working_f_number[0].item())
        / (8.0**0.5)
    )
    assert torch.as_tensor(result.pixel_pitch_um)[0].item() == pytest.approx(expected_image_delta_um)
    assert torch.isfinite(psf).all()
    assert torch.max(psf).item() > 0.0
    assert torch.isfinite(strehl_ratio).all()
    torch.testing.assert_close(
        torch.amax(psf, dim=(-2, -1)),
        strehl_ratio,
        atol=1e-12,
        rtol=0.0,
    )


def test_huygens_psf_can_select_all_wavelengths() -> None:
    """wavelength_index=-1 时计算全部波长。"""
    system = build_backward_paraxial_system().prepare()

    result = system.analysis.psf(
        oc.PSFSettings(
            pupil_sample_count=6,
            image_sample_count=5,
            wavelength_index=-1,
        )
    ).run()

    psf = torch.as_tensor(result.psf, dtype=torch.float64)
    strehl_ratio = torch.as_tensor(result.strehl_ratio, dtype=torch.float64)
    print(f"全波长 PSF shape: {tuple(psf.shape)}, strehl={strehl_ratio.tolist()}")

    psf_by_wavelength = torch.as_tensor(result.psf_by_wavelength, dtype=torch.float64)
    strehl_by_wavelength = torch.as_tensor(result.strehl_by_wavelength, dtype=torch.float64)
    assert tuple(psf.shape) == (system.system_count, 1, 5, 5)
    assert tuple(strehl_ratio.shape) == (system.system_count, 1)
    assert tuple(psf_by_wavelength.shape) == (system.system_count, len(system.wavelengths), 5, 5)
    assert tuple(strehl_by_wavelength.shape) == (system.system_count, len(system.wavelengths))
    assert result.wavelength_indices == tuple(range(len(system.wavelengths)))
    expected_image_delta_um = (
        max(float(wavelength.value_um) for wavelength in system.wavelengths)
        * float(system.first_order_data.working_f_number[0].item())
        / (6.0**0.5)
    )
    assert torch.as_tensor(result.pixel_pitch_um)[0].item() == pytest.approx(expected_image_delta_um)
    assert torch.isfinite(psf).all()
    assert torch.isfinite(strehl_ratio).all()

    spectral_weights = torch.tensor(
        [
            float(wavelength.weight)
            for wavelength in system.wavelengths
        ],
        dtype=torch.float64,
    )
    spectral_weights = spectral_weights / spectral_weights.sum()
    expected_mixed_psf = torch.sum(
        psf_by_wavelength * spectral_weights.reshape(1, -1, 1, 1),
        dim=1,
        keepdim=True,
    )
    torch.testing.assert_close(psf, expected_mixed_psf)
    torch.testing.assert_close(
        torch.amax(psf, dim=(-2, -1)),
        strehl_ratio,
    )


def test_huygens_psf_can_use_explicit_image_delta() -> None:
    """非零 Image Delta 应直接覆盖自动采样间隔。"""
    system = build_backward_paraxial_system().prepare()

    result = system.analysis.psf(
        oc.PSFSettings(
            pupil_sample_count=8,
            image_sample_count=6,
            image_delta_um=0.5,
        )
    ).run()

    assert torch.as_tensor(result.pixel_pitch_um)[0].item() == pytest.approx(0.5)


def test_huygens_psf_all_wavelengths_exports_single_mixed_image(tmp_path) -> None:
    """全波长 PSF 导出时应加权混合为一张图片。"""
    system = build_backward_paraxial_system().prepare()
    output_path = tmp_path / "all_wavelengths_psf.png"

    result = system.analysis.psf(
        oc.PSFSettings(
            pupil_sample_count=6,
            image_sample_count=5,
            wavelength_index=-1,
            save_path=str(output_path),
        )
    ).run()

    assert result.figure is not None
    assert result.axes is not None
    assert result.axes.get_title().startswith("design=0, all wavelengths")
    assert result.save_path == str(output_path)
    assert output_path.exists()

    expected_mixed_psf = torch.as_tensor(result.psf, dtype=torch.float64)[0, 0]
    plotted_psf = torch.as_tensor(result.axes.images[0].get_array(), dtype=torch.float64)
    torch.testing.assert_close(plotted_psf, expected_mixed_psf)


def test_huygens_psf_multi_design_calculation_does_not_support_image_export(tmp_path) -> None:
    """多设计 PSF 支持并行计算，但图片导出要求先选择单设计视图。"""
    system = build_multifield_multistructure_system().prepare()

    result = system.analysis.psf(
        oc.PSFSettings(
            pupil_sample_count=4,
            image_sample_count=4,
            field_index=0,
        )
    ).run()
    assert torch.as_tensor(result.psf).shape[0] == system.system_count

    with pytest.raises(ValueError, match="single design_view"):
        system.analysis.psf(
            oc.PSFSettings(
                pupil_sample_count=4,
                image_sample_count=4,
                field_index=0,
                save_path=str(tmp_path / "multi_design_psf.png"),
            )
        ).run()


def test_huygens_psf_can_select_field_and_wavelength() -> None:
    """PSF 接口支持指定视场和指定波长。"""
    system = build_backward_paraxial_system().prepare()

    result = system.analysis.psf(
        oc.PSFSettings(
            pupil_sample_count=6,
            image_sample_count=5,
            field_index=1,
            wavelength_index=0,
        )
    ).run()

    psf = torch.as_tensor(result.psf, dtype=torch.float64)
    print(f"指定视场/波长 PSF shape: {tuple(psf.shape)}")

    assert tuple(psf.shape) == (system.system_count, 1, 5, 5)
    assert result.field_index == 1
    assert result.wavelength_indices == (0,)


def test_huygens_psf_requires_prepared_system() -> None:
    """PSF 依赖准备态一阶数据。"""
    system = build_backward_paraxial_system()

    with pytest.raises(ValueError, match="prepare"):
        system.analysis.psf(oc.PSFSettings(pupil_sample_count=4, image_sample_count=4)).run()
