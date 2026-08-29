from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.fixtures import ExplicitPupilSampler
from tests.zemax.batch_ray_trace import (
    build_optics_core_direct_rays,
    build_direct_ray_set_from_zmx,
    sample_unit_disk_pupil_coordinates,
    trace_optics_core_direct,
    trace_zemax_batch_direct,
)
from tests.zemax.common import loaded_sequential_system
from tests.zemax.first_order import fetch_zemax_first_order_from_spec
from tests.zemax.opd import fetch_zemax_chief_referenced_opd
from tests.zemax.spherical_forward_trace import fetch_zemax_spherical_forward_trace_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec
from optics_core.wavefront import exit_pupil_referenced_opd, normalized
from optics_core.rays import RayAimingResult
from optics_core.sampling import SamplingResult
from optics_core.tracing._sampled_rays import build_input_rays_from_sample


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


DOUBLE_GAUSS_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx")
NEGATIVE_THICKNESS_ZMX_PATH = Path(
    "tests/zemax/zmx_files/Cooke 40 degree field negative thickness.zmx"
)
FORWARD_TRACE_PUPIL_COORDINATES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.25, -0.2),
    (-0.3, 0.15),
)

FORWARD_TRACE_ABS_TOL_MM = 1.5e-4
DOUBLE_GAUSS_FIRST_ORDER_EFFL_ABS_TOL_MM = 1e-3
DOUBLE_GAUSS_FIRST_ORDER_ENPP_ABS_TOL_MM = 1e-3
DOUBLE_GAUSS_FIRST_ORDER_ENPR_ABS_TOL_MM = 1e-3
DOUBLE_GAUSS_OPD_ABS_TOL_WAVES = 1e-9


def test_forward_multi_sphere_trace_matches_zemax() -> None:
    """用 Zemax 直接结果验证正向多球面、多视场、多波长追迹。"""
    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_spherical_forward_trace_from_spec(
            spec,
            oss,
            pupil_coordinates=FORWARD_TRACE_PUPIL_COORDINATES,
        )

    sampler = ExplicitPupilSampler(
        pupil_coordinates=torch.tensor(FORWARD_TRACE_PUPIL_COORDINATES, dtype=torch.float64),
    )
    system.prepare()
    result = system.trace(
        sampler=sampler,
        options=oc.TraceOptions(record_intersections=True),
    )

    print(f"Zemax surface indices: {reference.surface_indices}")
    print(f"波长 (um): {reference.wavelengths_um}")
    print(f"视场 (deg): {reference.field_points}")
    print(f"瞳坐标: {reference.pupil_coordinates}")
    print(f"Zemax 像面 x: {reference.x_mm[-1]}")
    print(f"OpticsCore 像面 x: {result.rays.x.reshape(-1).tolist()}")

    # 验证各折射面的折射率
    for surface_index, surface in enumerate(system.surfaces[:-1]):
        medium = surface.gap.medium
        if medium is None:
            continue
        actual_indices = medium.refractive_index(torch.tensor(spec.wavelengths_um, dtype=torch.float64))
        zemax_surface_index = reference.surface_indices[surface_index]
        zemax_indices = torch.tensor(
            reference.refractive_indices_by_surface[zemax_surface_index],
            dtype=torch.float64,
        )
        print(f"面 {surface_index} Zemax 折射率: {zemax_indices.tolist()}")
        print(f"面 {surface_index} OpticsCore 折射率: {actual_indices.tolist()}")
        torch.testing.assert_close(actual_indices, zemax_indices, atol=6e-5, rtol=0.0)

    # 验证各面的交点
    assert torch.all(result.valid)
    assert tuple(result.rays.x.shape) == (
        1,
        len(spec.field_points),
        len(spec.wavelengths_um),
        len(FORWARD_TRACE_PUPIL_COORDINATES),
    )
    assert len(result.intersections) == len(reference.surface_indices)

    for surface_index, hit in enumerate(result.intersections):
        torch.testing.assert_close(
            hit.position[0].reshape(-1),
            torch.tensor(reference.x_mm[surface_index], dtype=torch.float64),
            atol=FORWARD_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )
        torch.testing.assert_close(
            hit.position[1].reshape(-1),
            torch.tensor(reference.y_mm[surface_index], dtype=torch.float64),
            atol=FORWARD_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )
        torch.testing.assert_close(
            hit.position[2].reshape(-1),
            torch.tensor(reference.z_mm[surface_index], dtype=torch.float64),
            atol=FORWARD_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )

    # 验证像面方向余弦
    torch.testing.assert_close(
        result.rays.l.reshape(-1),
        torch.tensor(reference.direction_l[-1], dtype=torch.float64),
        atol=FORWARD_TRACE_ABS_TOL_MM,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.rays.m.reshape(-1),
        torch.tensor(reference.direction_m[-1], dtype=torch.float64),
        atol=FORWARD_TRACE_ABS_TOL_MM,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.rays.n.reshape(-1),
        torch.tensor(reference.direction_n[-1], dtype=torch.float64),
        atol=FORWARD_TRACE_ABS_TOL_MM,
        rtol=0.0,
    )


def test_negative_thickness_forward_trace_matches_zemax() -> None:
    """验证普通顺序面之间的负厚度传播与 Zemax 一致。"""
    spec = load_zmx_sequential_system_spec(NEGATIVE_THICKNESS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    pupil_coordinates = torch.tensor(FORWARD_TRACE_PUPIL_COORDINATES, dtype=torch.float64)
    field_points = (spec.field_points[0], spec.field_points[1])
    wavelength_indices = (spec.primary_wavelength_index + 1,)

    with loaded_sequential_system(spec.zmx_path) as oss:
        ray_set = build_direct_ray_set_from_zmx(
            oss,
            spec,
            pupil_coordinates,
            field_points=field_points,
            wavelength_indices=wavelength_indices,
        )
        zemax_result = trace_zemax_batch_direct(oss, spec, ray_set)

    optics_core_result = trace_optics_core_direct(system, ray_set)

    print(f"负厚度 Zemax valid={zemax_result.valid.reshape(-1).tolist()}")
    print(f"负厚度 OpticsCore valid={optics_core_result.valid.reshape(-1).tolist()}")
    print(f"负厚度 Zemax 像面 x={zemax_result.x_mm.reshape(-1).tolist()}")
    print(f"负厚度 OpticsCore 像面 x={optics_core_result.rays.x.reshape(-1).tolist()}")

    assert torch.all(zemax_result.valid)
    assert torch.equal(optics_core_result.valid, zemax_result.valid)
    torch.testing.assert_close(optics_core_result.rays.x, zemax_result.x_mm, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(optics_core_result.rays.y, zemax_result.y_mm, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(
        optics_core_result.rays.l,
        zemax_result.direction_l,
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.m,
        zemax_result.direction_m,
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.n,
        zemax_result.direction_n,
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("field_index", (0, 1), ids=("on-axis", "off-axis"))
def test_double_gauss_exit_pupil_opd_matches_zemax_at_selected_pupil_points(field_index: int) -> None:
    """检查轴上和离轴 pupil 点的出瞳参考 OPD 与 Zemax 一致。"""
    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec).prepare()
    coordinates = torch.tensor(FORWARD_TRACE_PUPIL_COORDINATES, dtype=torch.float64)
    wavelength_index = spec.primary_wavelength_index

    with loaded_sequential_system(spec.zmx_path) as oss:
        zemax_opd_waves = fetch_zemax_chief_referenced_opd(
            spec,
            oss,
            coordinates,
            field_index=field_index,
            wavelength_index=wavelength_index,
        )

    sample = SamplingResult(
        pupil_coordinates=coordinates,
        weights=torch.ones(coordinates.shape[0], dtype=torch.float64),
        pattern="fixed",
        chief_ray_index=0,
        sample_ray_count=coordinates.shape[0],
    )
    first_order = system.first_order_data
    rays = build_input_rays_from_sample(
        system,
        [system.fields[field_index]],
        [wavelength_index],
        sample,
        RayAimingResult(
            entrance_pupil_z=first_order.entrance_pupil_z,
            entrance_pupil_radius=first_order.entrance_pupil_radius,
        ),
    )
    trace_result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=False),
    )

    image_points = torch.stack(
        (
            trace_result.rays.x[:, 0],
            trace_result.rays.y[:, 0],
            trace_result.rays.z[:, 0],
        ),
        dim=-1,
    )
    ray_directions = normalized(
        torch.stack(
            (
                trace_result.rays.l[:, 0],
                trace_result.rays.m[:, 0],
                trace_result.rays.n[:, 0],
            ),
            dim=-1,
        )
    )
    _, optics_core_opd_mm = exit_pupil_referenced_opd(
        image_points=image_points,
        ray_directions=ray_directions,
        opl=trace_result.rays.opl[:, 0],
        chief_points=image_points[:, :, 0],
        exit_pupil_z=first_order.exit_pupil_z.reshape(system.system_count, 1),
        valid_points=trace_result.valid[:, 0],
    )
    wavelength_mm = spec.wavelengths_um[wavelength_index] * 1e-3
    optics_core_opd_waves = (
        optics_core_opd_mm[0, 0] - optics_core_opd_mm[0, 0, 0]
    ) / wavelength_mm

    print(f"OPD 视场索引: {field_index}")
    print(f"OPD pupil 坐标: {coordinates.tolist()}")
    print(f"Zemax OPD (waves): {zemax_opd_waves.tolist()}")
    print(f"OpticsCore OPD (waves): {optics_core_opd_waves.tolist()}")

    abs_tol_waves = DOUBLE_GAUSS_OPD_ABS_TOL_WAVES if field_index == 0 else 5e-8
    torch.testing.assert_close(
        optics_core_opd_waves,
        zemax_opd_waves,
        atol=abs_tol_waves,
        rtol=0.0,
    )


def test_double_gauss_zmx_forward_trace_matches_zemax_direct_rays() -> None:
    """验证 Double Gauss zmx 导入后的 direct ray trace 与 Zemax 严格对齐。"""

    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    pupil_coordinates = sample_unit_disk_pupil_coordinates(32, seed=0)

    with loaded_sequential_system(spec.zmx_path) as oss:
        ray_set = build_direct_ray_set_from_zmx(oss, spec, pupil_coordinates)
        zemax_result = trace_zemax_batch_direct(oss, spec, ray_set)

    optics_core_result = trace_optics_core_direct(system, ray_set)
    common_valid_mask = zemax_result.valid & optics_core_result.valid

    max_x_error = torch.max(
        torch.abs(optics_core_result.rays.x[common_valid_mask] - zemax_result.x_mm[common_valid_mask])
    ).item()
    max_y_error = torch.max(
        torch.abs(optics_core_result.rays.y[common_valid_mask] - zemax_result.y_mm[common_valid_mask])
    ).item()
    max_l_error = torch.max(
        torch.abs(optics_core_result.rays.l[common_valid_mask] - zemax_result.direction_l[common_valid_mask])
    ).item()
    max_m_error = torch.max(
        torch.abs(optics_core_result.rays.m[common_valid_mask] - zemax_result.direction_m[common_valid_mask])
    ).item()
    max_n_error = torch.max(
        torch.abs(optics_core_result.rays.n[common_valid_mask] - zemax_result.direction_n[common_valid_mask])
    ).item()

    print(f"Double Gauss common_valid_rays={int(common_valid_mask.sum().item())}")
    print(f"Double Gauss max_x_error_mm={max_x_error:.12e}")
    print(f"Double Gauss max_y_error_mm={max_y_error:.12e}")
    print(f"Double Gauss max_l_error={max_l_error:.12e}")
    print(f"Double Gauss max_m_error={max_m_error:.12e}")
    print(f"Double Gauss max_n_error={max_n_error:.12e}")

    assert torch.equal(optics_core_result.valid, zemax_result.valid)
    assert int(common_valid_mask.sum().item()) == len(spec.field_points) * len(spec.wavelengths_um) * 32

    torch.testing.assert_close(
        optics_core_result.rays.x[common_valid_mask],
        zemax_result.x_mm[common_valid_mask],
        atol=1e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.y[common_valid_mask],
        zemax_result.y_mm[common_valid_mask],
        atol=1e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.l[common_valid_mask],
        zemax_result.direction_l[common_valid_mask],
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.m[common_valid_mask],
        zemax_result.direction_m[common_valid_mask],
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.n[common_valid_mask],
        zemax_result.direction_n[common_valid_mask],
        atol=1e-12,
        rtol=0.0,
    )


def test_double_gauss_surface_history_matches_zemax() -> None:
    """验证一次追迹记录的多个表面出射状态与 Zemax 一致。"""
    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    selected = (0, len(spec.surfaces) // 2, len(spec.surfaces) - 1)

    with loaded_sequential_system(spec.zmx_path) as oss:
        ray_set = build_direct_ray_set_from_zmx(
            oss,
            spec,
            sample_unit_disk_pupil_coordinates(8, seed=0),
        )
        zemax_results = [
            trace_zemax_batch_direct(oss, spec, ray_set, end_surface=surface_index + 1)
            for surface_index in selected
        ]

    result = trace_optics_core_direct(
        system,
        ray_set,
        options=oc.TraceOptions(record_intersections=False, record_surface_states=selected),
    )
    history = result.surface_history
    assert history is not None

    vertex_z = 0.0
    vertex_z_by_surface = []
    for surface in spec.surfaces:
        vertex_z_by_surface.append(vertex_z)
        vertex_z += surface.thickness_mm

    for history_index, (surface_index, zemax) in enumerate(zip(selected, zemax_results, strict=True)):
        assert torch.equal(history.valid[:, history_index], zemax.valid)
        valid = history.valid[:, history_index]
        expected_z = zemax.z_mm + vertex_z_by_surface[surface_index]
        print(f"surface={surface_index}, valid_ray_count={int(valid.sum().item())}")
        for name, expected in (
            ("x", zemax.x_mm),
            ("y", zemax.y_mm),
            ("z", expected_z),
            ("l", zemax.direction_l),
            ("m", zemax.direction_m),
            ("n", zemax.direction_n),
        ):
            atol = 1e-10 if name in ("x", "y", "z") else 1e-12
            torch.testing.assert_close(
                getattr(history, name)[:, history_index][valid],
                expected[valid],
                atol=atol,
                rtol=0.0,
            )


def test_double_gauss_first_order_matches_zemax() -> None:
    """验证 Double Gauss 的一阶焦距和入瞳位置与 Zemax 对齐。"""

    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_first_order_from_spec(spec, oss)

    system.prepare()
    result = system.analysis.first_order().run()

    print(f"Double Gauss Zemax EFFL={reference.effective_focal_length_mm:.12f} mm")
    print(f"Double Gauss OpticsCore EFFL={result.effl.item():.12f} mm")
    print(f"Double Gauss Zemax ENPP={reference.entrance_pupil_z_mm:.12f} mm")
    print(f"Double Gauss OpticsCore ENPP={result.entrance_pupil_z.item():.12f} mm")
    print(f"Double Gauss Zemax ENPR={reference.entrance_pupil_radius_mm:.12f} mm")
    print(f"Double Gauss OpticsCore ENPR={result.entrance_pupil_radius.item():.12f} mm")
    print(f"Double Gauss TTL={result.ttl.item():.12f} mm")

    assert result.ttl.item() == pytest.approx(reference.total_track_length_mm, abs=5e-5)
    assert result.effl.item() == pytest.approx(
        reference.effective_focal_length_mm,
        abs=DOUBLE_GAUSS_FIRST_ORDER_EFFL_ABS_TOL_MM,
    )
    assert result.entrance_pupil_z.item() == pytest.approx(
        reference.entrance_pupil_z_mm,
        abs=DOUBLE_GAUSS_FIRST_ORDER_ENPP_ABS_TOL_MM,
    )
    assert result.entrance_pupil_radius.item() == pytest.approx(
        reference.entrance_pupil_radius_mm,
        abs=DOUBLE_GAUSS_FIRST_ORDER_ENPR_ABS_TOL_MM,
    )
