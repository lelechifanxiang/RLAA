from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.zemax.batch_ray_trace import build_optics_core_direct_rays, build_direct_ray_set_from_zmx
from tests.zemax.common import loaded_sequential_system
from tests.zemax.spherical_forward_trace import (
    fetch_zemax_spherical_clear_apertures_from_spec,
    fetch_zemax_spherical_forward_trace_from_spec,
)
from tests.zemax.spot_diagram import fetch_zemax_standard_spot_metrics_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


DOUBLE_GAUSS_CB_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field with CB.ZMX")
COORDINATE_BREAK_SPOT_OUTPUT_PATH = Path("tests/output/spot_coordinate_break.png")
COORDINATE_BREAK_PUPIL_COORDINATES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.8, 0.0),
    (-0.8, 0.0),
    (0.0, 0.8),
    (0.0, -0.8),
    (0.55, 0.55),
    (-0.55, 0.55),
    (0.55, -0.55),
    (-0.55, -0.55),
)
COORDINATE_BREAK_NONZERO_FIELD_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, -14.0),
    (0.0, -10.0),
    (0.0, -5.0),
    (0.0, 5.0),
    (0.0, 14.0),
)
COORDINATE_BREAK_FORWARD_TRACE_ABS_TOL_MM = 2e-5
COORDINATE_BREAK_SPOT_ABS_TOL_UM = 1e-3


def test_forward_trace_with_coordinate_break_matches_zemax() -> None:
    """验证包含 Coordinate Break 的正向实光线追迹交点与 Zemax 对齐。"""

    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_CB_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec).prepare()

    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_spherical_forward_trace_from_spec(
            spec,
            oss,
            pupil_coordinates=COORDINATE_BREAK_PUPIL_COORDINATES,
            field_points=COORDINATE_BREAK_NONZERO_FIELD_POINTS,
            wavelength_indices=(spec.primary_wavelength_index + 1,),
        )
        ray_set = build_direct_ray_set_from_zmx(
            oss,
            spec,
            torch.tensor(COORDINATE_BREAK_PUPIL_COORDINATES, dtype=torch.float64),
            field_points=COORDINATE_BREAK_NONZERO_FIELD_POINTS,
            wavelength_indices=(spec.primary_wavelength_index + 1,),
        )

    rays = build_optics_core_direct_rays(system, ray_set)
    result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=True),
    )

    assert torch.all(result.valid)
    assert reference.metadata["ray_count"] == 45
    assert tuple(result.rays.x.shape) == (
        1,
        len(COORDINATE_BREAK_NONZERO_FIELD_POINTS),
        1,
        len(COORDINATE_BREAK_PUPIL_COORDINATES),
    )
    assert int(result.valid.numel()) == 45
    assert len(result.intersections) == len(reference.surface_indices)

    for surface_index, hit in enumerate(result.intersections):
        print(f"面 {surface_index} Zemax x (mm): {reference.x_mm[surface_index][0]}")
        print(f"面 {surface_index} OpticsCore x (mm): {hit.position[0].reshape(-1).tolist()[0]}")
        torch.testing.assert_close(
            hit.position[0].reshape(-1),
            torch.tensor(reference.x_mm[surface_index], dtype=torch.float64),
            atol=COORDINATE_BREAK_FORWARD_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )
        torch.testing.assert_close(
            hit.position[1].reshape(-1),
            torch.tensor(reference.y_mm[surface_index], dtype=torch.float64),
            atol=COORDINATE_BREAK_FORWARD_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )
        torch.testing.assert_close(
            hit.position[2].reshape(-1),
            torch.tensor(reference.z_mm[surface_index], dtype=torch.float64),
            atol=COORDINATE_BREAK_FORWARD_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )


@pytest.mark.parametrize("pattern", ["hexapolar", "square"])
def test_spot_diagram_with_coordinate_break_matches_zemax_standard_spot(pattern: str) -> None:
    """验证包含 Coordinate Break 的 spot diagram RMS/GEO 半径与 Zemax Standard Spot 对齐。"""

    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_CB_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    settings = oc.SpotDiagramSettings(
        pattern=pattern,
        ray_density=30,
    )

    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_standard_spot_metrics_from_spec(
            spec,
            oss,
            pattern=pattern,
            ray_density=settings.ray_density,
        )

    try:
        system.prepare()
        result = system.analysis.spot_diagram(settings).run()
    except NotImplementedError as exc:
        pytest.xfail(
            "Coordinate Break spot diagram 当前卡在入瞳计算："
            f"{exc}. 后续可考虑让 spot 复用已知入瞳数据，或让 surface_position 支持 Coordinate Break。"
        )

    assert result.rms_radius_um is not None
    assert result.geo_radius_um is not None
    assert result.field_points == tuple(reference.field_points)

    actual_rms_radius_um = torch.as_tensor(result.rms_radius_um, dtype=torch.float64).reshape(-1)
    actual_geo_radius_um = torch.as_tensor(result.geo_radius_um, dtype=torch.float64).reshape(-1)
    expected_rms_radius_um = torch.tensor(reference.rms_radius_um, dtype=torch.float64)
    expected_geo_radius_um = torch.tensor(reference.geo_radius_um, dtype=torch.float64)

    print(f"CB spot 采样方式: {pattern}")
    print(f"CB spot 视场: {result.field_points}")
    print(f"Zemax RMS 半径 (um): {reference.rms_radius_um}")
    print(f"OpticsCore RMS 半径 (um): {actual_rms_radius_um.tolist()}")
    print(f"Zemax GEO 半径 (um): {reference.geo_radius_um}")
    print(f"OpticsCore GEO 半径 (um): {actual_geo_radius_um.tolist()}")

    torch.testing.assert_close(
        actual_rms_radius_um,
        expected_rms_radius_um,
        atol=COORDINATE_BREAK_SPOT_ABS_TOL_UM,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual_geo_radius_um,
        expected_geo_radius_um,
        atol=COORDINATE_BREAK_SPOT_ABS_TOL_UM,
        rtol=0.0,
    )


def test_spot_diagram_with_coordinate_break_exports_png_and_scatter_points() -> None:
    """导出包含 Coordinate Break 的点列图，供人工和 Zemax Standard Spot 对比。"""

    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_CB_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    settings = oc.SpotDiagramSettings(
        pattern="hexapolar",
        ray_density=30,
        save_path=str(COORDINATE_BREAK_SPOT_OUTPUT_PATH),
    )

    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_standard_spot_metrics_from_spec(
            spec,
            oss,
            pattern=settings.pattern,
            ray_density=settings.ray_density,
        )

    system.prepare()
    result = system.analysis.spot_diagram(settings).run()

    assert result.rms_radius_um is not None
    assert result.geo_radius_um is not None
    assert result.figure is not None
    assert result.axes is not None
    assert result.save_path is not None
    assert Path(result.save_path).exists()
    assert Path(result.save_path).stat().st_size > 0
    assert result.scatter_points is not None
    assert "field_0" in result.scatter_points
    assert "waves" in result.scatter_points["field_0"]
    assert "wave_0" in result.scatter_points["field_0"]["waves"]
    assert len(result.scatter_points["field_0"]["waves"]["wave_0"]["x_mm"]) > 0
    assert len(result.scatter_points["field_0"]["waves"]["wave_0"]["y_mm"]) > 0

    actual_rms_radius_um = torch.as_tensor(result.rms_radius_um, dtype=torch.float64).reshape(-1)
    actual_geo_radius_um = torch.as_tensor(result.geo_radius_um, dtype=torch.float64).reshape(-1)
    print(f"CB spot 导出路径: {result.save_path}")
    print(f"CB spot 人工对比配置: pattern={settings.pattern}, ray_density={settings.ray_density}")
    print(f"CB spot 视场: {result.field_points}")
    print(f"Zemax RMS 半径 (um): {reference.rms_radius_um}")
    print(f"OpticsCore RMS 半径 (um): {actual_rms_radius_um.tolist()}")
    print(f"Zemax GEO 半径 (um): {reference.geo_radius_um}")
    print(f"OpticsCore GEO 半径 (um): {actual_geo_radius_um.tolist()}")


def test_coordinate_break_clear_apertures_match_zemax_semi_diameter() -> None:
    """验证包含 Coordinate Break 的系统中，各表面半口径配置和 Zemax SemiDiameter 一致。"""

    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_CB_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    system.prepare()

    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_spherical_clear_apertures_from_spec(spec, oss)

    my_semi_diameter = torch.tensor(
        [
            0.0 if surface.semi_diameter is None else float(surface.semi_diameter)
            for surface in system.surfaces[: len(reference.surface_indices)]
        ],
        dtype=torch.float64,
    )
    expected_semi_diameter = torch.tensor(reference.semi_diameter_mm, dtype=torch.float64)

    print(f"CB Zemax surface indices: {reference.surface_indices}")
    print(f"CB Zemax SemiDiameter (mm): {reference.semi_diameter_mm}")
    print(f"CB OpticsCore 载入半口径 (mm): {my_semi_diameter.tolist()}")

    torch.testing.assert_close(
        my_semi_diameter,
        expected_semi_diameter,
        atol=1e-1,
        rtol=0.0,
    )
