from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.zemax.common import loaded_sequential_system
from tests.zemax.spherical_forward_trace import fetch_zemax_spherical_forward_trace_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


ZMX_PATH = Path("tests/zemax/zmx_files/Cooke 40 degree field.zmx")
LAYOUT_OUTPUT_PATH = Path("tests/output/layout_2d_four_surface.png")
LAYOUT_PUPIL_COORDINATES: tuple[tuple[float, float], ...] = (
    (0.0, -1.0),
    (0.0, -2.0 / 3.0),
    (0.0, -1.0 / 3.0),
    (0.0, 0.0),
    (0.0, 1.0 / 3.0),
    (0.0, 2.0 / 3.0),
    (0.0, 1.0),
)
LAYOUT_ABS_TOL_MM = 1e-6


def test_layout_2d_trace_matches_zemax_and_exports_png() -> None:
    """验证 2D layout 的 yz 截面追迹结果与 Zemax 一致，并导出图片。"""
    spec = load_zmx_sequential_system_spec(ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    expected_filtered_field_indices = tuple(
        field_index
        for field_index, (field_x_deg, _field_y_deg) in enumerate(spec.field_points)
        if field_x_deg == 0.0
    )
    expected_filtered_field_points = tuple(
        spec.field_points[field_index] for field_index in expected_filtered_field_indices
    )

    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_spherical_forward_trace_from_spec(
            spec,
            oss,
            pupil_coordinates=LAYOUT_PUPIL_COORDINATES,
            field_points=expected_filtered_field_points,
            wavelength_indices=(spec.primary_wavelength_index + 1,),
        )

    # 预先计算一阶数据以供 layout 使用
    system.prepare()

    result = system.analysis.layout_2d(
        oc.Layout2DSettings(save_path=str(LAYOUT_OUTPUT_PATH)),
    ).run()

    print(f"layout 过滤后的视场: {result.filtered_field_points}")
    print(f"layout 提示信息: {result.message}")
    print(f"layout 导出路径: {result.save_path}")

    assert result.filtered_field_indices == expected_filtered_field_indices
    assert result.filtered_field_points == expected_filtered_field_points
    assert result.trace_result is not None
    assert result.clear_aperture_result is not None
    assert result.figure is not None
    assert result.axes is not None
    assert result.save_path is not None
    assert Path(result.save_path).exists()
    assert Path(result.save_path).stat().st_size > 0

    for surface_index, hit in enumerate(result.trace_result.intersections):
        torch.testing.assert_close(
            torch.as_tensor(hit.position[1], dtype=torch.float64).reshape(-1),
            torch.tensor(reference.y_mm[surface_index], dtype=torch.float64),
            atol=LAYOUT_ABS_TOL_MM,
            rtol=0.0,
        )
        torch.testing.assert_close(
            torch.as_tensor(hit.position[2], dtype=torch.float64).reshape(-1),
            torch.tensor(reference.z_mm[surface_index], dtype=torch.float64),
            atol=LAYOUT_ABS_TOL_MM,
            rtol=0.0,
        )
