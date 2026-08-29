from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.zemax.common import loaded_sequential_system
from tests.zemax.multispectral_sphere_material import fetch_zemax_surface_sag_reference_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


SINGLE_SPHERE_ZMX_PATH = Path("tests/zemax/zmx_files/single_sphere_material_reference.zmx")
SAG_ABS_TOL_MM = 1e-6
SAG_TEST_POINTS_MM: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (3.0, 4.0),
    (6.0, 0.0),
    (0.0, 7.5),
    (5.0, -5.0),
)


@pytest.mark.parametrize(
    "sag_sample_xy_mm",
    SAG_TEST_POINTS_MM,
    ids=[f"x{point[0]:g}_y{point[1]:g}" for point in SAG_TEST_POINTS_MM],
)
def test_single_sphere_sag_matches_zemax_reference(
    sag_sample_xy_mm: tuple[float, float],
) -> None:
    """验证从 zmx 规格重建的单球面矢高与 Zemax 对齐。"""

    # 读取 zmx 规格，并在 optics_core 中重建同一设计
    spec = load_zmx_sequential_system_spec(SINGLE_SPHERE_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_surface_sag_reference_from_spec(
            spec,
            oss,
            sag_sample_xy_mm=sag_sample_xy_mm,
        )

    assert len(spec.surfaces) == 1
    lens_surface = system.surfaces[0]
    lens_material = lens_surface.gap.medium
    assert isinstance(lens_surface.geometry, oc.StandardGeometry)
    assert isinstance(lens_material, oc.AbbeModelMaterial)

    # 计算 optics_core 在同一点坐标下的球面矢高
    x, y = sag_sample_xy_mm
    actual_sag_mm = lens_surface.geometry.sag(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
    )
    actual_sag_value = float(actual_sag_mm.item())

    print(f"zmx 文件: {SINGLE_SPHERE_ZMX_PATH}")
    print(f"材料名称: {lens_material.name}")
    print(f"表面半径 (mm): {float(lens_surface.geometry.radius)}")
    print(f"测试点坐标 (mm): ({x}, {y})")
    print(f"Zemax 参考矢高 (mm): {reference.surface_sag_mm:.6f}")
    print(f"OpticsCore 计算矢高 (mm): {actual_sag_value:.6f}")
    assert actual_sag_value == pytest.approx(reference.surface_sag_mm, abs=SAG_ABS_TOL_MM)
