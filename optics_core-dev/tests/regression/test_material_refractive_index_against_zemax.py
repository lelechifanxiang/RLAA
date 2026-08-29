from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.zemax.common import loaded_sequential_system
from tests.zemax.multispectral_sphere_material import fetch_zemax_surface_refractive_indices_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


BK7_SINGLE_SPHERE_ZMX_PATH = Path("tests/zemax/zmx_files/single_sphere_material_reference.zmx")
K9_SINGLE_SPHERE_ZMX_PATH = Path("tests/zemax/zmx_files/single_sphere_k9.zmx")
REFRACTIVE_INDEX_ABS_TOL = 1e-4


def _assert_surface_refractive_indices_against_zemax(
    zmx_path: Path,
    *,
    use_real_materials: bool,
    expected_material_type: type[oc.Material],
) -> None:
    # 读取 zmx 规格，并在 optics_core 中重建同一设计
    spec = load_zmx_sequential_system_spec(zmx_path)
    system = build_optics_core_system_from_zmx_spec(spec, use_real_materials=use_real_materials)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_surface_refractive_indices_from_spec(spec, oss)

    assert len(spec.surfaces) == 1
    lens_surface = system.surfaces[0]
    lens_material = lens_surface.gap.medium
    assert isinstance(lens_surface.geometry, oc.StandardGeometry)
    assert isinstance(lens_material, expected_material_type)

    # 计算 optics_core 在同一组波长下的折射率
    wavelength_values = torch.tensor(
        [wavelength.value_um for wavelength in system.wavelengths],
        dtype=torch.float64,
    )
    actual_indices = lens_material.refractive_index(wavelength_values)

    assert wavelength_values.tolist() == pytest.approx(reference.wavelengths_um, abs=1e-12)
    print(f"zmx 文件: {zmx_path}")
    print(f"材料名称: {lens_material.name}")
    print(f"表面半径 (mm): {float(lens_surface.geometry.radius)}")
    print(f"测试系统波长 (um): {wavelength_values.tolist()}")
    print(f"Zemax 参考折射率: {reference.refractive_indices}")
    print(f"OpticsCore 计算折射率: {actual_indices.tolist()}")
    torch.testing.assert_close(
        actual_indices,
        torch.tensor(reference.refractive_indices, dtype=torch.float64),
        atol=REFRACTIVE_INDEX_ABS_TOL,
        rtol=0.0,
    )


def test_abbe_material_refractive_indices_match_zemax_reference() -> None:
    """验证从 zmx 规格重建的 Abbe 材料折射率与 Zemax 对齐。"""

    _assert_surface_refractive_indices_against_zemax(
        BK7_SINGLE_SPHERE_ZMX_PATH,
        use_real_materials=False,
        expected_material_type=oc.AbbeModelMaterial,
    )


def test_real_material_refractive_indices_match_zemax_reference() -> None:
    """验证从 zmx 规格重建的真实材料折射率与 Zemax 对齐。"""

    _assert_surface_refractive_indices_against_zemax(
        K9_SINGLE_SPHERE_ZMX_PATH,
        use_real_materials=True,
        expected_material_type=oc.RealMaterial,
    )
