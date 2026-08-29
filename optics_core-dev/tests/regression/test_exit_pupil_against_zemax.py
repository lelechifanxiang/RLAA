from __future__ import annotations

from pathlib import Path

import pytest

from tests.zemax.common import loaded_sequential_system
from tests.zemax.first_order import fetch_zemax_first_order_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


EXIT_PUPIL_ZMX_PATHS = (
    Path("tests/zemax/zmx_files/paraxial_single_lens.zmx"),
    Path("tests/zemax/zmx_files/four_surface_spherical.zmx"),
    Path("tests/zemax/zmx_files/Double Gauss 28 degree field.zmx"),
)
EXIT_PUPIL_ABS_TOL_MM = 1e-4


@pytest.mark.parametrize("zmx_path", EXIT_PUPIL_ZMX_PATHS)
def test_exit_pupil_matches_zemax_first_order_reference(zmx_path: Path) -> None:
    """验证 prepare 阶段缓存的出瞳位置和半径与 Zemax 一阶量对齐。"""

    spec = load_zmx_sequential_system_spec(zmx_path)
    system = build_optics_core_system_from_zmx_spec(spec)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_first_order_from_spec(spec, oss)

    system.prepare()
    first_order = system.first_order_data
    assert first_order is not None

    # Zemax EXPP 是相对像面的出瞳距；内部缓存使用全局 z 位置。
    actual_exit_pupil_distance_mm = first_order.exit_pupil_z - first_order.ttl

    print(f"zmx 文件: {zmx_path}")
    print(f"Zemax EXPP (mm): {reference.exit_pupil_z_mm:.12f}")
    print(f"OpticsCore EXPP (mm): {actual_exit_pupil_distance_mm.item():.12f}")
    print(f"Zemax EXPR (mm): {reference.exit_pupil_radius_mm:.12f}")
    print(f"OpticsCore EXPR (mm): {first_order.exit_pupil_radius.item():.12f}")
    print(f"OpticsCore stop 半径 (mm): {first_order.stop_radius.item():.12f}")

    assert actual_exit_pupil_distance_mm.item() == pytest.approx(
        reference.exit_pupil_z_mm,
        abs=EXIT_PUPIL_ABS_TOL_MM,
    )
    assert first_order.exit_pupil_radius.item() == pytest.approx(
        reference.exit_pupil_radius_mm,
        abs=EXIT_PUPIL_ABS_TOL_MM,
    )

