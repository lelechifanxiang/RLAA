from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.zemax.common import loaded_sequential_system
from tests.zemax.spot_diagram import fetch_zemax_standard_spot_metrics_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


DOUBLE_GAUSS_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx")
SPOT_DIAG_OUTPUT_PATH = Path("tests/output/spot_double_gauss.png")
SPOT_DIAGRAM_ABS_TOL_UM = 1e-3


@pytest.mark.parametrize(
    ("pattern", "abs_tol_um"),
    [
        ("square", SPOT_DIAGRAM_ABS_TOL_UM),
        ("hexapolar", SPOT_DIAGRAM_ABS_TOL_UM),
    ],
)
def test_spot_diagram_metrics_match_zemax_standard_spot(pattern: str, abs_tol_um: float) -> None:
    """验证 spot diagram 的每个视场 RMS/GEO 半径与 Zemax Standard Spot 对齐。"""
    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
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

    system.prepare()
    result = system.analysis.spot_diagram(settings).run()

    assert result.rms_radius_um is not None
    assert result.geo_radius_um is not None
    assert result.field_points == tuple(reference.field_points)

    actual_rms_radius_um = torch.as_tensor(result.rms_radius_um, dtype=torch.float64).reshape(-1)
    actual_geo_radius_um = torch.as_tensor(result.geo_radius_um, dtype=torch.float64).reshape(-1)
    expected_rms_radius_um = torch.tensor(reference.rms_radius_um, dtype=torch.float64)
    expected_geo_radius_um = torch.tensor(reference.geo_radius_um, dtype=torch.float64)

    print(f"spot 采样方式: {pattern}")
    print(f"spot 视场: {result.field_points}")
    print(f"Zemax RMS 半径 (um): {reference.rms_radius_um}")
    print(f"OpticsCore RMS 半径 (um): {actual_rms_radius_um.tolist()}")
    print(f"Zemax GEO 半径 (um): {reference.geo_radius_um}")
    print(f"OpticsCore GEO 半径 (um): {actual_geo_radius_um.tolist()}")

    torch.testing.assert_close(
        actual_rms_radius_um,
        expected_rms_radius_um,
        atol=abs_tol_um,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual_geo_radius_um,
        expected_geo_radius_um,
        atol=abs_tol_um,
        rtol=0.0,
    )


def test_spot_diagram_exports_png_and_scatter_points(tmp_path: Path) -> None:
    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    settings = oc.SpotDiagramSettings(
        pattern="hexapolar",
        ray_density=30,
        save_path=str(SPOT_DIAG_OUTPUT_PATH),
    )

    system.prepare()
    result = system.analysis.spot_diagram(settings).run()

    print(f"spot 导出路径: {result.save_path}")
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
