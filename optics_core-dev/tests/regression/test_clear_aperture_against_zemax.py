from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.fixtures import ExplicitPupilSampler
from tests.zemax.common import loaded_sequential_system
from tests.zemax.spherical_forward_trace import fetch_zemax_spherical_clear_apertures_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


FOUR_SURFACE_SPHERICAL_ZMX_PATH = Path("tests/zemax/zmx_files/four_surface_spherical.zmx")
EXTREME_PUPIL_COORDINATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)


def test_calculate_clear_apertures_matches_zemax_extreme_pupil_reference() -> None:
    """验证需求半口径计算结果不超过 Zemax 直接配置的 SemiDiameter。"""

    spec = load_zmx_sequential_system_spec(FOUR_SURFACE_SPHERICAL_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    system.prepare()
    sampler = ExplicitPupilSampler(
        pupil_coordinates=torch.tensor(EXTREME_PUPIL_COORDINATES, dtype=torch.float64),
    )
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_spherical_clear_apertures_from_spec(spec, oss)

    result = oc.calculate_clear_apertures(
        system,
        sampler=sampler,
        keep_trace_result=True,
    )

    print(f"Zemax surface indices: {reference.surface_indices}")
    print(f"OpticsCore surface indices: {result.surface_indices}")
    print(f"Zemax SemiDiameter (mm): {reference.semi_diameter_mm}")
    print(f"OpticsCore 需求半口径 (mm): {result.semi_diameter.reshape(-1).tolist()}")

    assert tuple(result.semi_diameter.shape) == (1, len(spec.surfaces))
    assert tuple(result.valid.shape) == (1, len(spec.surfaces))
    assert result.surface_indices == tuple(range(len(spec.surfaces)))
    assert torch.all(result.valid)
    assert result.trace_result is not None

    expected_semi_diameter = torch.tensor(reference.semi_diameter_mm, dtype=torch.float64)
    assert torch.all(
        result.semi_diameter[0] <= expected_semi_diameter + 1e-12
    )

    # prepare 阶段会把 auto 面回填为当前系统的需求半口径。
    assert system.clear_aperture_data is not None
    assert [
        surface.semi_diameter_solve
        for surface in system.surfaces[: len(spec.surfaces)]
    ] == ["auto"] * len(spec.surfaces)
    torch.testing.assert_close(
        torch.tensor(
            [float(surface.semi_diameter) for surface in system.surfaces[: len(spec.surfaces)]],
            dtype=torch.float64,
        ),
        system.clear_aperture_data.semi_diameter[0],
        atol=1e-12,
        rtol=0.0,
    )
