from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.fixtures import ExplicitPupilSampler
from tests.zemax.common import loaded_sequential_system
from tests.zemax.spherical_forward_trace import fetch_zemax_spherical_forward_trace_from_spec
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.regression, pytest.mark.zemax]


FOUR_SURFACE_SPHERICAL_ZMX_PATH = Path("tests/zemax/zmx_files/four_surface_spherical.zmx")
EXTREME_PUPIL_COORDINATES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)
EXTREME_TRACE_ABS_TOL_MM = 7e-4


def test_extreme_pupil_spherical_trace_matches_zemax() -> None:
    """验证中心和边缘视场的极限入瞳光线追迹。"""

    spec = load_zmx_sequential_system_spec(FOUR_SURFACE_SPHERICAL_ZMX_PATH)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_spherical_forward_trace_from_spec(
            spec,
            oss,
            pupil_coordinates=EXTREME_PUPIL_COORDINATES,
        )

    system = build_optics_core_system_from_zmx_spec(spec)
    sampler = ExplicitPupilSampler(
        pupil_coordinates=torch.tensor(EXTREME_PUPIL_COORDINATES, dtype=torch.float64),
    )

    system.prepare()
    result = system.trace(
        sampler=sampler,
        options=oc.TraceOptions(record_intersections=True),
    )

    print(f"极限 pupil 坐标: {reference.pupil_coordinates}")
    print(f"视场 (deg): {reference.field_points}")
    print(f"Zemax 第一面 x: {reference.x_mm[0]}")
    print(f"OpticsCore 第一面 x: {result.intersections[0].position[0].reshape(-1).tolist()}")

    assert torch.all(result.valid)
    assert tuple(result.rays.x.shape) == (
        1,
        len(spec.field_points),
        len(spec.wavelengths_um),
        len(EXTREME_PUPIL_COORDINATES),
    )

    for surface_index, hit in enumerate(result.intersections):
        torch.testing.assert_close(
            hit.position[0].reshape(-1),
            torch.tensor(reference.x_mm[surface_index], dtype=torch.float64),
            atol=EXTREME_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )
        torch.testing.assert_close(
            hit.position[1].reshape(-1),
            torch.tensor(reference.y_mm[surface_index], dtype=torch.float64),
            atol=EXTREME_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )
        torch.testing.assert_close(
            hit.position[2].reshape(-1),
            torch.tensor(reference.z_mm[surface_index], dtype=torch.float64),
            atol=EXTREME_TRACE_ABS_TOL_MM,
            rtol=0.0,
        )

    torch.testing.assert_close(
        result.rays.l.reshape(-1),
        torch.tensor(reference.direction_l[-1], dtype=torch.float64),
        atol=EXTREME_TRACE_ABS_TOL_MM,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.rays.m.reshape(-1),
        torch.tensor(reference.direction_m[-1], dtype=torch.float64),
        atol=EXTREME_TRACE_ABS_TOL_MM,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.rays.n.reshape(-1),
        torch.tensor(reference.direction_n[-1], dtype=torch.float64),
        atol=EXTREME_TRACE_ABS_TOL_MM,
        rtol=0.0,
    )
