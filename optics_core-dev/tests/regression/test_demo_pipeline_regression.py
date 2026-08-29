from __future__ import annotations

import pytest
import torch

import optics_core as oc
from tests.support import build_showcase_system


pytestmark = pytest.mark.regression


def test_showcase_surface_layout_regression() -> None:
    showcase_system = build_showcase_system()

    assert showcase_system.name == "showcase"
    assert [surface.label for surface in showcase_system.surfaces] == ["OBJ", "S1", "S2", "PX", "CB", "GEN", "IMG"]
    assert [type(surface).__name__ for surface in showcase_system.surfaces] == [
        "ObjectSurface",
        "SphereSurface",
        "SphereSurface",
        "ParaxialSurface",
        "CoordinateBreak",
        "SphereSurface",
        "ImageSurface",
    ]
    assert showcase_system.surfaces.stop_index == 1
    assert showcase_system.aperture is not None
    assert showcase_system.aperture.kind == "entrance_pupil_diameter"
    assert showcase_system.aperture.value == 12.0


def test_showcase_trace_regression() -> None:
    showcase_system = build_showcase_system()
    showcase_system.prepare()
    result = showcase_system.trace(
        sampler=oc.SquarePupilSampler(nx=2, ny=2),
        options=oc.TraceOptions(record_intersections=True, record_ray_angles=True),
    )

    assert tuple(result.rays.x.shape) == (1, 2, 2, 5)
    assert tuple(result.valid.shape) == (1, 2, 2, 5)
    assert torch.all(result.valid)
    assert [hit.surface_index for hit in result.intersections] == list(range(len(showcase_system.surfaces)))
    assert result.cache["batch_shape"] == (1, 2, 2, 5)
