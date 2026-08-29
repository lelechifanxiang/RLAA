from __future__ import annotations

import torch
import pytest

import optics_core as oc


pytestmark = pytest.mark.contract


def test_standard_geometry_radius_zero_matches_plane_geometry_for_tensor_inputs() -> None:
    plane = oc.PlaneGeometry(label="plane")
    standard = oc.StandardGeometry(label="degenerate_plane", radius=0.0, conic=1.5)

    x = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64)
    y = torch.tensor([0.0, -3.0, 4.0], dtype=torch.float64)

    torch.testing.assert_close(standard.sag(x, y), plane.sag(x, y))

    standard_nx, standard_ny, standard_nz = standard.normal(x, y)
    plane_nx, plane_ny, plane_nz = plane.normal(x, y)
    torch.testing.assert_close(standard_nx, plane_nx)
    torch.testing.assert_close(standard_ny, plane_ny)
    torch.testing.assert_close(standard_nz, plane_nz)


def test_standard_geometry_radius_zero_intersects_as_plane_without_side_effects(capsys: pytest.CaptureFixture[str]) -> None:
    geometry = oc.StandardGeometry(radius=0.0, conic=2.0)

    intersection = geometry.intersect(
        torch.tensor((0.0, 0.0, -1.0), dtype=torch.float64),
        torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64),
    )

    assert float(intersection.item()) == pytest.approx(1.0)
    assert geometry.radius == 0.0
    assert capsys.readouterr().out == ""