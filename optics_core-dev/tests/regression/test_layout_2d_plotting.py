from __future__ import annotations

import pytest
import torch

import optics_core as oc
from tests.fixtures.cases import DEFAULT_FORWARD_MULTI_SPHERE_CASE
from tests.fixtures.recipes import build_spherical_forward_architecture
from tests.fixtures.systems import build_tracing_system


pytestmark = [pytest.mark.regression]


def _build_layout_plot_system() -> oc.MultiOpticalSystem:
    architecture = build_spherical_forward_architecture(
        DEFAULT_FORWARD_MULTI_SPHERE_CASE,
        name="layout_plot_check",
    )
    return build_tracing_system(
        architecture,
        field_points=((0.0, 0.0), (0.0, -3.0)),
        wavelengths_um=(0.5875618,),
        wavelength_labels=("d",),
        primary_wavelength_index=0,
        aperture_diameter=36.0,
        stop_surface=0,
    )


def test_layout_2d_plot_starts_rays_at_first_surface_and_draws_lens_edges() -> None:
    """验证 layout 图中的光线从首面交点开始，且实体镜片边缘会闭合。"""
    system = _build_layout_plot_system()
    system.prepare()
    output_path = "tests/output/layout_plot_check.png"

    result = system.analysis.layout_2d(
        oc.Layout2DSettings(save_path=output_path),
    ).run()

    assert result.trace_result is not None
    assert result.axes is not None

    first_hit_y = torch.as_tensor(
        result.trace_result.intersections[0].position[1][0, 0, 0, 0],
        dtype=torch.float64,
    ).item()
    first_hit_z = torch.as_tensor(
        result.trace_result.intersections[0].position[2][0, 0, 0, 0],
        dtype=torch.float64,
    ).item()
    entrance_pupil_z = torch.as_tensor(
        system.first_order_data.entrance_pupil_z[0],
        dtype=torch.float64,
    ).item()

    chief_field_line = next(
        line for line in result.axes.lines
        if line.get_label() == "field x=0.0, y=0.0"
    )
    x_data = chief_field_line.get_xdata()
    y_data = chief_field_line.get_ydata()

    assert x_data[0] == pytest.approx(first_hit_z, abs=1e-12)
    assert y_data[0] == pytest.approx(first_hit_y, abs=1e-12)
    assert x_data[0] != pytest.approx(entrance_pupil_z, abs=1e-12)

    black_line_count = sum(1 for line in result.axes.lines if line.get_color() == "black")
    print(f"layout 黑色轮廓线数量: {black_line_count}")
    assert black_line_count > len(system.surfaces)
