from __future__ import annotations

import pytest
import torch

import optics_core as oc
from optics_core._first_order_probes import (
    build_front_paraxial_probe_rays,
    build_marginal_probe_ray,
)
from optics_core.first_order import (
    compute_first_order_data,
    resolve_system_aperture,
)
from tests.fixtures import build_backward_paraxial_system, build_multifield_multistructure_system


pytestmark = pytest.mark.regression


class _CountingTracer(oc.SequentialSurfaceRayTracer):
    """记录一阶量计算发起的追迹。"""

    def __init__(self) -> None:
        self.options: list[oc.TraceOptions] = []
        self.ray_counts: list[int] = []

    def trace(
        self,
        system: oc.MultiOpticalSystem,
        rays: oc.RayBundle,
        options: oc.TraceOptions | None = None,
    ) -> oc.TraceResult:
        resolved_options = options or oc.TraceOptions()
        self.options.append(resolved_options)
        self.ray_counts.append(rays.x.shape[-1])
        return super().trace(system, rays, resolved_options)


def test_trace_backward_inverts_paraxial_forward_path() -> None:
    system = build_backward_paraxial_system()
    tracer = system.tracer
    rays = oc.RayBundle(
        x=torch.tensor([[0.0, 2.0]], dtype=torch.float64),
        y=torch.tensor([[0.0, -1.0]], dtype=torch.float64),
        z=torch.zeros((1, 2), dtype=torch.float64),
        l=torch.tensor([[0.0, 0.05]], dtype=torch.float64),
        m=torch.tensor([[0.0, -0.02]], dtype=torch.float64),
        n=torch.ones((1, 2), dtype=torch.float64),
        wavelength_index=torch.full((1, 2), system.wavelengths.primary_index, dtype=torch.int64),
    )

    forward = tracer.trace(system, rays, options=oc.TraceOptions(record_intersections=False))
    backward = tracer.trace(
        system,
        forward.rays,
        options=oc.TraceOptions(direction="backward", record_intersections=False),
    )

    assert torch.allclose(backward.rays.x, torch.as_tensor(rays.x, dtype=torch.float64))
    assert torch.allclose(backward.rays.y, torch.as_tensor(rays.y, dtype=torch.float64))
    assert torch.allclose(backward.rays.z, torch.as_tensor(rays.z, dtype=torch.float64))
    assert torch.allclose(backward.rays.l, torch.as_tensor(rays.l, dtype=torch.float64))
    assert torch.allclose(backward.rays.m, torch.as_tensor(rays.m, dtype=torch.float64))
    assert backward.cache["mode"] == "backward"
    assert backward.rays.metadata["trace_mode"] == "sequential_surface_backward"


def test_first_order_queries_resolve_stop_geometry() -> None:
    system = build_backward_paraxial_system()
    system.prepare()
    stop_index = system.surfaces.stop_index
    stop_position = system.frame_data.surface_z(stop_index)

    assert stop_index == 1
    assert stop_position.detach().cpu().tolist() == pytest.approx([20.0], abs=1e-12)
    assert system.first_order_data.entrance_pupil_radius.detach().cpu().tolist() == pytest.approx([6.0], abs=1e-12)


def test_first_order_probe_groups_share_primary_wavelength() -> None:
    """前组近轴光线和真实边缘光线应分别使用主波长。"""
    system = build_backward_paraxial_system()
    probe_height = torch.full((system.system_count,), 1e-6, dtype=torch.float64)
    front_rays = build_front_paraxial_probe_rays(system, probe_height)
    marginal_ray = build_marginal_probe_ray(
        system,
        entrance_pupil_z=torch.tensor([40.0], dtype=torch.float64),
        entrance_pupil_radius=torch.tensor([6.0], dtype=torch.float64),
    )

    assert tuple(front_rays.x.shape) == (system.system_count, 4)
    assert tuple(marginal_ray.x.shape) == (system.system_count, 1)
    assert torch.equal(
        front_rays.wavelength_index,
        torch.full(
            (system.system_count, 4),
            system.wavelengths.primary_index,
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(marginal_ray.y[:, 0], torch.tensor([6.0], dtype=torch.float64))


def test_image_f_number_resolves_design_dependent_pupil_radius() -> None:
    system = build_multifield_multistructure_system()
    system.set_aperture("image_f_number", 4.0, stop_surface=0, label="FNO")
    effl = torch.tensor([40.0, 48.0, 56.0], dtype=torch.float64)
    front_magnification = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

    aperture = resolve_system_aperture(
        system,
        effl=effl,
        front_pupil_magnification=front_magnification,
    )

    torch.testing.assert_close(
        aperture.entrance_pupil_radius,
        torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        aperture.stop_radius,
        torch.tensor([5.0, 3.0, 14.0], dtype=torch.float64),
    )


def test_object_na_remains_unsupported() -> None:
    system = build_backward_paraxial_system()
    system.set_aperture("object_na", 0.1, stop_surface=1, label="OBNA")

    with pytest.raises(NotImplementedError):
        compute_first_order_data(system)


def test_first_order_run_updates_system_result() -> None:
    system = build_backward_paraxial_system()
    system.prepare()

    result = system.analysis.first_order().run()

    assert isinstance(system.first_order_data, oc.FirstOrderData)
    assert result.ttl.detach().cpu().tolist() == pytest.approx([20.0], abs=1e-12)
    assert result.effl.detach().cpu().tolist() == pytest.approx([40.0], abs=1e-12)
    assert result.entrance_pupil_z.detach().cpu().tolist() == pytest.approx([40.0], abs=1e-12)
    assert result.entrance_pupil_radius.detach().cpu().tolist() == pytest.approx([6.0], abs=1e-12)


def test_prepare_populates_frame_and_first_order_data() -> None:
    system = build_backward_paraxial_system()

    prepared_system = system.prepare()

    assert prepared_system is system
    assert isinstance(system.frame_data, oc.FrameData)
    assert isinstance(system.first_order_data, oc.FirstOrderData)
    assert tuple(system.frame_data.origins.shape) == (1, 2, 3)
    assert system.frame_data.origins[0, :, 2].detach().cpu().tolist() == pytest.approx([0.0, 20.0], abs=1e-12)
    assert system.first_order_data.ttl.detach().cpu().tolist() == pytest.approx([20.0], abs=1e-12)
    assert system.first_order_data.effl.detach().cpu().tolist() == pytest.approx([40.0], abs=1e-12)
    assert system.first_order_data.entrance_pupil_z.detach().cpu().tolist() == pytest.approx([40.0], abs=1e-12)


def test_first_order_data_uses_separate_probe_groups() -> None:
    """一阶量应依次追迹 4 根近轴、1 根边缘和 2 根出瞳光线。"""
    system = build_multifield_multistructure_system()
    tracer = _CountingTracer()
    system.set_tracer(tracer)

    first_order = compute_first_order_data(system)

    assert tracer.ray_counts == [4, 1, 2]
    assert len(tracer.options) == 3
    assert tracer.options[0].record_intersections is True
    assert tracer.options[0].record_opd is False
    assert tracer.options[0].ignore_coordinate_breaks is True
    assert tracer.options[1].record_intersections is False
    assert tracer.options[1].record_opd is False
    assert tracer.options[1].ignore_coordinate_breaks is True
    assert tracer.options[2].start_surface == system.surfaces.stop_index
    assert tracer.options[2].record_intersections is False
    assert tracer.options[2].record_opd is False
    assert tracer.options[2].ignore_coordinate_breaks is True
    assert torch.isfinite(first_order.effl).all()
    assert torch.isfinite(first_order.working_f_number).all()
    assert torch.isfinite(first_order.entrance_pupil_z).all()
    assert torch.isfinite(first_order.exit_pupil_z).all()
