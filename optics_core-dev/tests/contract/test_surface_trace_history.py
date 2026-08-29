from __future__ import annotations

import pytest
import torch

import optics_core as oc
from tests.fixtures import build_multi_system_multi_sphere_system, build_surface_trace_system


pytestmark = pytest.mark.contract


def _batched_rays(system: oc.MultiOpticalSystem) -> oc.RayBundle:
    shape = (system.system_count, 2, 3, 2)
    l = torch.tensor([0.0, 0.025], dtype=torch.float64).reshape(1, 2, 1, 1).expand(shape)
    m = torch.tensor([0.0, -0.015], dtype=torch.float64).reshape(1, 2, 1, 1).expand(shape)
    return oc.RayBundle(
        x=torch.tensor([-0.8, 0.9], dtype=torch.float64).reshape(1, 1, 1, 2).expand(shape).clone(),
        y=torch.tensor([0.5, -0.4], dtype=torch.float64).reshape(1, 1, 1, 2).expand(shape).clone(),
        z=torch.full(shape, -15.0, dtype=torch.float64),
        l=l,
        m=m,
        n=torch.sqrt(1.0 - l * l - m * m),
        wavelength_index=torch.arange(3, dtype=torch.int64).reshape(1, 1, 3, 1).expand(shape),
        opl=torch.full(shape, 1.25, dtype=torch.float64),
    )


def _design_rays(rays: oc.RayBundle, design_index: int) -> oc.RayBundle:
    return oc.RayBundle(
        **{
            name: getattr(rays, name)[design_index : design_index + 1]
            for name in ("x", "y", "z", "l", "m", "n", "wavelength_index", "opl")
        }
    )


def _assert_history_state(history: oc.SurfaceTraceHistory, history_index: int, result: oc.TraceResult) -> None:
    for name in ("x", "y", "z", "l", "m", "n", "opl"):
        torch.testing.assert_close(getattr(history, name)[:, history_index], getattr(result.rays, name))
    assert torch.equal(history.valid[:, history_index], result.valid)


def test_surface_history_records_post_surface_batch_states() -> None:
    system = build_multi_system_multi_sphere_system()
    rays = _batched_rays(system)
    result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_surface_states="all"),
    )
    history = result.surface_history

    assert history is not None
    assert history.surface_indices == tuple(range(len(system.surfaces)))
    assert history.x.shape == (system.system_count, len(system.surfaces), 2, 3, 2)
    assert history.x.dtype == torch.float64
    assert history.valid.dtype == torch.bool
    assert len(result.intersections) == len(system.surfaces)
    _assert_history_state(history, -1, result)

    for design_index in range(system.system_count):
        design = system.design_view(design_index)
        design_result = design.tracer.trace(
            design,
            _design_rays(rays, design_index),
            options=oc.TraceOptions(record_intersections=False, record_surface_states="all"),
        )
        design_history = design_result.surface_history
        assert design_history is not None
        for name in ("x", "y", "z", "l", "m", "n", "opl", "valid"):
            torch.testing.assert_close(
                getattr(history, name)[design_index : design_index + 1],
                getattr(design_history, name),
            )


def test_selected_surface_history_matches_individual_stop_traces() -> None:
    system = build_multi_system_multi_sphere_system()
    rays = _batched_rays(system)
    selected = (len(system.surfaces) - 1, 0, 2)
    result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=False, record_surface_states=selected),
    )
    history = result.surface_history

    assert history is not None
    assert history.surface_indices == (0, 2, len(system.surfaces) - 1)
    for history_index, surface_index in enumerate(history.surface_indices):
        stopped = system.tracer.trace(
            system,
            rays,
            options=oc.TraceOptions(stop_surface=surface_index, record_intersections=False),
        )
        _assert_history_state(history, history_index, stopped)

    disabled = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=False),
    )
    without_opl = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=False, record_opd=False, record_surface_states=(0,)),
    )
    assert disabled.surface_history is None
    assert without_opl.surface_history is not None
    assert without_opl.surface_history.opl is None

    segment = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(start_surface=1, stop_surface=2, record_surface_states="all"),
    )
    assert segment.surface_history is not None
    assert segment.surface_history.surface_indices == (1, 2)


def test_surface_history_preserves_invalid_outgoing_state() -> None:
    system = build_surface_trace_system(
        oc.SphereSurface(
            radius=50.0,
            thickness=0.0,
            semi_diameter=1.0,
            semi_diameter_solve="fixed",
            aperture_type="floating",
        )
    )
    rays = oc.RayBundle(
        x=torch.tensor([[2.0]], dtype=torch.float64),
        y=torch.zeros((1, 1), dtype=torch.float64),
        z=torch.full((1, 1), -10.0, dtype=torch.float64),
        l=torch.zeros((1, 1), dtype=torch.float64),
        m=torch.zeros((1, 1), dtype=torch.float64),
        n=torch.ones((1, 1), dtype=torch.float64),
        wavelength_index=torch.zeros((1, 1), dtype=torch.int64),
    )
    result = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=False, record_surface_states="all"),
    )

    assert result.surface_history is not None
    assert not torch.any(result.surface_history.valid)
    assert torch.isnan(result.surface_history.l).all()
    assert torch.isnan(result.surface_history.opl).all()


def test_segmented_trace_matches_complete_trace() -> None:
    """上一段终态可直接作为下一段输入，续追结果与完整追迹一致。"""
    system = build_multi_system_multi_sphere_system()
    rays = _batched_rays(system)
    split_surface = 1
    full = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(record_intersections=False, record_surface_states="all"),
    )
    first = system.tracer.trace(
        system,
        rays,
        options=oc.TraceOptions(stop_surface=split_surface, record_intersections=False),
    )
    resumed = system.tracer.trace(
        system,
        first.rays,
        options=oc.TraceOptions(start_surface=split_surface + 1, record_intersections=False),
    )

    for name in ("x", "y", "z", "l", "m", "n", "opl"):
        torch.testing.assert_close(getattr(resumed.rays, name), getattr(full.rays, name))
    assert torch.equal(resumed.valid, full.valid)
    assert full.surface_history is not None
    _assert_history_state(full.surface_history, split_surface, first)
