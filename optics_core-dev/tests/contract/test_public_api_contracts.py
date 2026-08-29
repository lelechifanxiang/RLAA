from __future__ import annotations

import pytest
import torch

import optics_core as oc
from tests.support import build_showcase_system


pytestmark = pytest.mark.contract


def test_public_namespace_exports_core_symbols() -> None:
    required_names = {
        "AnalysisHub",
        "ImageFormationModel",
        "MultiOpticalSystem",
        "OpticalArchitecture",
        "ParameterSchema",
        "ParameterVectorBatch",
        "build_parameter_vector_grid",
        "SequentialSurfaceRayTracer",
        "SurfaceTraceHistory",
        "TraceOptions",
    }

    assert required_names.issubset(set(oc.__all__))
    for name in required_names:
        assert getattr(oc, name).__name__ == name

def test_trace_uses_declared_fields_and_wavelengths_by_default() -> None:
    showcase_system = build_showcase_system()
    showcase_system.prepare()
    trace = showcase_system.trace(sampler=oc.SquarePupilSampler(nx=2, ny=3))

    assert tuple(trace.rays.wavelength_index.shape) == (1, 2, 2, 7)
    assert tuple(trace.valid.shape) == (1, 2, 2, 7)
    assert torch.equal(trace.rays.wavelength_index[0, 0, 0], torch.zeros(7, dtype=torch.int64))
    assert torch.equal(trace.rays.wavelength_index[0, 0, 1], torch.ones(7, dtype=torch.int64))
    assert trace.rays.metadata == {}
    assert trace.cache["batch_shape"] == tuple(trace.rays.x.shape)


def test_trace_uses_configured_runtime_device_when_cuda_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")

    showcase_system = build_showcase_system()
    showcase_system.config.backend.device = "cuda"
    showcase_system.prepare()
    trace = showcase_system.trace(sampler=oc.SquarePupilSampler(nx=2, ny=2))

    assert trace.rays.x.device.type == "cuda"
    assert trace.rays.y.device.type == "cuda"
    assert trace.rays.z.device.type == "cuda"
    assert trace.rays.wavelength_index.device.type == "cuda"


def test_square_pupil_sampler_returns_tensor_coordinates() -> None:
    sample = oc.SquarePupilSampler(nx=2, ny=3).sample()

    assert isinstance(sample.pupil_coordinates, torch.Tensor)
    assert sample.pupil_coordinates.dtype == torch.float64
    assert tuple(sample.pupil_coordinates.shape) == (7, 2)
    assert sample.sample_ray_count == 6
    assert sample.chief_ray_index == 6
    assert sample.weights is not None
    assert torch.as_tensor(sample.weights, dtype=torch.float64).sum().item() == pytest.approx(1.0)
    assert torch.as_tensor(sample.weights, dtype=torch.float64)[sample.chief_ray_index].item() == pytest.approx(0.0)
    torch.testing.assert_close(
        sample.pupil_coordinates[sample.chief_ray_index],
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_square_pupil_sampler_keeps_reference_chief_ray_at_tail_when_grid_contains_origin() -> None:
    sample = oc.SquarePupilSampler(nx=3, ny=3).sample()

    assert tuple(sample.pupil_coordinates.shape) == (10, 2)
    assert sample.sample_ray_count == 9
    assert sample.chief_ray_index == 9
    torch.testing.assert_close(
        sample.pupil_coordinates[4],
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    chief_coordinate = sample.pupil_coordinates[sample.chief_ray_index]
    torch.testing.assert_close(
        chief_coordinate,
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_hexapolar_pupil_sampler_contains_center_ray() -> None:
    sample = oc.HexapolarPupilSampler(rings=3).sample()

    assert tuple(sample.pupil_coordinates.shape) == (38, 2)
    assert sample.sample_ray_count == 37
    assert sample.chief_ray_index == 37
    torch.testing.assert_close(
        sample.pupil_coordinates[0],
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        sample.pupil_coordinates[sample.chief_ray_index],
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_explicit_pupil_sampler_does_not_append_reference_chief_ray() -> None:
    coordinates = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
    sample = oc.ExplicitPupilSampler(pupil_coordinates=coordinates).sample()

    assert tuple(sample.pupil_coordinates.shape) == (2, 2)
    assert sample.sample_ray_count == 2
    assert sample.chief_ray_index is None


def test_wavelength_sequence_keeps_single_primary() -> None:
    wavelengths = oc.WavelengthSequence()
    first = wavelengths.add(0.4861, is_primary=True, label="F")
    second = wavelengths.add(0.5876, is_primary=True, label="d")

    assert first.is_primary is False
    assert second.is_primary is True
    assert wavelengths.primary_index == 1
    assert wavelengths.primary is second

def test_parameter_vector_grid_builds_expected_batch() -> None:
    schema = oc.ParameterSchema(
        [
            oc.ParameterSpec(name="radius", path="surface[1].geometry.radius", default=24.0),
            oc.ParameterSpec(name="distance", path="surface[0].gap.thickness", default=1000.0),
        ]
    )
    parameters = oc.build_parameter_vector_grid(
        schema,
        [
            oc.ParameterSweepAxis(parameter="radius", values=[24.0, 32.0, 45.0]),
            oc.ParameterSweepAxis(parameter="distance", values=[1000.0, 500.0]),
        ],
    )

    assert schema.parameter_count == 2
    assert parameters.system_count == 6
    assert parameters[0] == pytest.approx([24.0, 1000.0])
    assert parameters[-1] == pytest.approx([45.0, 500.0])
