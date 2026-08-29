from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.fixtures import (
    MULTIFIELD_PARAXIAL_FIELD_CASES,
    build_multifield_multistructure_system,
)
from tests.zemax.common import loaded_sequential_system
from tests.zemax.paraxial_focus import fetch_zemax_paraxial_trace_from_spec
from tests.zemax.temp_structures import ParaxialTraceReference
from tests.zemax.zmx_loader import load_zmx_sequential_system_spec


pytestmark = pytest.mark.regression


PARAXIAL_SINGLE_LENS_ZMX_PATH = Path("tests/zemax/zmx_files/paraxial_single_lens.zmx")
MULTIFIELD_EDGE_FIELD_DEG = (20.0, 30.0)


def snapshot_system_trace(
    result: oc.TraceResult,
    pupil_coordinates: torch.Tensor,
    system_index: int,
    field_index: int,
    wavelength_index: int = 0,
) -> ParaxialTraceReference:
    """从批量 TraceResult 中抽取单个结构的对标结果。"""
    paraxial_hit = result.intersections[0]
    return sort_trace(
        ParaxialTraceReference(
            paraxial_x_mm=_floats(paraxial_hit.position[0][system_index, field_index, wavelength_index]),
            paraxial_y_mm=_floats(paraxial_hit.position[1][system_index, field_index, wavelength_index]),
            paraxial_z_mm=_floats(paraxial_hit.position[2][system_index, field_index, wavelength_index]),
            image_x_mm=_floats(result.rays.x[system_index, field_index, wavelength_index]),
            image_y_mm=_floats(result.rays.y[system_index, field_index, wavelength_index]),
            image_z_mm=_floats(result.rays.z[system_index, field_index, wavelength_index]),
            direction_l=_floats(result.rays.l[system_index, field_index, wavelength_index]),
            direction_m=_floats(result.rays.m[system_index, field_index, wavelength_index]),
            direction_n=_floats(result.rays.n[system_index, field_index, wavelength_index]),
            valid=_bools(result.valid[system_index, field_index, wavelength_index]),
            metadata={"pupil_coordinates": pupil_coordinates},
        )
    )


def sort_trace(trace: ParaxialTraceReference) -> ParaxialTraceReference:
    pupil_coordinates = trace.metadata.get("pupil_coordinates")
    if pupil_coordinates is None:
        raise ValueError("pupil_coordinates is required for trace sorting.")

    order = sorted(
        range(len(trace.valid)),
        key=lambda index: _pupil_coordinate_key(pupil_coordinates[index]),
    )

    def pick(values):
        return [values[index] for index in order]

    sorted_pupil_coordinates = [_pupil_coordinate_key(pupil_coordinates[index]) for index in order]
    return ParaxialTraceReference(
        paraxial_x_mm=pick(trace.paraxial_x_mm),
        paraxial_y_mm=pick(trace.paraxial_y_mm),
        paraxial_z_mm=pick(trace.paraxial_z_mm),
        image_x_mm=pick(trace.image_x_mm),
        image_y_mm=pick(trace.image_y_mm),
        image_z_mm=pick(trace.image_z_mm),
        direction_l=pick(trace.direction_l),
        direction_m=pick(trace.direction_m),
        direction_n=pick(trace.direction_n),
        valid=pick(trace.valid),
        metadata={**trace.metadata, "pupil_coordinates": sorted_pupil_coordinates},
    )


def _floats(values) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()
    return [float(value) for value in values]


def _bools(values) -> list[bool]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()
    return [bool(value) for value in values]


def _pupil_coordinate_key(value) -> tuple[float, float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return float(value[0]), float(value[1])


def assert_trace_matches_zemax(
    actual: ParaxialTraceReference,
    reference: ParaxialTraceReference,
) -> None:
    reference = sort_trace(reference)
    assert actual.valid == reference.valid
    assert actual.paraxial_x_mm == pytest.approx(reference.paraxial_x_mm, abs=1e-9)
    assert actual.paraxial_y_mm == pytest.approx(reference.paraxial_y_mm, abs=1e-9)
    assert actual.paraxial_z_mm == pytest.approx(reference.paraxial_z_mm, abs=1e-9)
    assert actual.direction_l == pytest.approx(reference.direction_l, abs=1e-9)
    assert actual.direction_m == pytest.approx(reference.direction_m, abs=1e-9)
    assert actual.direction_n == pytest.approx(reference.direction_n, abs=1e-9)
    assert actual.image_x_mm == pytest.approx(reference.image_x_mm, abs=1e-9)
    assert actual.image_y_mm == pytest.approx(reference.image_y_mm, abs=1e-9)
    assert actual.image_z_mm == pytest.approx(reference.image_z_mm, abs=1e-9)


@pytest.mark.zemax
def test_multifield_multistructure_paraxial_trace_matches_zemax_reference() -> None:
    spec = load_zmx_sequential_system_spec(PARAXIAL_SINGLE_LENS_ZMX_PATH)
    system = build_multifield_multistructure_system()
    sample = oc.SquarePupilSampler(nx=3, ny=3).sample()
    sampler = oc.ExplicitPupilSampler(pupil_coordinates=sample.pupil_coordinates[: sample.sample_ray_count])
    options = oc.TraceOptions(record_intersections=True)

    assert system.system_count == 3
    assert len(system.fields) == len(MULTIFIELD_PARAXIAL_FIELD_CASES)

    system.prepare()
    result = system.trace(
        sampler=sampler,
        options=options,
    )

    assert tuple(result.rays.x.shape) == (system.system_count, len(MULTIFIELD_PARAXIAL_FIELD_CASES), 1, 9)
    assert torch.all(result.valid)
    assert [hit.surface_index for hit in result.intersections] == [0, 1]

    with loaded_sequential_system(spec.zmx_path) as oss:
        for field_index, (hx, hy) in enumerate(MULTIFIELD_PARAXIAL_FIELD_CASES):
            for system_index, parameter_vector in enumerate(system.parameters):
                focal_length, image_distance = parameter_vector
                reference = fetch_zemax_paraxial_trace_from_spec(
                    spec,
                    oss,
                    focal_length_mm=focal_length,
                    image_plane_distance_mm=image_distance,
                    field_point_deg=(MULTIFIELD_EDGE_FIELD_DEG[0] * hx, MULTIFIELD_EDGE_FIELD_DEG[1] * hy),
                    edge_field_deg=MULTIFIELD_EDGE_FIELD_DEG,
                    pupil_grid_shape=(3, 3),
                )
                assert_trace_matches_zemax(
                    snapshot_system_trace(
                        result,
                        sample.pupil_coordinates[: sample.sample_ray_count],
                        system_index,
                        field_index,
                    ),
                    reference,
                )
