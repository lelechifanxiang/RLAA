from __future__ import annotations

from pathlib import Path

import pytest
import torch

import optics_core as oc
from tests.zemax.common import loaded_sequential_system
from tests.zemax.first_order import fetch_zemax_first_order_from_spec
from tests.zemax.paraxial_focus import fetch_zemax_paraxial_trace_from_spec
from tests.zemax.temp_structures import ParaxialTraceReference
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = pytest.mark.regression


PARAXIAL_SINGLE_LENS_ZMX_PATH = Path("tests/zemax/zmx_files/paraxial_single_lens.zmx")
PARAXIAL_TRACE_ABS_TOL_MM = 1e-6


def snapshot_trace_result(result: oc.TraceResult) -> ParaxialTraceReference:
    """把 TraceResult 收敛成统一的对标结果。"""
    if not result.intersections:
        raise ValueError("record_intersections must be enabled when building a trace snapshot.")

    def floats(values) -> list[float]:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().flatten().tolist()
        return [float(value) for value in values]

    def bools(values) -> list[bool]:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().flatten().tolist()
        return [bool(value) for value in values]

    px = result.intersections[0]
    return sort_trace(
        ParaxialTraceReference(
            paraxial_x_mm=floats(px.position[0]),
            paraxial_y_mm=floats(px.position[1]),
            paraxial_z_mm=floats(px.position[2]),
            image_x_mm=floats(result.rays.x),
            image_y_mm=floats(result.rays.y),
            image_z_mm=floats(result.rays.z),
            direction_l=floats(result.rays.l),
            direction_m=floats(result.rays.m),
            direction_n=floats(result.rays.n),
            valid=bools(result.valid),
            metadata=dict(result.cache),
        )
    )


def sort_trace(trace: ParaxialTraceReference) -> ParaxialTraceReference:
    """按近轴面坐标对齐不同来源的光线顺序。"""
    order = sorted(
        range(len(trace.valid)),
        key=lambda index: (trace.paraxial_x_mm[index], trace.paraxial_y_mm[index]),
    )

    def pick(values):
        return [values[index] for index in order]

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
        metadata=dict(trace.metadata),
    )


def assert_trace_matches_reference(
    actual: ParaxialTraceReference,
    reference: ParaxialTraceReference,
) -> None:
    """比较 optics_core 和 Zemax 的近轴追迹结果。"""
    actual = sort_trace(actual)
    reference = sort_trace(reference)

    assert list(actual.valid) == list(reference.valid)
    assert actual.paraxial_x_mm == pytest.approx(reference.paraxial_x_mm, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.paraxial_y_mm == pytest.approx(reference.paraxial_y_mm, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.paraxial_z_mm == pytest.approx(reference.paraxial_z_mm, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.direction_l == pytest.approx(reference.direction_l, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.direction_m == pytest.approx(reference.direction_m, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.direction_n == pytest.approx(reference.direction_n, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.image_x_mm == pytest.approx(reference.image_x_mm, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.image_y_mm == pytest.approx(reference.image_y_mm, abs=PARAXIAL_TRACE_ABS_TOL_MM)
    assert actual.image_z_mm == pytest.approx(reference.image_z_mm, abs=PARAXIAL_TRACE_ABS_TOL_MM)


@pytest.mark.zemax
def test_basic_workflow() -> None:
    """覆盖系统构造、追迹和 Zemax 对标的最小端到端流程。"""
    spec = load_zmx_sequential_system_spec(PARAXIAL_SINGLE_LENS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    system.prepare()
    sample = oc.SquarePupilSampler(nx=3, ny=3).sample()
    pupil_coordinates = sample.pupil_coordinates[: sample.sample_ray_count]
    result = system.trace(
        sampler=oc.ExplicitPupilSampler(pupil_coordinates=pupil_coordinates),
        options=oc.TraceOptions(record_intersections=True),
    )

    actual = snapshot_trace_result(result)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_paraxial_trace_from_spec(
            spec,
            oss,
            pupil_grid_shape=(3, 3),
        )
    assert_trace_matches_reference(actual, reference)


@pytest.mark.zemax
def test_paraxial_first_order_matches_zemax() -> None:
    """验证单近轴面的 first_order.run() 输出。"""
    spec = load_zmx_sequential_system_spec(PARAXIAL_SINGLE_LENS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_first_order_from_spec(spec, oss)

    system.prepare()
    result = system.analysis.first_order().run()

    print(f"Paraxial Zemax EFFL={reference.effective_focal_length_mm:.12f} mm")
    print(f"Paraxial OpticsCore EFFL={result.effl.item():.12f} mm")
    print(f"Paraxial Zemax ENPP={reference.entrance_pupil_z_mm:.12f} mm")
    print(f"Paraxial OpticsCore ENPP={result.entrance_pupil_z.item():.12f} mm")
    print(f"Paraxial Zemax ENPR={reference.entrance_pupil_radius_mm:.12f} mm")
    print(f"Paraxial OpticsCore ENPR={result.entrance_pupil_radius.item():.12f} mm")
    print(f"Paraxial Zemax BFL={reference.back_focal_length_mm:.12f} mm")
    print(f"Paraxial OpticsCore BFL={result.bfl.item():.12f} mm")

    assert result.ttl.item() == pytest.approx(reference.total_track_length_mm, abs=5e-5)
    assert result.effl.item() == pytest.approx(reference.effective_focal_length_mm, abs=1e-12)
    assert result.working_f_number.item() == pytest.approx(reference.working_f_number, abs=1e-6)
    assert result.image_plane_distance.item() == pytest.approx(reference.image_plane_distance_mm, abs=1e-12)
    assert result.bfl.item() == pytest.approx(reference.back_focal_length_mm, abs=1e-12)
    assert result.valid.item()
    assert result.entrance_pupil_z.item() == pytest.approx(reference.entrance_pupil_z_mm, abs=1e-12)
    assert result.entrance_pupil_radius.item() == pytest.approx(reference.entrance_pupil_radius_mm, abs=1e-12)


@pytest.mark.zemax
@pytest.mark.parametrize("object_distance_mm", [None, 100.0])
def test_image_f_number_first_order_matches_zemax(
    tmp_path: Path,
    object_distance_mm: float | None,
) -> None:
    """验证无限和有限物距 FNUM 系统的一阶量与 Zemax 一致。"""
    case_name = "infinite" if object_distance_mm is None else "finite"
    zmx_path = tmp_path / f"paraxial_fnum_{case_name}.zmx"
    zmx_text = PARAXIAL_SINGLE_LENS_ZMX_PATH.read_text(encoding="utf-16")
    zmx_text = zmx_text.replace("ENPD 12", "FNUM 4")
    if object_distance_mm is not None:
        zmx_text = zmx_text.replace("DISZ INFINITY", f"DISZ {object_distance_mm}")
    zmx_path.write_text(zmx_text, encoding="utf-16")
    spec = load_zmx_sequential_system_spec(zmx_path)
    system = build_optics_core_system_from_zmx_spec(spec)
    with loaded_sequential_system(spec.zmx_path) as oss:
        reference = fetch_zemax_first_order_from_spec(spec, oss)

    result = system.prepare().first_order_data

    print(f"FNUM Zemax ISFN={reference.image_f_number:.12f}")
    print(f"FNUM Zemax WFNO={reference.working_f_number:.12f}")
    print(f"FNUM OpticsCore WFNO={result.working_f_number.item():.12f}")
    print(f"FNUM Zemax ENPR={reference.entrance_pupil_radius_mm:.12f} mm")
    print(f"FNUM OpticsCore ENPR={result.entrance_pupil_radius.item():.12f} mm")

    assert reference.image_f_number == pytest.approx(4.0, abs=1e-12)
    assert result.effl.item() == pytest.approx(reference.effective_focal_length_mm, abs=1e-12)
    assert result.entrance_pupil_z.item() == pytest.approx(reference.entrance_pupil_z_mm, abs=1e-12)
    assert result.entrance_pupil_radius.item() == pytest.approx(reference.entrance_pupil_radius_mm, abs=1e-12)
    assert result.exit_pupil_z.item() - result.ttl.item() == pytest.approx(
        reference.exit_pupil_z_mm,
        abs=1e-12,
    )
    assert result.exit_pupil_radius.item() == pytest.approx(reference.exit_pupil_radius_mm, abs=1e-12)
    assert result.working_f_number.item() == pytest.approx(reference.working_f_number, abs=1e-9)
