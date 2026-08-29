from __future__ import annotations

import pytest

import optics_core as oc
from tests.support import build_showcase_system


pytestmark = pytest.mark.benchmark


def test_trace_benchmark(benchmark_runner, record_property) -> None:
    showcase_system = build_showcase_system()
    showcase_system.prepare()
    sampler = oc.SquarePupilSampler(nx=16, ny=16)
    expected_ray_count = sampler.sample().pupil_coordinates.shape[0]
    benchmark = benchmark_runner(
        lambda: showcase_system.trace(sampler=sampler),
        warmup=3,
        iterations=25,
    )

    record_property("trace_avg_ms", round(benchmark.avg_ms, 6))
    assert benchmark.last_result.valid.shape[-1] == expected_ray_count
    assert benchmark.avg_ms >= 0.0
