from __future__ import annotations

from pathlib import Path

import optics_core as oc
import pytest
import torch

from tests.zemax.batch_ray_trace import (
    benchmark_zemax_batch_direct_kernel,
    build_optics_core_direct_rays,
    build_direct_ray_set_from_zmx,
)
from tests.zemax.common import loaded_sequential_system
from tests.zemax.zmx_loader import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


pytestmark = [pytest.mark.benchmark, pytest.mark.zemax]


DOUBLE_GAUSS_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx")
DEFAULT_PUPIL_RAY_COUNT = 100
POSITION_ABS_TOL_MM = 1e-6
DIRECTION_ABS_TOL = 1e-6


def test_double_gauss_zmx_batch_ray_trace_benchmark(benchmark_runner, record_property) -> None:
    """基于 zmx 文件对比 Zemax BatchRayTrace 与 optics_core 直接光线追迹耗时。"""

    # 读取zmx规格，构建系统
    spec = load_zmx_sequential_system_spec(DOUBLE_GAUSS_ZMX_PATH)
    system = build_optics_core_system_from_zmx_spec(spec)
    if torch.cuda.is_available():
        system.config.backend.device = "cuda"
    system.prepare()
    # 随机采样光线
    sample = oc.RandomPupilSampler(ray_count=DEFAULT_PUPIL_RAY_COUNT, seed=0).sample()
    if sample.pupil_coordinates is None:
        raise ValueError("RandomPupilSampler must provide pupil_coordinates.")
    pupil_coordinates = sample.pupil_coordinates[: sample.sample_ray_count]

    with loaded_sequential_system(spec.zmx_path) as oss:
        ray_set = build_direct_ray_set_from_zmx(oss, spec, pupil_coordinates)
        zemax_kernel_benchmark = benchmark_zemax_batch_direct_kernel(
            oss,
            spec,
            ray_set,
            warmup=1,
            iterations=3,
        )

    optics_core_rays = build_optics_core_direct_rays(system, ray_set)

    def run_optics_core_trace_kernel_and_sync():
        if optics_core_rays.x.device.type == "cuda":
            torch.cuda.synchronize(optics_core_rays.x.device)
        result = system.tracer.trace(
            system,
            optics_core_rays,
            options=oc.TraceOptions(record_intersections=False),
        )
        if result.rays.x.device.type == "cuda":
            torch.cuda.synchronize(result.rays.x.device)
        return result

    optics_core_kernel_benchmark = benchmark_runner(
        run_optics_core_trace_kernel_and_sync,
        warmup=1,
        iterations=3,
    )

    zemax_result = zemax_kernel_benchmark.last_result
    optics_core_result = optics_core_kernel_benchmark.last_result
    runtime_device = optics_core_result.rays.x.device
    zemax_valid = zemax_result.valid.to(device=runtime_device)
    zemax_x_mm = zemax_result.x_mm.to(device=runtime_device)
    zemax_y_mm = zemax_result.y_mm.to(device=runtime_device)
    zemax_direction_l = zemax_result.direction_l.to(device=runtime_device)
    zemax_direction_m = zemax_result.direction_m.to(device=runtime_device)
    zemax_direction_n = zemax_result.direction_n.to(device=runtime_device)
    common_valid_mask = zemax_valid & optics_core_result.valid

    total_ray_count = int(ray_set.x.numel())
    common_valid_count = int(common_valid_mask.sum().item())

    print(f"Double Gauss zmx: {spec.zmx_path}")
    print(f"field_count={len(spec.field_points)}, wavelength_count={len(spec.wavelengths_um)}, pupil_ray_count={DEFAULT_PUPIL_RAY_COUNT}")
    print(f"total_rays={total_ray_count}, common_valid_rays={common_valid_count}")
    print(f"OpticsCore runtime device={runtime_device}")
    print(f"Zemax BatchRayTrace kernel_run avg_ms={zemax_kernel_benchmark.run_avg_ms:.6f}")
    print(f"Zemax BatchRayTrace add_rays avg_ms={zemax_kernel_benchmark.add_rays_avg_ms:.6f}")
    print(f"Zemax BatchRayTrace read_results avg_ms={zemax_kernel_benchmark.read_results_avg_ms:.6f}")
    print(f"OpticsCore trace kernel avg_ms={optics_core_kernel_benchmark.avg_ms:.6f}")
    if runtime_device.type == "cuda":
        print("说明: 当前环境已启用 GPU，OpticsCore 耗时包含 CUDA 同步后的真实执行时间。")
    else:
        print("说明: 当前环境未启用 GPU，本机 optics_core 耗时接近或高于 Zemax 并不意外。")

    record_property("double_gauss_total_rays", total_ray_count)
    record_property("double_gauss_zemax_batch_kernel_run_avg_ms", round(zemax_kernel_benchmark.run_avg_ms, 6))
    record_property("double_gauss_zemax_batch_add_rays_avg_ms", round(zemax_kernel_benchmark.add_rays_avg_ms, 6))
    record_property("double_gauss_zemax_batch_read_results_avg_ms", round(zemax_kernel_benchmark.read_results_avg_ms, 6))
    record_property("double_gauss_optics_core_kernel_avg_ms", round(optics_core_kernel_benchmark.avg_ms, 6))

    assert total_ray_count == len(spec.field_points) * len(spec.wavelengths_um) * DEFAULT_PUPIL_RAY_COUNT
    assert common_valid_count == total_ray_count
    assert torch.equal(
        optics_core_result.valid,
        zemax_valid,
    )

    torch.testing.assert_close(
        optics_core_result.rays.x[common_valid_mask],
        zemax_x_mm[common_valid_mask],
        atol=POSITION_ABS_TOL_MM,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.y[common_valid_mask],
        zemax_y_mm[common_valid_mask],
        atol=POSITION_ABS_TOL_MM,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.l[common_valid_mask],
        zemax_direction_l[common_valid_mask],
        atol=DIRECTION_ABS_TOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.m[common_valid_mask],
        zemax_direction_m[common_valid_mask],
        atol=DIRECTION_ABS_TOL,
        rtol=0.0,
    )
    torch.testing.assert_close(
        optics_core_result.rays.n[common_valid_mask],
        zemax_direction_n[common_valid_mask],
        atol=DIRECTION_ABS_TOL,
        rtol=0.0,
    )

    assert zemax_kernel_benchmark.run_avg_ms >= 0.0
    assert optics_core_kernel_benchmark.avg_ms >= 0.0
