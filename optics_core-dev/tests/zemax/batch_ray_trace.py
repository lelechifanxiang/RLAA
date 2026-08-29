from __future__ import annotations

import time
from typing import Any

import torch

import optics_core as oc
from optics_core._material_batch import compile_batched_material_data
from optics_core._runtime import default_device
from tests.zemax.common import normalized_field_coordinate, zp
from tests.zemax.temp_structures import (
    ZemaxBatchKernelBenchmarkResult,
    ZemaxBatchTraceResult,
    ZemaxDirectRaySet,
    ZemaxSequentialSystemSpec,
)


def sample_unit_disk_pupil_coordinates(
    ray_count: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """兼容旧测试入口，内部复用 RandomPupilSampler。"""

    sample = oc.RandomPupilSampler(ray_count=ray_count, seed=seed).sample()
    if sample.pupil_coordinates is None:
        raise ValueError("RandomPupilSampler must provide pupil coordinates.")
    return sample.pupil_coordinates[: sample.sample_ray_count]


def build_direct_ray_set_from_zmx(
    oss: Any,
    spec: ZemaxSequentialSystemSpec,
    pupil_coordinates: torch.Tensor,
    *,
    field_points: tuple[tuple[float, float], ...] | None = None,
    wavelength_indices: tuple[int, ...] | None = None,
) -> ZemaxDirectRaySet:
    """使用 Zemax GetDirectFieldCoordinates 构建显式入射光线。"""

    pupil_tensor = torch.as_tensor(pupil_coordinates, dtype=torch.float64)
    if pupil_tensor.ndim != 2 or pupil_tensor.shape[-1] != 2:
        raise ValueError("pupil_coordinates must have shape (ray_count, 2).")

    selected_field_points = spec.field_points if field_points is None else field_points
    selected_wavelength_indices = (
        tuple(range(1, len(spec.wavelengths_um) + 1))
        if wavelength_indices is None
        else wavelength_indices
    )
    for wavelength_index in selected_wavelength_indices:
        if wavelength_index < 1 or wavelength_index > len(spec.wavelengths_um):
            raise ValueError("wavelength_indices must be 1-based Zemax wavelength numbers.")

    field_count = len(selected_field_points)
    wavelength_count = len(selected_wavelength_indices)
    ray_count = pupil_tensor.shape[0]
    batch_shape = (1, field_count, wavelength_count, ray_count)

    x = torch.empty(batch_shape, dtype=torch.float64)
    y = torch.empty(batch_shape, dtype=torch.float64)
    z = torch.empty(batch_shape, dtype=torch.float64)
    l = torch.empty(batch_shape, dtype=torch.float64)
    m = torch.empty(batch_shape, dtype=torch.float64)
    n = torch.empty(batch_shape, dtype=torch.float64)
    wavelength_um = torch.empty(batch_shape, dtype=torch.float64)

    edge_field_x_deg = max((abs(field_x) for field_x, _field_y in spec.field_points), default=0.0)
    edge_field_y_deg = max((abs(field_y) for _field_x, field_y in spec.field_points), default=0.0)

    tool = oss.Tools.OpenBatchRayTrace()
    try:
        # 枚举视场 (Hx, Hy)
        for field_index, (field_x_deg, field_y_deg) in enumerate(selected_field_points):
            normalized_hx = normalized_field_coordinate(field_x_deg, edge_field_x_deg)
            normalized_hy = normalized_field_coordinate(field_y_deg, edge_field_y_deg)
            # 枚举波长
            for output_wavelength_index, wavelength_index in enumerate(selected_wavelength_indices):
                wavelength_value_um = spec.wavelengths_um[wavelength_index - 1]
                # 枚举孔径坐标 (Px, Py)
                for pupil_index in range(ray_count):
                    pupil_x = float(pupil_tensor[pupil_index, 0])
                    pupil_y = float(pupil_tensor[pupil_index, 1])
                    # 获取实际光线(x, y, z, l, m, n)
                    ok, ray_x, ray_y, ray_z, ray_l, ray_m, ray_n = tool.GetDirectFieldCoordinates(
                        wavelength_index,
                        zp.constants.Tools.RayTrace.RaysType.Real,
                        normalized_hx,
                        normalized_hy,
                        pupil_x,
                        pupil_y,
                    )
                    if not ok:
                        raise ValueError("GetDirectFieldCoordinates failed.")

                    # 存储批量光线
                    x[0, field_index, output_wavelength_index, pupil_index] = ray_x
                    y[0, field_index, output_wavelength_index, pupil_index] = ray_y
                    z[0, field_index, output_wavelength_index, pupil_index] = ray_z
                    l[0, field_index, output_wavelength_index, pupil_index] = ray_l
                    m[0, field_index, output_wavelength_index, pupil_index] = ray_m
                    n[0, field_index, output_wavelength_index, pupil_index] = ray_n
                    wavelength_um[0, field_index, output_wavelength_index, pupil_index] = wavelength_value_um
    finally:
        tool.Close()

    return ZemaxDirectRaySet(
        x=x,
        y=y,
        z=z,
        l=l,
        m=m,
        n=n,
        wavelength_um=wavelength_um,
        wavelength_indices=selected_wavelength_indices,
        field_points=tuple(selected_field_points),
        pupil_coordinates=pupil_tensor,
        metadata={
            "batch_shape": batch_shape,
            "edge_field_x_deg": edge_field_x_deg,
            "edge_field_y_deg": edge_field_y_deg,
        },
    )


def move_direct_ray_set_to_device(
    ray_set: ZemaxDirectRaySet,
    *,
    device: torch.device,
) -> ZemaxDirectRaySet:
    """将 direct ray set 迁移到目标设备。"""

    return ZemaxDirectRaySet(
        x=ray_set.x.to(device=device),
        y=ray_set.y.to(device=device),
        z=ray_set.z.to(device=device),
        l=ray_set.l.to(device=device),
        m=ray_set.m.to(device=device),
        n=ray_set.n.to(device=device),
        wavelength_um=ray_set.wavelength_um.to(device=device),
        wavelength_indices=ray_set.wavelength_indices,
        field_points=ray_set.field_points,
        pupil_coordinates=ray_set.pupil_coordinates.to(device=device),
        metadata=dict(ray_set.metadata),
    )


def _create_zemax_direct_reader(
    tool: Any,
    spec: ZemaxSequentialSystemSpec,
    ray_set: ZemaxDirectRaySet,
    *,
    end_surface: int | None = None,
) -> Any:
    """创建 Zemax direct unpolarized reader，并写入全部光线。"""

    batch_shape = ray_set.x.shape
    ray_count = batch_shape[1] * batch_shape[2] * batch_shape[3]
    reader = tool.CreateDirectUnpol(
        ray_count,
        zp.constants.Tools.RayTrace.RaysType.Real,
        0,
        spec.image_surface_index if end_surface is None else int(end_surface),
    )
    for field_index in range(batch_shape[1]):
        for wavelength_index, wavelength_number in enumerate(ray_set.wavelength_indices):
            for pupil_index in range(batch_shape[3]):
                reader.AddRay(
                    wavelength_number,
                    float(ray_set.x[0, field_index, wavelength_index, pupil_index]),
                    float(ray_set.y[0, field_index, wavelength_index, pupil_index]),
                    float(ray_set.z[0, field_index, wavelength_index, pupil_index]),
                    float(ray_set.l[0, field_index, wavelength_index, pupil_index]),
                    float(ray_set.m[0, field_index, wavelength_index, pupil_index]),
                    float(ray_set.n[0, field_index, wavelength_index, pupil_index]),
                )
    return reader


def _read_zemax_direct_results(
    reader: Any,
    spec: ZemaxSequentialSystemSpec,
    ray_set: ZemaxDirectRaySet,
) -> ZemaxBatchTraceResult:
    """读取 Zemax direct unpolarized 追迹结果。"""

    batch_shape = ray_set.x.shape
    x_mm = torch.empty(batch_shape, dtype=torch.float64)
    y_mm = torch.empty(batch_shape, dtype=torch.float64)
    z_mm = torch.empty(batch_shape, dtype=torch.float64)
    direction_l = torch.empty(batch_shape, dtype=torch.float64)
    direction_m = torch.empty(batch_shape, dtype=torch.float64)
    direction_n = torch.empty(batch_shape, dtype=torch.float64)
    valid = torch.empty(batch_shape, dtype=torch.bool)
    error_codes = torch.empty(batch_shape, dtype=torch.int64)
    vignette_codes = torch.empty(batch_shape, dtype=torch.int64)

    for field_index in range(batch_shape[1]):
        for wavelength_index in range(batch_shape[2]):
            for pupil_index in range(batch_shape[3]):
                result = reader.ReadNextResult()
                if len(result) != 14:
                    raise ValueError(f"Unexpected CreateDirectUnpol result length: {len(result)}")
                ok = bool(result[0])
                error_code = int(result[2])
                vignette_code = int(result[3])

                x_mm[0, field_index, wavelength_index, pupil_index] = float(result[4])
                y_mm[0, field_index, wavelength_index, pupil_index] = float(result[5])
                z_mm[0, field_index, wavelength_index, pupil_index] = float(result[6])
                direction_l[0, field_index, wavelength_index, pupil_index] = float(result[7])
                direction_m[0, field_index, wavelength_index, pupil_index] = float(result[8])
                direction_n[0, field_index, wavelength_index, pupil_index] = float(result[9])
                error_codes[0, field_index, wavelength_index, pupil_index] = error_code
                vignette_codes[0, field_index, wavelength_index, pupil_index] = vignette_code
                valid[0, field_index, wavelength_index, pupil_index] = ok and error_code == 0 and vignette_code == 0

    return ZemaxBatchTraceResult(
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        direction_l=direction_l,
        direction_m=direction_m,
        direction_n=direction_n,
        valid=valid,
        error_codes=error_codes,
        vignette_codes=vignette_codes,
        metadata={
            "ray_count": batch_shape[1] * batch_shape[2] * batch_shape[3],
            "image_surface_index": spec.image_surface_index,
        },
    )


def benchmark_zemax_batch_direct_kernel(
    oss: Any,
    spec: ZemaxSequentialSystemSpec,
    ray_set: ZemaxDirectRaySet,
    *,
    warmup: int = 0,
    iterations: int = 1,
) -> ZemaxBatchKernelBenchmarkResult:
    """分段测量 Zemax BatchRayTrace，run_avg_ms 仅包含 RunAndWaitForCompletion。"""

    last_result = None
    total_add_rays_ms = 0.0
    total_run_ms = 0.0
    total_read_results_ms = 0.0

    for iteration_index in range(warmup + iterations):
        tool = oss.Tools.OpenBatchRayTrace()
        try:
            started_add = time.perf_counter()
            reader = _create_zemax_direct_reader(tool, spec, ray_set)
            elapsed_add_ms = (time.perf_counter() - started_add) * 1000.0

            started_run = time.perf_counter()
            tool.RunAndWaitForCompletion()
            elapsed_run_ms = (time.perf_counter() - started_run) * 1000.0

            started_read = time.perf_counter()
            result = _read_zemax_direct_results(reader, spec, ray_set)
            elapsed_read_ms = (time.perf_counter() - started_read) * 1000.0
        finally:
            tool.Close()

        if iteration_index < warmup:
            continue

        last_result = result
        total_add_rays_ms += elapsed_add_ms
        total_run_ms += elapsed_run_ms
        total_read_results_ms += elapsed_read_ms

    if last_result is None:
        raise ValueError("iterations must be greater than zero.")

    divisor = max(iterations, 1)
    return ZemaxBatchKernelBenchmarkResult(
        run_avg_ms=total_run_ms / divisor,
        add_rays_avg_ms=total_add_rays_ms / divisor,
        read_results_avg_ms=total_read_results_ms / divisor,
        last_result=last_result,
    )


def trace_zemax_batch_direct(
    oss: Any,
    spec: ZemaxSequentialSystemSpec,
    ray_set: ZemaxDirectRaySet,
    *,
    end_surface: int | None = None,
) -> ZemaxBatchTraceResult:
    """使用 ZOS-API BatchRayTrace 追迹显式光线。"""
    tool = oss.Tools.OpenBatchRayTrace()
    try:
        reader = _create_zemax_direct_reader(tool, spec, ray_set, end_surface=end_surface)
        tool.RunAndWaitForCompletion()
        result = _read_zemax_direct_results(reader, spec, ray_set)
    finally:
        tool.Close()
    result.metadata["end_surface"] = spec.image_surface_index if end_surface is None else int(end_surface)
    return result


def build_optics_core_direct_rays(
    system: oc.MultiOpticalSystem,
    ray_set: ZemaxDirectRaySet,
) -> oc.RayBundle:
    """将 Zemax 显式光线转换为 optics_core 运行设备上的 RayBundle。"""

    target_device = default_device(system)
    resolved_ray_set = move_direct_ray_set_to_device(ray_set, device=target_device)
    wavelength_index = torch.tensor(
        [index - 1 for index in resolved_ray_set.wavelength_indices],
        dtype=torch.int64,
        device=target_device,
    ).reshape(1, 1, -1, 1).expand_as(resolved_ray_set.x)
    return oc.RayBundle(
        x=resolved_ray_set.x,
        y=resolved_ray_set.y,
        z=resolved_ray_set.z,
        l=resolved_ray_set.l,
        m=resolved_ray_set.m,
        n=resolved_ray_set.n,
        wavelength_index=wavelength_index,
        intensity=torch.ones_like(resolved_ray_set.x, dtype=torch.float64),
        opl=torch.zeros_like(resolved_ray_set.x, dtype=torch.float64),
        metadata={
            "source": "zemax_direct_field_coordinates",
            "field_points": resolved_ray_set.field_points,
            "pupil_coordinates": resolved_ray_set.pupil_coordinates,
            "runtime_device": str(target_device),
        },
    )


def trace_optics_core_direct(
    system: oc.MultiOpticalSystem,
    ray_set: ZemaxDirectRaySet,
    *,
    options: oc.TraceOptions | None = None,
) -> oc.TraceResult:
    """使用显式光线直接驱动 optics_core 顺序追迹。"""

    system._material_data = compile_batched_material_data(system, device=default_device(system))
    rays = build_optics_core_direct_rays(system, ray_set)
    return system.tracer.trace(
        system,
        rays,
        options=options or oc.TraceOptions(record_intersections=False),
    )
