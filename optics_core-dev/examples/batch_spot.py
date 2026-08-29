from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optics_core as oc
from optics_core.spot_diagram import (
    build_spot_diagram_sampler,
    compute_spot_metrics,
    extract_spot_data,
    trace_spot_diagram_rays,
)
from zemax_utils import build_optics_core_system_from_zmx_spec, load_zmx_sequential_system_spec


DOUBLE_GAUSS_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx")
OUTPUT_DIR = REPO_ROOT / "examples/output"
OUTPUT_CSV_PATH = OUTPUT_DIR / "batch_spot.csv"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_SURFACE_NUMBERS = (1, 2, 10, 11)
DEFAULT_RAY_DENSITY = 30
THICKNESS_DELTA_MM = 0.1
CURVATURE_DELTA_INV_MM = 1e-4


def synchronized_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双高斯批量点列图极限性能测试")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="运行设备，例如 cpu、cuda:0，默认 cuda:0",
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        default=[str(number) for number in DEFAULT_SURFACE_NUMBERS],
        help="Zemax 面号列表，支持空格分隔或逗号分隔，默认 1 2 10 11",
    )
    parser.add_argument(
        "--ray-density",
        type=int,
        default=DEFAULT_RAY_DENSITY,
        help="spot 六边采样密度，默认 30",
    )
    parser.add_argument(
        "--csv-path",
        default=str(OUTPUT_CSV_PATH),
        help=f"CSV 输出路径，默认 {OUTPUT_CSV_PATH}",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="只执行分析并输出统计信息，不保存逐设计 CSV",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="将本次运行的关键信息导出到 JSON 文件",
    )
    return parser.parse_args()


def parse_surface_numbers(tokens: list[str]) -> list[int]:
    numbers: list[int] = []
    for token in tokens:
        for part in token.split(","):
            stripped = part.strip()
            if stripped:
                numbers.append(int(stripped))
    if not numbers:
        raise ValueError("扰动面列表不能为空。")
    return numbers


def resolve_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求的设备 {device_text!r} 不可用，当前环境未检测到 CUDA。")
    return device


def resolve_surface_indices(
    spec,
    zemax_surface_numbers: list[int],
) -> list[int]:
    surface_indices: list[int] = []
    surface_count = len(spec.surfaces)
    for surface_number in zemax_surface_numbers:
        if surface_number < 1 or surface_number > surface_count:
            raise ValueError(
                f"Zemax 面号 {surface_number} 超出范围，当前系统可扰动面号范围为 1..{surface_count}。"
            )
        surface_indices.append(surface_number - 1)
    return surface_indices


def perturbation_triplet(base_value: float, delta: float) -> list[float]:
    return [base_value - delta, base_value, base_value + delta]


def curvature_triplet_to_radius_values(base_radius_mm: float) -> tuple[list[float], list[float]]:
    if abs(base_radius_mm) <= 1e-12:
        raise ValueError("当前脚本不支持对平面执行曲率扰动。")

    base_curvature = 1.0 / base_radius_mm
    curvature_values = perturbation_triplet(base_curvature, CURVATURE_DELTA_INV_MM)
    radius_values_mm = []
    for curvature in curvature_values:
        if abs(curvature) <= 1e-12:
            raise ValueError("曲率扰动后出现接近平面的半径，当前脚本不支持该情况。")
        radius_values_mm.append(1.0 / curvature)
    return curvature_values, radius_values_mm


def build_multi_system(
    zmx_path: Path,
    *,
    device: torch.device,
    zemax_surface_numbers: list[int],
) -> tuple[oc.MultiOpticalSystem, list[dict[str, float]], tuple[tuple[float, float], ...], list[int]]:
    # 加载规格
    spec = load_zmx_sequential_system_spec(zmx_path)
    # 构建系统
    base_system = build_optics_core_system_from_zmx_spec(spec)
    surface_indices = resolve_surface_indices(spec, zemax_surface_numbers)

    # 设置多重结构
    schema = oc.ParameterSchema()
    axes: list[oc.ParameterSweepAxis] = []
    for zemax_surface_number, surface_index in zip(zemax_surface_numbers, surface_indices, strict=True):
        surface_spec = spec.surfaces[surface_index]

        # 多重结构定义
        thickness_parameter = f"s{zemax_surface_number}_thickness_mm"
        radius_parameter = f"s{zemax_surface_number}_radius_mm"
        schema.add(
            thickness_parameter,
            f"surface[{surface_index}].gap.thickness",
            default=float(surface_spec.thickness_mm),
        )
        schema.add(
            radius_parameter,
            f"surface[{surface_index}].geometry.radius",
            default=float(surface_spec.radius_mm),
        )

        # 多重结构扫描参数
        thickness_values_mm = perturbation_triplet(float(surface_spec.thickness_mm), THICKNESS_DELTA_MM)
        _curvature_values_inv_mm, radius_values_mm = curvature_triplet_to_radius_values(float(surface_spec.radius_mm))
        axes.append(oc.ParameterSweepAxis(thickness_parameter, thickness_values_mm))
        axes.append(oc.ParameterSweepAxis(radius_parameter, radius_values_mm))

    # 构建多重结构向量
    parameters = oc.build_parameter_vector_grid(schema, axes)

    # 构建系统
    system = oc.MultiOpticalSystem(
        architecture=base_system.architecture,
        name=base_system.name,
        parameter_schema=schema,
        parameters=parameters,
        config=copy.deepcopy(base_system.config),
        tracer=base_system.tracer,
        materials=base_system.materials,
        fields=copy.deepcopy(list(base_system.fields)),
        wavelengths=copy.deepcopy(list(base_system.wavelengths)),
        aperture=copy.deepcopy(base_system.aperture),
    )
    system.config.backend.device = str(device)

    # 结果导出表头
    design_records: list[dict[str, float]] = []
    for design_index, parameter_vector in enumerate(system.parameters):
        record: dict[str, float] = {"design_index": design_index}
        for zemax_surface_number in zemax_surface_numbers:
            thickness_name = f"s{zemax_surface_number}_thickness_mm"
            radius_name = f"s{zemax_surface_number}_radius_mm"
            thickness_mm = float(parameter_vector[schema.index_of(thickness_name)])
            radius_mm = float(parameter_vector[schema.index_of(radius_name)])
            record[f"s{zemax_surface_number}_thickness_mm"] = thickness_mm
            record[f"s{zemax_surface_number}_radius_mm"] = radius_mm
        design_records.append(record)

    # 打印多重结构基本信息
    field_points = tuple((float(field.x), float(field.y)) for field in system.fields)
    print(f"zmx 文件: {zmx_path}")
    print(f"设备: {device}")
    print(f"Zemax 扰动面号: {zemax_surface_numbers}")
    print(f"内部 surface 索引: {surface_indices}")
    print(f"视场点: {field_points}")
    print(f"波长(um): {[float(wavelength.value_um) for wavelength in system.wavelengths]}")
    print(f"总设计数: {system.system_count}")
    print(f"固定厚度扰动: +/- {THICKNESS_DELTA_MM:.6f} mm")
    print(f"固定曲率扰动: +/- {CURVATURE_DELTA_INV_MM:.6e} 1/mm")
    return system, design_records, field_points, surface_indices


def run_parallel_spot(
    system: oc.MultiOpticalSystem,
    *,
    ray_density: int,
) -> tuple[list[dict[str, float]], dict[str, int | float | str]]:
    settings = oc.SpotDiagramSettings(
        pattern="hexapolar",
        ray_density=ray_density,
        save_path=None,
    )
    # 光线采样
    sampler = build_spot_diagram_sampler(settings)
    sample = sampler.sample()
    if sample.pupil_coordinates is None:
        raise ValueError("spot sampler did not produce pupil coordinates.")

    runtime_device = resolve_device(system.config.backend.device or "cpu")
    pupil_ray_count = int(sample.pupil_coordinates.shape[0])
    total_ray_count = system.system_count * len(system.fields) * len(system.wavelengths) * pupil_ray_count

    # 开始追迹
    started_at = synchronized_now(runtime_device)
    trace_result = trace_spot_diagram_rays(system, sample)
    elapsed_ms = (synchronized_now(runtime_device) - started_at) * 1000.0

    # 统计结果
    spot_data = extract_spot_data(system, trace_result, sample)
    rms_radius_um, geo_radius_um = compute_spot_metrics(system, spot_data)
    valid_ray_count = int(torch.as_tensor(trace_result.valid, dtype=torch.bool).sum().item())

    rms_tensor = torch.as_tensor(rms_radius_um, dtype=torch.float64).detach().cpu()
    geo_tensor = torch.as_tensor(geo_radius_um, dtype=torch.float64).detach().cpu()
    result_records: list[dict[str, float]] = []
    for design_index in range(system.system_count):
        record: dict[str, float] = {"design_index": design_index}
        for field_index in range(len(system.fields)):
            record[f"field_{field_index}_rms_radius_um"] = float(rms_tensor[design_index, field_index].item())
            record[f"field_{field_index}_geo_radius_um"] = float(geo_tensor[design_index, field_index].item())
        result_records.append(record)

    print("并行 spot 信息:")
    print(f"pattern=hexapolar, ray_density={ray_density}, pupil_ray_count={pupil_ray_count}")
    print(
        f"design_count={system.system_count}, field_count={len(system.fields)}, wavelength_count={len(system.wavelengths)}"
    )
    print(f"total_ray_count={total_ray_count}, valid_ray_count={valid_ray_count}")
    print(f"spot 分析耗时: {elapsed_ms:.3f} ms")

    return result_records, {
        "pattern": "hexapolar",
        "ray_density": ray_density,
        "pupil_ray_count": pupil_ray_count,
        "total_ray_count": total_ray_count,
        "valid_ray_count": valid_ray_count,
        "elapsed_ms": elapsed_ms,
    }


def save_csv(
    path: Path,
    *,
    zemax_surface_numbers: list[int],
    field_count: int,
    design_records: list[dict[str, float]],
    spot_records: list[dict[str, float]],
) -> None:
    if len(design_records) != len(spot_records):
        raise ValueError("design_records and spot_records length must match.")

    fieldnames = ["design_index"]
    parameter_fieldnames: list[str] = []
    for zemax_surface_number in zemax_surface_numbers:
        parameter_fieldnames.append(f"s{zemax_surface_number}_thickness_mm")
        parameter_fieldnames.append(f"s{zemax_surface_number}_radius_mm")
    fieldnames.extend(parameter_fieldnames)

    metric_fieldnames: list[str] = []
    for field_index in range(field_count):
        metric_fieldnames.append(f"field_{field_index}_rms_radius_um")
        metric_fieldnames.append(f"field_{field_index}_geo_radius_um")
    fieldnames.extend(metric_fieldnames)

    merged_rows: list[dict[str, str | int]] = []
    for design_record, spot_record in zip(design_records, spot_records, strict=True):
        merged_row: dict[str, str | int] = {"design_index": int(design_record["design_index"])}
        for fieldname in parameter_fieldnames:
            merged_row[fieldname] = f"{float(design_record[fieldname]):.4f}"
        for fieldname in metric_fieldnames:
            merged_row[fieldname] = f"{float(spot_record[fieldname]):.3f}"
        merged_rows.append(merged_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"CSV 已保存: {path}")


def save_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON 已保存: {path}")


def run_batch_spot_analysis(
    *,
    device: torch.device,
    surface_numbers: list[int],
    ray_density: int,
    csv_path: Path | None,
) -> dict[str, object]:
    # 1. 为双高斯zmx文件添加扰动，构建多重结构系统
    system, design_records, field_points, surface_indices = build_multi_system(
        REPO_ROOT / DOUBLE_GAUSS_ZMX_PATH,
        device=device,
        zemax_surface_numbers=surface_numbers,
    )
    system.prepare()

    # 2. 开始并行光线追迹
    spot_records, spot_summary = run_parallel_spot(
        system,
        ray_density=ray_density,
    )

    if csv_path is not None:
        save_csv(
            csv_path,
            zemax_surface_numbers=surface_numbers,
            field_count=len(system.fields),
            design_records=design_records,
            spot_records=spot_records,
        )

    return {
        "analysis_type": "spot",
        "zmx_path": str(REPO_ROOT / DOUBLE_GAUSS_ZMX_PATH),
        "device": str(device),
        "zemax_surface_numbers": list(surface_numbers),
        "internal_surface_indices": list(surface_indices),
        "surface_count": len(surface_numbers),
        "field_points": [list(point) for point in field_points],
        "field_count": len(system.fields),
        "wavelengths_um": [float(wavelength.value_um) for wavelength in system.wavelengths],
        "wavelength_count": len(system.wavelengths),
        "design_count": system.system_count,
        "thickness_delta_mm": THICKNESS_DELTA_MM,
        "curvature_delta_inv_mm": CURVATURE_DELTA_INV_MM,
        "case_name": "ray_density",
        "case_value": int(ray_density),
        "csv_path": str(csv_path) if csv_path is not None else None,
        **spot_summary,
    }


def main() -> None:
    # 1. 参数解析
    args = parse_args()
    device = resolve_device(args.device)
    surface_numbers = parse_surface_numbers(args.surfaces)
    if args.ray_density < 3:
        raise ValueError("ray_density 必须大于等于 3。")
    csv_path = None if args.skip_csv else Path(args.csv_path)

    # 2. 执行批量点列图分析
    summary = run_batch_spot_analysis(
        device=device,
        surface_numbers=surface_numbers,
        ray_density=args.ray_density,
        csv_path=csv_path,
    )

    # 3. 输出统计信息
    if args.summary_json is not None:
        save_summary_json(Path(args.summary_json), summary)

    print("批量点列图测试完成。")
    print(f"pupil_ray_count={summary['pupil_ray_count']}")
    print(f"total_ray_count={summary['total_ray_count']}")
    print(f"valid_ray_count={summary['valid_ray_count']}")
    print(f"elapsed_ms={float(summary['elapsed_ms']):.3f}")


if __name__ == "__main__":
    main()
