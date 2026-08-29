from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zemax_utils.zmx_loader import load_zmx_sequential_system_spec


DOUBLE_GAUSS_ZMX_PATH = Path("tests/zemax/zmx_files/Double Gauss 28 degree field real.zmx")
DEFAULT_DEVICE = "cpu"
DEFAULT_SURFACE_NUMBERS = (1, 2)
THICKNESS_DELTA_MM = 0.1
CURVATURE_DELTA_INV_MM = 1e-4


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


def resolve_surface_indices(
    surface_count: int,
    zemax_surface_numbers: list[int],
) -> list[int]:
    surface_indices: list[int] = []
    for surface_number in zemax_surface_numbers:
        if surface_number < 1 or surface_number > surface_count:
            raise ValueError(
                f"Zemax 面号 {surface_number} 超出范围，当前系统可扰动面号范围为 1..{surface_count}。"
            )
        surface_indices.append(surface_number - 1)
    return surface_indices


def perturbation_triplet(base_value: float, delta: float) -> list[float]:
    return [base_value - delta, base_value, base_value + delta]


def curvature_triplet_to_radius_values(base_radius_mm: float) -> list[float]:
    if abs(base_radius_mm) <= 1e-12:
        raise ValueError("当前脚本不支持对平面执行曲率扰动。")

    base_curvature = 1.0 / base_radius_mm
    curvature_values = perturbation_triplet(base_curvature, CURVATURE_DELTA_INV_MM)
    radius_values_mm: list[float] = []
    for curvature in curvature_values:
        if abs(curvature) <= 1e-12:
            raise ValueError("曲率扰动后出现接近平面的半径，当前脚本不支持该情况。")
        radius_values_mm.append(1.0 / curvature)
    return radius_values_mm


def build_design_records(
    *,
    zmx_path: Path,
    zemax_surface_numbers: list[int],
    device_argument: str,
) -> tuple[list[dict[str, float]], tuple[tuple[float, float], ...], list[float], list[int]]:
    """根据 zmx 规格构造所有扰动组合。"""
    spec = load_zmx_sequential_system_spec(zmx_path)
    surface_indices = resolve_surface_indices(len(spec.surfaces), zemax_surface_numbers)

    field_points = tuple((float(field_x), float(field_y)) for field_x, field_y in spec.field_points)
    wavelengths_um = [float(wavelength_um) for wavelength_um in spec.wavelengths_um]

    parameter_axes: list[tuple[str, list[float]]] = []
    for zemax_surface_number, surface_index in zip(zemax_surface_numbers, surface_indices, strict=True):
        surface_spec = spec.surfaces[surface_index]
        thickness_values_mm = perturbation_triplet(float(surface_spec.thickness_mm), THICKNESS_DELTA_MM)
        radius_values_mm = curvature_triplet_to_radius_values(float(surface_spec.radius_mm))
        parameter_axes.append((f"s{zemax_surface_number}_thickness_mm", thickness_values_mm))
        parameter_axes.append((f"s{zemax_surface_number}_radius_mm", radius_values_mm))

    parameter_names = [name for name, _values in parameter_axes]
    parameter_value_axes = [values for _name, values in parameter_axes]

    design_records: list[dict[str, float]] = []
    for design_index, parameter_values in enumerate(itertools.product(*parameter_value_axes)):
        record: dict[str, float] = {"design_index": design_index}
        for parameter_name, value in zip(parameter_names, parameter_values, strict=True):
            record[parameter_name] = float(value)
        design_records.append(record)

    print(f"zmx 文件: {zmx_path}")
    print(f"device 参数: {device_argument} 仅用于接口对齐，Zemax API 不区分 CPU/GPU")
    print(f"Zemax 扰动面号: {zemax_surface_numbers}")
    print(f"内部 surface 索引: {surface_indices}")
    print(f"视场点: {field_points}")
    print(f"波长(um): {wavelengths_um}")
    print(f"总设计数: {len(design_records)}")
    print(f"固定厚度扰动: +/- {THICKNESS_DELTA_MM:.6f} mm")
    print(f"固定曲率扰动: +/- {CURVATURE_DELTA_INV_MM:.6e} 1/mm")

    return design_records, field_points, wavelengths_um, surface_indices


def save_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON 已保存: {path}")
