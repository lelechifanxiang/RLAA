from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.batch_analysis_common import (
    CURVATURE_DELTA_INV_MM,
    DEFAULT_DEVICE,
    DEFAULT_SURFACE_NUMBERS,
    DOUBLE_GAUSS_ZMX_PATH,
    THICKNESS_DELTA_MM,
    build_design_records,
    parse_surface_numbers,
    save_summary_json,
)
from zemax_utils.common import loaded_sequential_system


try:
    import zospy as zp
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("运行 batch_spot_zemax.py 需要先安装 zospy 并配置 Zemax。") from exc


DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples/output/batch_spot_scaling"
DEFAULT_RAY_DENSITY = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 Zemax API 进行批量点列图分析")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="接口对齐参数，Zemax API 实际不区分 CPU/GPU，默认 cpu",
    )
    parser.add_argument(
        "--surfaces",
        nargs="+",
        default=[str(number) for number in DEFAULT_SURFACE_NUMBERS],
        help="Zemax 面号列表，支持空格分隔或逗号分隔，默认 1 2",
    )
    parser.add_argument(
        "--ray-density",
        type=int,
        default=DEFAULT_RAY_DENSITY,
        help="spot 六边采样密度，默认 30",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="CSV 输出路径；未提供时按 output 目录自动生成",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="只执行分析并输出统计信息，不保存逐设计 CSV",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="JSON 输出路径；未提供时按 output 目录自动生成",
    )
    return parser.parse_args()
def hexapolar_pupil_ray_count(ray_density: int) -> int:
    return 1 + 3 * ray_density * (ray_density + 1)


def pattern_constant() -> object:
    return zp.constants.Analysis.Settings.Spot.Patterns.Hexapolar
def configure_standard_spot(analysis, *, ray_density: int) -> None:
    settings = analysis.GetSettings()
    settings.Field.UseAllFields()
    settings.Wavelength.UseAllWavelengths()
    settings.Surface.UseImageSurface()
    settings.Pattern = pattern_constant()
    settings.RayDensity = int(ray_density)
    settings.ReferTo = zp.constants.Analysis.Settings.Spot.Reference.ChiefRay


def apply_design_to_system(
    oss,
    *,
    zemax_surface_numbers: list[int],
    design_record: dict[str, float],
) -> None:
    for zemax_surface_number in zemax_surface_numbers:
        surface = oss.LDE.GetSurfaceAt(int(zemax_surface_number))
        surface.Thickness = float(design_record[f"s{zemax_surface_number}_thickness_mm"])
        surface.Radius = float(design_record[f"s{zemax_surface_number}_radius_mm"])


def extract_standard_spot_metrics(analysis) -> tuple[list[float], list[float]]:
    spot_data = analysis.GetResults().SpotData
    field_count = int(spot_data.NumberOfFields)
    rms_radius_um = [
        float(spot_data.GetRMSSpotSizeFor(field_index, 0))
        for field_index in range(1, field_count + 1)
    ]
    geo_radius_um = [
        float(spot_data.GetGeoSpotSizeFor(field_index, 0))
        for field_index in range(1, field_count + 1)
    ]
    return rms_radius_um, geo_radius_um


def run_zemax_batch_spot(
    *,
    zmx_path: Path,
    zemax_surface_numbers: list[int],
    ray_density: int,
    device_argument: str,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    design_records, field_points, wavelengths_um, surface_indices = build_design_records(
        zmx_path=zmx_path,
        zemax_surface_numbers=zemax_surface_numbers,
        device_argument=device_argument,
    )

    pupil_ray_count = hexapolar_pupil_ray_count(ray_density)
    total_ray_count = len(design_records) * len(field_points) * len(wavelengths_um) * pupil_ray_count

    result_records: list[dict[str, float]] = []
    started_at = time.perf_counter()
    with loaded_sequential_system(zmx_path) as oss:
        analysis = oss.Analyses.New_StandardSpot()
        try:
            configure_standard_spot(analysis, ray_density=ray_density)
            for design_index, design_record in enumerate(design_records):
                apply_design_to_system(
                    oss,
                    zemax_surface_numbers=zemax_surface_numbers,
                    design_record=design_record,
                )
                analysis.ApplyAndWaitForCompletion()
                rms_radius_um, geo_radius_um = extract_standard_spot_metrics(analysis)

                record = dict(design_record)
                for field_index, rms_value in enumerate(rms_radius_um):
                    record[f"field_{field_index}_rms_radius_um"] = float(rms_value)
                for field_index, geo_value in enumerate(geo_radius_um):
                    record[f"field_{field_index}_geo_radius_um"] = float(geo_value)
                result_records.append(record)

                if (design_index + 1) % 100 == 0 or design_index + 1 == len(design_records):
                    print(f"进度: {design_index + 1}/{len(design_records)}")
        finally:
            analysis.Close()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    print("并行 spot 信息:")
    print(f"pattern=hexapolar, ray_density={ray_density}, pupil_ray_count={pupil_ray_count}")
    print(
        f"design_count={len(design_records)}, field_count={len(field_points)}, wavelength_count={len(wavelengths_um)}"
    )
    print(f"total_ray_count={total_ray_count}")
    print(f"Zemax spot 分析耗时: {elapsed_ms:.3f} ms")

    summary = {
        "analysis_type": "spot",
        "source": "ZOSAPI.StandardSpot.SpotData",
        "zmx_path": str(zmx_path.resolve()),
        "device": str(device_argument),
        "device_argument": str(device_argument),
        "device_note": "Zemax API 不区分 CPU/GPU，device 参数仅用于接口对齐",
        "pattern": "hexapolar",
        "ray_density": int(ray_density),
        "case_name": "ray_density",
        "case_value": int(ray_density),
        "pupil_ray_count": pupil_ray_count,
        "total_ray_count": total_ray_count,
        "zemax_surface_numbers": list(zemax_surface_numbers),
        "internal_surface_indices": list(surface_indices),
        "surface_count": len(zemax_surface_numbers),
        "field_points": [list(point) for point in field_points],
        "field_count": len(field_points),
        "wavelengths_um": wavelengths_um,
        "wavelength_count": len(wavelengths_um),
        "design_count": len(design_records),
        "thickness_delta_mm": THICKNESS_DELTA_MM,
        "curvature_delta_inv_mm": CURVATURE_DELTA_INV_MM,
        "elapsed_ms": elapsed_ms,
        "avg_design_ms": elapsed_ms / max(len(design_records), 1),
    }
    return result_records, summary


def save_csv(
    path: Path,
    *,
    zemax_surface_numbers: list[int],
    field_count: int,
    result_records: list[dict[str, float]],
) -> None:
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

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result_record in result_records:
            row: dict[str, str | int] = {"design_index": int(result_record["design_index"])}
            for fieldname in parameter_fieldnames:
                row[fieldname] = f"{float(result_record[fieldname]):.4f}"
            for fieldname in metric_fieldnames:
                row[fieldname] = f"{float(result_record[fieldname]):.3f}"
            writer.writerow(row)
    print(f"CSV 已保存: {path}")
def default_output_paths(
    *,
    output_dir: Path,
    zemax_surface_numbers: list[int],
    ray_density: int,
) -> tuple[Path, Path]:
    surface_slug = "-".join(str(number) for number in zemax_surface_numbers)
    run_slug = f"s{surface_slug}_d{ray_density}"
    csv_path = output_dir / f"zemax_batch_spot_{run_slug}.csv"
    summary_json_path = output_dir / f"zemax_batch_spot_{run_slug}.json"
    return csv_path, summary_json_path


def main() -> None:
    args = parse_args()
    if args.ray_density < 3:
        raise ValueError("ray_density 必须大于等于 3。")

    zmx_path = REPO_ROOT / DOUBLE_GAUSS_ZMX_PATH
    zemax_surface_numbers = parse_surface_numbers(args.surfaces)
    output_dir = Path(args.output)
    default_csv_path, default_summary_json_path = default_output_paths(
        output_dir=output_dir,
        zemax_surface_numbers=zemax_surface_numbers,
        ray_density=int(args.ray_density),
    )
    csv_path = None if args.skip_csv else Path(args.csv_path) if args.csv_path is not None else default_csv_path
    summary_json_path = Path(args.summary_json) if args.summary_json is not None else default_summary_json_path

    result_records, summary = run_zemax_batch_spot(
        zmx_path=zmx_path,
        zemax_surface_numbers=zemax_surface_numbers,
        ray_density=int(args.ray_density),
        device_argument=str(args.device),
    )
    summary["csv_path"] = str(csv_path) if csv_path is not None else None
    summary["summary_json_path"] = str(summary_json_path)
    summary["output_dir"] = str(output_dir)

    if csv_path is not None:
        save_csv(
            csv_path,
            zemax_surface_numbers=zemax_surface_numbers,
            field_count=int(summary["field_count"]),
            result_records=result_records,
        )
    save_summary_json(summary_json_path, summary)

    print("Zemax 批量点列图分析完成。")
    print(f"pupil_ray_count={summary['pupil_ray_count']}")
    print(f"total_ray_count={summary['total_ray_count']}")
    print(f"elapsed_ms={float(summary['elapsed_ms']):.3f}")


if __name__ == "__main__":
    main()
