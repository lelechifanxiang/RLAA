from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.batch_analysis_common import (
    DEFAULT_DEVICE,
    save_summary_json,
)
from scripts.batch_tolerance_common import (
    DEFAULT_COORDINATE_BREAK_PAIRS,
    DEFAULT_MONTE_CARLO_DESIGN_COUNT,
    DEFAULT_RANDOM_SEED,
    DOUBLE_GAUSS_CB_ZMX_PATH,
    TOLERANCE_PARAMETER_FIELDNAMES,
    apply_assembly_tolerance_to_zemax,
    build_random_assembly_tolerance_design_records,
    monte_carlo_tolerance_summary_fields,
    parse_coordinate_break_pairs,
    tolerance_parameter_fieldnames,
)
from zemax_utils.zmx_loader import load_zmx_sequential_system_spec


try:
    from tests.zemax.common import loaded_sequential_system
    from tests.zemax.huygens_mtf import fetch_zemax_huygens_mtf_from_spec
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("运行 batch_mtf_zemax.py 需要先安装 zospy 并配置 Zemax。") from exc


DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples/output/batch_mtf_scaling"
DEFAULT_FREQUENCIES_LP_PER_MM = (50.0, 100.0)
DEFAULT_PUPIL_SAMPLE_COUNT = 32
DEFAULT_IMAGE_SAMPLE_COUNT = 32
DEFAULT_IMAGE_DELTA_UM = 0.0
DEFAULT_ZMX_PATH = REPO_ROOT / DOUBLE_GAUSS_CB_ZMX_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 Zemax API 进行装配公差 Monte Carlo Huygens MTF 分析")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="接口对齐参数，Zemax API 实际不区分 CPU/GPU，默认 cpu",
    )
    parser.add_argument("--zmx-path", default=str(DEFAULT_ZMX_PATH), help=f"ZMX 文件路径，默认 {DEFAULT_ZMX_PATH}")
    parser.add_argument(
        "--coordinate-break-pairs",
        nargs="+",
        type=int,
        default=[number for pair in DEFAULT_COORDINATE_BREAK_PAIRS for number in pair],
        help="按 first return 成对输入 Zemax CB 面号，例如 1 4 7 10",
    )
    parser.add_argument(
        "--design-count",
        type=int,
        default=DEFAULT_MONTE_CARLO_DESIGN_COUNT,
        help=f"随机装配公差设计数量，默认 {DEFAULT_MONTE_CARLO_DESIGN_COUNT}",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help=f"随机种子，默认 {DEFAULT_RANDOM_SEED}")
    parser.add_argument(
        "--pupil-sample-count",
        type=int,
        default=DEFAULT_PUPIL_SAMPLE_COUNT,
        help=f"光瞳采样率，默认 {DEFAULT_PUPIL_SAMPLE_COUNT}",
    )
    parser.add_argument(
        "--image-sample-count",
        type=int,
        default=DEFAULT_IMAGE_SAMPLE_COUNT,
        help=f"像面采样率，默认 {DEFAULT_IMAGE_SAMPLE_COUNT}",
    )
    parser.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=list(DEFAULT_FREQUENCIES_LP_PER_MM),
        help="导出 CSV 的目标频率，单位 lp/mm，默认 50 100",
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


def frequency_token(frequency_lp_per_mm: float) -> str:
    rounded = round(float(frequency_lp_per_mm))
    if abs(float(frequency_lp_per_mm) - rounded) <= 1e-9:
        return str(int(rounded))
    return str(float(frequency_lp_per_mm)).replace(".", "p")


def sample_curve_at_frequency(
    frequencies_lp_per_mm: torch.Tensor,
    values: torch.Tensor,
    target_frequency_lp_per_mm: float,
) -> float:
    target = float(target_frequency_lp_per_mm)
    exact_match = torch.isclose(
        frequencies_lp_per_mm,
        torch.tensor(target, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    if torch.any(exact_match):
        return float(values[torch.nonzero(exact_match, as_tuple=False)[0, 0]].item())

    upper = int(torch.searchsorted(frequencies_lp_per_mm, torch.tensor(target, dtype=torch.float64)).item())
    if upper <= 0 or upper >= frequencies_lp_per_mm.numel():
        raise ValueError(f"目标频率 {target_frequency_lp_per_mm} lp/mm 超出 Zemax 输出范围。")
    lower = upper - 1
    lower_frequency = float(frequencies_lp_per_mm[lower].item())
    upper_frequency = float(frequencies_lp_per_mm[upper].item())
    weight = (target - lower_frequency) / (upper_frequency - lower_frequency)
    lower_value = float(values[lower].item())
    upper_value = float(values[upper].item())
    return lower_value + (upper_value - lower_value) * weight


def save_csv(
    path: Path,
    *,
    field_count: int,
    frequencies_lp_per_mm: tuple[float, ...],
    result_records: list[dict[str, float]],
    parameter_fieldnames: tuple[str, ...] = TOLERANCE_PARAMETER_FIELDNAMES,
) -> None:
    fieldnames = ["design_index"]
    fieldnames.extend(parameter_fieldnames)

    metric_fieldnames: list[str] = []
    for field_index in range(field_count):
        for frequency_lp_per_mm in frequencies_lp_per_mm:
            token = frequency_token(frequency_lp_per_mm)
            metric_fieldnames.append(f"field_{field_index}_freq_{token}_sagittal")
            metric_fieldnames.append(f"field_{field_index}_freq_{token}_tangential")
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
                row[fieldname] = f"{float(result_record[fieldname]):.6f}"
            writer.writerow(row)
    print(f"CSV 已保存: {path}")


def run_zemax_batch_mtf(
    *,
    zmx_path: Path,
    design_count: int,
    random_seed: int,
    pupil_sample_count: int,
    image_sample_count: int,
    frequencies_lp_per_mm: tuple[float, ...],
    device_argument: str,
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    spec = load_zmx_sequential_system_spec(zmx_path)
    design_records = build_random_assembly_tolerance_design_records(
        int(design_count),
        int(random_seed),
        coordinate_break_pair_count=len(coordinate_break_pairs),
    )
    field_points = tuple((float(field_x), float(field_y)) for field_x, field_y in spec.field_points)
    wavelengths_um = [float(wavelength_um) for wavelength_um in spec.wavelengths_um]

    maximum_frequency_lp_per_mm = max(frequencies_lp_per_mm)
    total_pupil_sample_count = (
        len(design_records)
        * len(field_points)
        * len(wavelengths_um)
        * int(pupil_sample_count)
        * int(pupil_sample_count)
    )
    total_phase_sample_count = total_pupil_sample_count * int(image_sample_count) * int(image_sample_count)

    result_records: list[dict[str, float]] = []
    started_at = time.perf_counter()
    with loaded_sequential_system(zmx_path) as oss:
        for design_index, design_record in enumerate(design_records):
            apply_assembly_tolerance_to_zemax(oss, design_record, coordinate_break_pairs)
            record = dict(design_record)
            for field_index in range(len(field_points)):
                reference = fetch_zemax_huygens_mtf_from_spec(
                    spec,
                    oss,
                    pupil_sample_count=int(pupil_sample_count),
                    image_sample_count=int(image_sample_count),
                    image_delta_um=DEFAULT_IMAGE_DELTA_UM,
                    maximum_frequency_lp_per_mm=float(maximum_frequency_lp_per_mm),
                    field_index=field_index,
                    wavelength_index=-1,
                )
                for frequency_lp_per_mm in frequencies_lp_per_mm:
                    token = frequency_token(frequency_lp_per_mm)
                    record[f"field_{field_index}_freq_{token}_sagittal"] = sample_curve_at_frequency(
                        torch.as_tensor(reference.frequencies_lp_per_mm, dtype=torch.float64),
                        torch.as_tensor(reference.sagittal, dtype=torch.float64),
                        frequency_lp_per_mm,
                    )
                    record[f"field_{field_index}_freq_{token}_tangential"] = sample_curve_at_frequency(
                        torch.as_tensor(reference.frequencies_lp_per_mm, dtype=torch.float64),
                        torch.as_tensor(reference.tangential, dtype=torch.float64),
                        frequency_lp_per_mm,
                    )
            result_records.append(record)

            if (design_index + 1) % 20 == 0 or design_index + 1 == len(design_records):
                print(f"进度: {design_index + 1}/{len(design_records)}")
    elapsed_seconds = time.perf_counter() - started_at
    elapsed_ms = elapsed_seconds * 1000.0
    design_count = len(design_records)
    designs_per_second = design_count / elapsed_seconds if elapsed_seconds > 0.0 else float("inf")
    mtf_shape = (design_count, len(field_points), len(frequencies_lp_per_mm))

    print("并行 Huygens MTF 信息:")
    print(f"design_count={design_count}, field_count={len(field_points)}, wavelength_count={len(wavelengths_um)}")
    print(f"random_seed={random_seed}")
    print(f"pupil_sample_count={pupil_sample_count}, image_sample_count={image_sample_count}")
    print(f"frequencies_lp_per_mm={list(frequencies_lp_per_mm)}")
    print(f"elapsed_seconds={elapsed_seconds:.6f}, designs_per_second={designs_per_second:.3f}")
    print(f"total_pupil_sample_count={total_pupil_sample_count}")
    print(f"total_phase_sample_count={total_phase_sample_count}")
    print(f"mtf_shape={mtf_shape}")
    print(f"Zemax MTF 分析耗时: {elapsed_ms:.3f} ms")

    sample_count = int(pupil_sample_count) if int(pupil_sample_count) == int(image_sample_count) else None
    summary = {
        "analysis_type": "mtf",
        "source": "zospy.HuygensMTF",
        "device": str(device_argument),
        "device_argument": str(device_argument),
        "device_note": "Zemax API 不区分 CPU/GPU，device 参数仅用于接口对齐",
        **monte_carlo_tolerance_summary_fields(
            int(design_count),
            int(random_seed),
            zmx_path=zmx_path,
            coordinate_break_pairs=coordinate_break_pairs,
        ),
        "field_points": [list(point) for point in field_points],
        "field_count": len(field_points),
        "wavelengths_um": wavelengths_um,
        "wavelength_count": len(wavelengths_um),
        "design_count": design_count,
        "frequencies_lp_per_mm": list(frequencies_lp_per_mm),
        "pupil_sample_count": int(pupil_sample_count),
        "image_sample_count": int(image_sample_count),
        "sample_count": sample_count,
        "case_name": "sample_count",
        "case_value": int(sample_count) if sample_count is not None else int(pupil_sample_count),
        "image_delta_um": DEFAULT_IMAGE_DELTA_UM,
        "maximum_frequency_lp_per_mm": float(maximum_frequency_lp_per_mm),
        "total_pupil_sample_count": total_pupil_sample_count,
        "total_phase_sample_count": total_phase_sample_count,
        "detected_design_batch_size": None,
        "design_batch_size": 1,
        "minibatch_count": None,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_ms": elapsed_ms,
        "designs_per_second": designs_per_second,
        "avg_design_ms": elapsed_ms / max(design_count, 1),
        "mtf_shape": list(mtf_shape),
    }
    return result_records, summary


def default_output_paths(
    *,
    output_dir: Path,
    design_count: int,
    random_seed: int,
    pupil_sample_count: int,
    image_sample_count: int,
) -> tuple[Path, Path]:
    run_slug = f"cbmc_n{design_count}_seed{random_seed}_p{pupil_sample_count}_i{image_sample_count}"
    csv_path = output_dir / f"zemax_batch_mtf_{run_slug}.csv"
    summary_json_path = output_dir / f"zemax_batch_mtf_{run_slug}.json"
    return csv_path, summary_json_path


def main() -> None:
    args = parse_args()
    if int(args.design_count) <= 0:
        raise ValueError("随机设计数量必须为正整数。")
    if int(args.pupil_sample_count) <= 0 or int(args.image_sample_count) <= 0:
        raise ValueError("采样率必须为正整数。")

    zmx_path = Path(args.zmx_path)
    if not zmx_path.is_absolute():
        zmx_path = REPO_ROOT / zmx_path
    coordinate_break_pairs = parse_coordinate_break_pairs(args.coordinate_break_pairs)
    frequencies_lp_per_mm = tuple(float(value) for value in args.frequencies)
    if not frequencies_lp_per_mm:
        raise ValueError("频率列表不能为空。")
    output_dir = Path(args.output)
    default_csv_path, default_summary_json_path = default_output_paths(
        output_dir=output_dir,
        design_count=int(args.design_count),
        random_seed=int(args.random_seed),
        pupil_sample_count=int(args.pupil_sample_count),
        image_sample_count=int(args.image_sample_count),
    )
    csv_path = None if args.skip_csv else Path(args.csv_path) if args.csv_path is not None else default_csv_path
    summary_json_path = Path(args.summary_json) if args.summary_json is not None else default_summary_json_path

    result_records, summary = run_zemax_batch_mtf(
        zmx_path=zmx_path,
        design_count=int(args.design_count),
        random_seed=int(args.random_seed),
        pupil_sample_count=int(args.pupil_sample_count),
        image_sample_count=int(args.image_sample_count),
        frequencies_lp_per_mm=frequencies_lp_per_mm,
        device_argument=str(args.device),
        coordinate_break_pairs=coordinate_break_pairs,
    )
    summary["csv_path"] = str(csv_path) if csv_path is not None else None
    summary["summary_json_path"] = str(summary_json_path)
    summary["output_dir"] = str(output_dir)

    if csv_path is not None:
        save_csv(
            csv_path,
            field_count=int(summary["field_count"]),
            frequencies_lp_per_mm=frequencies_lp_per_mm,
            result_records=result_records,
            parameter_fieldnames=tolerance_parameter_fieldnames(len(coordinate_break_pairs)),
        )
    save_summary_json(summary_json_path, summary)

    print("Zemax 批量 Huygens MTF 分析完成。")
    print(f"total_pupil_sample_count={summary['total_pupil_sample_count']}")
    print(f"total_phase_sample_count={summary['total_phase_sample_count']}")
    print(f"elapsed_ms={float(summary['elapsed_ms']):.3f}")


if __name__ == "__main__":
    main()
