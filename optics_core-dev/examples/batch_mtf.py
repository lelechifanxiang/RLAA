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

import optics_core as oc
from examples.batch_spot import (
    resolve_device,
    save_summary_json,
)
from scripts.batch_tolerance_common import (
    DEFAULT_MONTE_CARLO_DESIGN_COUNT,
    DEFAULT_RANDOM_SEED,
    DOUBLE_GAUSS_CB_ZMX_PATH,
    TOLERANCE_PARAMETER_FIELDNAMES,
    build_random_assembly_tolerance_multi_system,
    monte_carlo_tolerance_summary_fields,
)


DEFAULT_DEVICE = "cuda:0"
DEFAULT_OUTPUT_CSV_PATH = REPO_ROOT / "examples/output/batch_mtf.csv"
DEFAULT_PUPIL_SAMPLE_COUNT = 32
DEFAULT_IMAGE_SAMPLE_COUNT = 32
DEFAULT_FREQUENCIES_LP_PER_MM = (50.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双高斯装配公差 Monte Carlo 全波长 Huygens MTF 性能测试")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="运行设备，默认 cuda:0；CPU 可手动指定 cpu")
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
        "--csv-path",
        default=str(DEFAULT_OUTPUT_CSV_PATH),
        help=f"CSV 输出路径，默认 {DEFAULT_OUTPUT_CSV_PATH}",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="只执行分析并输出统计信息，不保存逐设计 CSV",
    )
    parser.add_argument("--summary-json", default=None, help="导出本次性能摘要")
    return parser.parse_args()


def synchronized_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def frequency_token(frequency_lp_per_mm: float) -> str:
    rounded = round(float(frequency_lp_per_mm))
    if abs(float(frequency_lp_per_mm) - rounded) <= 1e-9:
        return str(int(rounded))
    return str(float(frequency_lp_per_mm)).replace(".", "p")


def build_mtf_records(
    result: oc.MTFResult,
    *,
    field_indices: tuple[int, ...],
    frequencies_lp_per_mm: tuple[float, ...],
) -> list[dict[str, float]]:
    sagittal = torch.as_tensor(result.sagittal, dtype=torch.float64).detach().cpu()
    tangential = torch.as_tensor(result.tangential, dtype=torch.float64).detach().cpu()
    result_records: list[dict[str, float]] = []
    for design_index in range(sagittal.shape[0]):
        record: dict[str, float] = {"design_index": design_index}
        for local_field_index, field_index in enumerate(field_indices):
            for frequency_index, frequency_lp_per_mm in enumerate(frequencies_lp_per_mm):
                token = frequency_token(frequency_lp_per_mm)
                record[f"field_{field_index}_freq_{token}_sagittal"] = float(
                    sagittal[design_index, local_field_index, frequency_index].item()
                )
                record[f"field_{field_index}_freq_{token}_tangential"] = float(
                    tangential[design_index, local_field_index, frequency_index].item()
                )
        result_records.append(record)
    return result_records


def save_csv(
    path: Path,
    *,
    field_indices: tuple[int, ...],
    frequencies_lp_per_mm: tuple[float, ...],
    design_records: list[dict[str, float]],
    mtf_records: list[dict[str, float]],
) -> None:
    if len(design_records) != len(mtf_records):
        raise ValueError("design_records and mtf_records length must match.")

    fieldnames = ["design_index"]
    parameter_fieldnames = list(TOLERANCE_PARAMETER_FIELDNAMES)
    fieldnames.extend(parameter_fieldnames)

    metric_fieldnames: list[str] = []
    for field_index in field_indices:
        for frequency_lp_per_mm in frequencies_lp_per_mm:
            token = frequency_token(frequency_lp_per_mm)
            metric_fieldnames.append(f"field_{field_index}_freq_{token}_sagittal")
            metric_fieldnames.append(f"field_{field_index}_freq_{token}_tangential")
    fieldnames.extend(metric_fieldnames)

    merged_rows: list[dict[str, str | int]] = []
    for design_record, mtf_record in zip(design_records, mtf_records, strict=True):
        merged_row: dict[str, str | int] = {"design_index": int(design_record["design_index"])}
        for fieldname in parameter_fieldnames:
            merged_row[fieldname] = f"{float(design_record[fieldname]):.4f}"
        for fieldname in metric_fieldnames:
            merged_row[fieldname] = f"{float(mtf_record[fieldname]):.6f}"
        merged_rows.append(merged_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"CSV 已保存: {path}")


def run_batch_mtf_analysis(
    *,
    device: torch.device,
    design_count: int,
    random_seed: int,
    pupil_sample_count: int,
    image_sample_count: int,
    frequencies_lp_per_mm: tuple[float, ...],
    csv_path: Path | None,
) -> dict[str, object]:
    system, design_records, field_points, surface_indices = build_random_assembly_tolerance_multi_system(
        REPO_ROOT / DOUBLE_GAUSS_CB_ZMX_PATH,
        device=device,
        design_count=int(design_count),
        random_seed=int(random_seed),
    )
    system.prepare()
    field_indices = tuple(range(len(system.fields)))
    settings = oc.MTFSettings(
        pupil_sample_count=int(pupil_sample_count),
        image_sample_count=int(image_sample_count),
        frequencies_lp_per_mm=frequencies_lp_per_mm,
        field_indices=field_indices,
        wavelength_index=-1,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = synchronized_now(device)
    result = system.analysis.mtf(settings).run()
    elapsed_seconds = synchronized_now(device) - started_at

    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    if device.type == "cuda":
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))

    mtf_records = build_mtf_records(
        result,
        field_indices=field_indices,
        frequencies_lp_per_mm=frequencies_lp_per_mm,
    )
    if csv_path is not None:
        save_csv(
            csv_path,
            field_indices=field_indices,
            frequencies_lp_per_mm=frequencies_lp_per_mm,
            design_records=design_records,
            mtf_records=mtf_records,
        )

    designs_per_second = system.system_count / elapsed_seconds
    mtf_shape = tuple(torch.as_tensor(result.sagittal).shape)
    total_pupil_sample_count = (
        system.system_count
        * len(field_indices)
        * len(system.wavelengths)
        * int(pupil_sample_count)
        * int(pupil_sample_count)
    )
    total_phase_sample_count = total_pupil_sample_count * int(image_sample_count) * int(image_sample_count)
    print("并行 Huygens MTF 信息:")
    print(f"design_count={system.system_count}, field_count={len(field_indices)}, wavelength_count={len(system.wavelengths)}")
    print(f"pupil_sample_count={pupil_sample_count}, image_sample_count={image_sample_count}")
    print(f"frequencies_lp_per_mm={list(frequencies_lp_per_mm)}")
    print(
        f"detected_design_batch_size={result.detected_design_batch_size}, "
        f"design_batch_size={result.design_batch_size}, minibatch_count={result.minibatch_count}"
    )
    print(f"elapsed_seconds={elapsed_seconds:.6f}, designs_per_second={designs_per_second:.3f}")
    print(f"peak_allocated_gib={peak_allocated_bytes / 2**30:.3f}")
    print(f"peak_reserved_gib={peak_reserved_bytes / 2**30:.3f}")
    print(f"total_pupil_sample_count={total_pupil_sample_count}")
    print(f"total_phase_sample_count={total_phase_sample_count}")
    print(f"mtf_shape={mtf_shape}")

    sample_count = int(pupil_sample_count) if int(pupil_sample_count) == int(image_sample_count) else None
    return {
        "analysis_type": "mtf",
        "device": str(device),
        **monte_carlo_tolerance_summary_fields(int(design_count), int(random_seed)),
        "internal_surface_indices": list(surface_indices),
        "field_points": [list(point) for point in field_points],
        "design_count": system.system_count,
        "field_count": len(field_indices),
        "wavelength_count": len(system.wavelengths),
        "frequencies_lp_per_mm": list(frequencies_lp_per_mm),
        "pupil_sample_count": int(pupil_sample_count),
        "image_sample_count": int(image_sample_count),
        "sample_count": sample_count,
        "case_name": "sample_count",
        "case_value": int(sample_count) if sample_count is not None else int(pupil_sample_count),
        "total_pupil_sample_count": total_pupil_sample_count,
        "total_phase_sample_count": total_phase_sample_count,
        "detected_design_batch_size": result.detected_design_batch_size,
        "design_batch_size": result.design_batch_size,
        "minibatch_count": result.minibatch_count,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_ms": elapsed_seconds * 1000.0,
        "designs_per_second": designs_per_second,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "mtf_shape": list(mtf_shape),
        "csv_path": str(csv_path) if csv_path is not None else None,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if int(args.design_count) <= 0:
        raise ValueError("随机设计数量必须为正整数。")
    if int(args.pupil_sample_count) <= 0 or int(args.image_sample_count) <= 0:
        raise ValueError("采样率必须为正整数。")
    frequencies_lp_per_mm = tuple(float(value) for value in args.frequencies)
    if not frequencies_lp_per_mm:
        raise ValueError("频率列表不能为空。")
    csv_path = None if args.skip_csv else Path(args.csv_path)

    summary = run_batch_mtf_analysis(
        device=device,
        design_count=int(args.design_count),
        random_seed=int(args.random_seed),
        pupil_sample_count=int(args.pupil_sample_count),
        image_sample_count=int(args.image_sample_count),
        frequencies_lp_per_mm=frequencies_lp_per_mm,
        csv_path=csv_path,
    )
    if args.summary_json is not None:
        save_summary_json(Path(args.summary_json), summary)


if __name__ == "__main__":
    main()
