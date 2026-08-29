from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ZOSAPI_REFERENCE_DIR = REPO_ROOT / "reference/PSF0606-L1xy"
if str(ZOSAPI_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(ZOSAPI_REFERENCE_DIR))

from PythonStandaloneApplication import PythonStandaloneApplication
from scripts.batch_analysis_common import DEFAULT_DEVICE, save_summary_json
from scripts.batch_tolerance_common import (
    DEFAULT_COORDINATE_BREAK_PAIRS,
    DEFAULT_MONTE_CARLO_DESIGN_COUNT,
    DEFAULT_RANDOM_SEED,
    DOUBLE_GAUSS_CB_ZMX_PATH,
    TOLERANCE_PARAMETER_FIELDNAMES,
    build_random_assembly_tolerance_design_records,
    monte_carlo_tolerance_summary_fields,
    parse_coordinate_break_pairs,
    tolerance_parameter_fieldnames,
)
from zemax_utils.zmx_loader import load_zmx_sequential_system_spec


DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples/output/batch_mtf_scaling"
DEFAULT_FREQUENCIES_LP_PER_MM = (50.0, 100.0)
DEFAULT_PUPIL_SAMPLE_COUNT = 32
DEFAULT_IMAGE_SAMPLE_COUNT = 32
DEFAULT_IMAGE_DELTA_UM = 0.0
DEFAULT_ZMX_PATH = REPO_ROOT / DOUBLE_GAUSS_CB_ZMX_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="直接调用 ZOS API 进行装配公差 Monte Carlo Huygens MTF 分析")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="接口对齐参数，ZOS API 实际不区分 CPU/GPU，默认 cpu",
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


def sample_size_constant(ZOSAPI: Any, sample_count: int) -> Any:
    """把 32 转成 ZOSAPI.Analysis.SampleSizes.S_32x32。"""
    constant_name = f"S_{int(sample_count)}x{int(sample_count)}"
    try:
        return getattr(ZOSAPI.Analysis.SampleSizes, constant_name)
    except AttributeError as exc:
        raise ValueError(f"ZOS API 不支持采样规格 {constant_name}。") from exc


def mtf_type_modulation_constant(ZOSAPI: Any) -> Any:
    """读取 Huygens MTF 的 Modulation 类型常量。"""
    try:
        return ZOSAPI.Analysis.Settings.Mtf.HuygensMtfTypes.Modulation
    except AttributeError as exc:
        raise RuntimeError("当前 ZOS API 未暴露 HuygensMtfTypes.Modulation 常量。") from exc


def load_lens_system(the_system: Any, zmx_path: Path) -> None:
    """加载镜头文件并做基本校验。"""
    lens_file = str(zmx_path.resolve())
    if not Path(lens_file).is_file():
        raise FileNotFoundError(f"镜头文件不存在: {lens_file}")

    the_system.LoadFile(lens_file, False)
    surface_count = int(the_system.LDE.NumberOfSurfaces)
    if surface_count <= 0:
        raise RuntimeError(f"镜头文件加载异常: {lens_file}")
    print(f"已加载镜头文件: {lens_file}")
    print(f"表面总数: {surface_count}")


def apply_assembly_tolerance_to_zosapi(
    the_system: Any,
    design_record: dict[str, float],
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> None:
    """直接用 ZOS API 写入一组或多组 first CB 的 Par1..Par4。"""
    parameter_fieldnames = tolerance_parameter_fieldnames(len(coordinate_break_pairs))
    for pair_index, (first_surface_number, _return_surface_number) in enumerate(coordinate_break_pairs):
        surface = the_system.LDE.GetSurfaceAt(int(first_surface_number))
        values = (
            float(design_record[name])
            for name in parameter_fieldnames[pair_index * 4 : pair_index * 4 + 4]
        )
        for parameter_number, value in enumerate(values, start=1):
            # Zemax LDE 中 Par1..Par4 对应 cell 12..15。
            surface.GetCellAt(11 + int(parameter_number)).DoubleValue = value


def new_huygens_mtf_analysis(the_system: Any) -> Any:
    """创建 Huygens MTF 分析；不同版本方法名略有差异。"""
    analyses = the_system.Analyses
    if hasattr(analyses, "New_HuygensMtf"):
        return analyses.New_HuygensMtf()
    if hasattr(analyses, "New_HuygensMTF"):
        return analyses.New_HuygensMTF()
    raise RuntimeError("当前 ZOS API 未找到 New_HuygensMtf/New_HuygensMTF。")


def analysis_settings(analysis: Any) -> Any:
    """获取原生 settings 对象。"""
    settings = analysis.GetSettings() if hasattr(analysis, "GetSettings") else analysis.Settings
    return getattr(settings, "__implementation__", settings)


def configure_huygens_mtf(
    analysis: Any,
    ZOSAPI: Any,
    *,
    pupil_sample_count: int,
    image_sample_count: int,
    image_delta_um: float,
    maximum_frequency_lp_per_mm: float,
    field_index: int,
) -> None:
    """配置与 zospy.HuygensMTF 相同的分析参数。"""
    settings = analysis_settings(analysis)
    settings.PupilSampleSize = sample_size_constant(ZOSAPI, pupil_sample_count)
    settings.ImageSampleSize = sample_size_constant(ZOSAPI, image_sample_count)
    settings.ImageDelta = float(image_delta_um)
    settings.Wavelength.SetWavelengthNumber(0)
    settings.Field.SetFieldNumber(int(field_index) + 1)
    settings.Type = mtf_type_modulation_constant(ZOSAPI)
    settings.MaximumFrequency = float(maximum_frequency_lp_per_mm)
    settings.UsePolarization = False
    settings.UseDashes = False


def unpack_data_series(data_series: Any) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """把 ZOS API DataSeries 转为频率、标签和数值矩阵。"""
    frequencies = torch.tensor([float(value) for value in data_series.XData.Data], dtype=torch.float64)
    labels = [str(label) for label in data_series.SeriesLabels]
    rows = int(data_series.YData.Rows)
    cols = int(data_series.YData.Cols)
    values = torch.tensor([float(value) for value in data_series.YData.Data], dtype=torch.float64).reshape(rows, cols)
    return frequencies, labels, values


def extract_huygens_mtf_curves(analysis: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 Huygens MTF 结果中提取 sagittal/tangential 曲线。"""
    results = analysis.GetResults() if hasattr(analysis, "GetResults") else analysis.Results
    series_count = int(results.NumberOfDataSeries)
    if series_count <= 0:
        raise ValueError("ZOS API Huygens MTF did not return a data series.")

    all_labels: list[str] = []
    all_values: list[torch.Tensor] = []
    frequencies: torch.Tensor | None = None
    for series_index in range(series_count):
        data_series = results.DataSeries[series_index]
        series_frequencies, labels, values = unpack_data_series(data_series)
        if frequencies is None:
            frequencies = series_frequencies
        else:
            torch.testing.assert_close(frequencies, series_frequencies)
        all_labels.extend(labels)
        all_values.extend(values[:, column_index] for column_index in range(values.shape[1]))

    if frequencies is None:
        raise ValueError("ZOS API Huygens MTF did not return frequencies.")
    lower_labels = [label.lower() for label in all_labels]
    sagittal_index = next((index for index, label in enumerate(lower_labels) if "sagittal" in label), None)
    tangential_index = next((index for index, label in enumerate(lower_labels) if "tangential" in label), None)
    if sagittal_index is None or tangential_index is None:
        if len(all_values) < 2:
            raise ValueError(f"无法从 ZOS API Huygens MTF 标签中识别 S/T 曲线: {all_labels}")
        sagittal_index = 0
        tangential_index = 1
    return frequencies, all_values[sagittal_index], all_values[tangential_index]


def fetch_zosapi_huygens_mtf(
    analysis: Any,
    ZOSAPI: Any,
    *,
    pupil_sample_count: int,
    image_sample_count: int,
    image_delta_um: float,
    maximum_frequency_lp_per_mm: float,
    field_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """直接通过 ZOS API 执行一次 Huygens MTF。"""
    configure_huygens_mtf(
        analysis,
        ZOSAPI,
        pupil_sample_count=pupil_sample_count,
        image_sample_count=image_sample_count,
        image_delta_um=image_delta_um,
        maximum_frequency_lp_per_mm=maximum_frequency_lp_per_mm,
        field_index=field_index,
    )
    analysis.ApplyAndWaitForCompletion()
    return extract_huygens_mtf_curves(analysis)


def run_zemax_batch_mtf_zosapi(
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

    print(f"zmx 文件: {zmx_path}")
    print(f"device 参数: {device_argument} 仅用于接口对齐，ZOS API 不区分 CPU/GPU")
    print(f"坐标间断面对: {coordinate_break_pairs}")
    print(f"视场点: {field_points}")
    print(f"波长(um): {wavelengths_um}")
    print(f"总设计数: {len(design_records)}")
    print(f"Monte Carlo 随机种子: {random_seed}")

    result_records: list[dict[str, float]] = []
    zos: PythonStandaloneApplication | None = None
    started_at = time.perf_counter()
    try:
        zos = PythonStandaloneApplication()
        ZOSAPI = zos.ZOSAPI
        the_system = zos.TheSystem
        load_lens_system(the_system, zmx_path)
        analysis = new_huygens_mtf_analysis(the_system)

        for design_index, design_record in enumerate(design_records):
            apply_assembly_tolerance_to_zosapi(the_system, design_record, coordinate_break_pairs)
            record = dict(design_record)
            for field_index in range(len(field_points)):
                zemax_frequencies, sagittal, tangential = fetch_zosapi_huygens_mtf(
                    analysis,
                    ZOSAPI,
                    pupil_sample_count=int(pupil_sample_count),
                    image_sample_count=int(image_sample_count),
                    image_delta_um=DEFAULT_IMAGE_DELTA_UM,
                    maximum_frequency_lp_per_mm=float(maximum_frequency_lp_per_mm),
                    field_index=field_index,
                )
                for frequency_lp_per_mm in frequencies_lp_per_mm:
                    token = frequency_token(frequency_lp_per_mm)
                    record[f"field_{field_index}_freq_{token}_sagittal"] = sample_curve_at_frequency(
                        zemax_frequencies,
                        sagittal,
                        frequency_lp_per_mm,
                    )
                    record[f"field_{field_index}_freq_{token}_tangential"] = sample_curve_at_frequency(
                        zemax_frequencies,
                        tangential,
                        frequency_lp_per_mm,
                    )
            result_records.append(record)

            if (design_index + 1) % 20 == 0 or design_index + 1 == len(design_records):
                print(f"进度: {design_index + 1}/{len(design_records)}")
    finally:
        if zos is not None:
            del zos

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
        "source": "zosapi.HuygensMTF",
        "device": str(device_argument),
        "device_argument": str(device_argument),
        "device_note": "ZOS API 不区分 CPU/GPU，device 参数仅用于接口对齐",
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
    csv_path = output_dir / f"zemax_zosapi_batch_mtf_{run_slug}.csv"
    summary_json_path = output_dir / f"zemax_zosapi_batch_mtf_{run_slug}.json"
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

    result_records, summary = run_zemax_batch_mtf_zosapi(
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

    print("ZOS API 批量 Huygens MTF 分析完成。")
    print(f"total_pupil_sample_count={summary['total_pupil_sample_count']}")
    print(f"total_phase_sample_count={summary['total_phase_sample_count']}")
    print(f"elapsed_ms={float(summary['elapsed_ms']):.3f}")


if __name__ == "__main__":
    main()
