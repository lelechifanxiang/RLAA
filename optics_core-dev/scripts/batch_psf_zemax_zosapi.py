from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.batch_spot import save_summary_json
from scripts.batch_mtf_zemax_zosapi import (
    PythonStandaloneApplication,
    analysis_settings,
    apply_assembly_tolerance_to_zosapi,
    load_lens_system,
    sample_size_constant,
)
from scripts.batch_tolerance_common import (
    DEFAULT_COORDINATE_BREAK_PAIRS,
    DEFAULT_MONTE_CARLO_DESIGN_COUNT,
    DEFAULT_RANDOM_SEED,
    DOUBLE_GAUSS_CB_ZMX_PATH,
    build_random_assembly_tolerance_design_records,
    monte_carlo_tolerance_summary_fields,
    parse_coordinate_break_pairs,
)
from zemax_utils.zmx_loader import load_zmx_sequential_system_spec


DEFAULT_DEVICE = "cuda:0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples/output/batch_psf_zemax_zosapi"
DEFAULT_PUPIL_SAMPLE_COUNT = 32
DEFAULT_IMAGE_SAMPLE_COUNT = 32
DEFAULT_IMAGE_DELTA_UM = 0.0
PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]
DEFAULT_ZMX_PATH = REPO_ROOT / DOUBLE_GAUSS_CB_ZMX_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="直接调用 ZOS API 进行装配公差 Monte Carlo Huygens PSF 图片导出")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="接口对齐参数，ZOS API 实际不区分 CPU/GPU")
    parser.add_argument("--zmx-path", default=str(DEFAULT_ZMX_PATH), help=f"ZMX 文件路径，默认 {DEFAULT_ZMX_PATH}")
    parser.add_argument(
        "--coordinate-break-pairs",
        nargs="+",
        type=int,
        default=[number for pair in DEFAULT_COORDINATE_BREAK_PAIRS for number in pair],
        help="按 first return 成对输入 Zemax CB 面号，例如 1 4 7 10",
    )
    parser.add_argument("--design-count", type=int, default=DEFAULT_MONTE_CARLO_DESIGN_COUNT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--pupil-sample-count", type=int, default=DEFAULT_PUPIL_SAMPLE_COUNT)
    parser.add_argument("--image-sample-count", type=int, default=DEFAULT_IMAGE_SAMPLE_COUNT)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary-json", default=None, help="导出本次性能摘要")
    parser.add_argument("--skip-images", action="store_true", help="只执行 Huygens PSF 计算，不导出 PNG 图片")
    return parser.parse_args()


def configure_huygens_psf(
    analysis: Any,
    ZOSAPI: Any,
    *,
    pupil_sample_count: int,
    image_sample_count: int,
    field_index: int,
) -> None:
    settings = analysis_settings(analysis)
    settings.PupilSampleSize = sample_size_constant(ZOSAPI, pupil_sample_count)
    settings.ImageSampleSize = sample_size_constant(ZOSAPI, image_sample_count)
    settings.ImageDelta = DEFAULT_IMAGE_DELTA_UM
    settings.Normalize = False
    settings.Wavelength.SetWavelengthNumber(0)
    settings.Field.SetFieldNumber(int(field_index) + 1)


def normalize_to_uint8(psf: np.ndarray) -> np.ndarray:
    psf = np.asarray(psf, dtype=np.float64)
    value_range = float(psf.max() - psf.min())
    if value_range <= np.finfo(np.float64).eps:
        return np.zeros(psf.shape, dtype=np.uint8)
    return np.rint((psf - psf.min()) / value_range * 255.0).clip(0, 255).astype(np.uint8)


def fetch_zosapi_psf_image(analysis: Any, ZOSAPI: Any, *, pupil_sample_count: int, image_sample_count: int, field_index: int) -> np.ndarray:
    configure_huygens_psf(
        analysis,
        ZOSAPI,
        pupil_sample_count=pupil_sample_count,
        image_sample_count=image_sample_count,
        field_index=field_index,
    )
    analysis.ApplyAndWaitForCompletion()
    values = analysis.GetResults().GetDataGrid(0).Values
    return np.flip(np.array(values, dtype=np.float64), axis=0)


def save_psf_png(path: Path, psf: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), normalize_to_uint8(psf), PNG_WRITE_PARAMS):
        raise RuntimeError(f"failed to write PSF png: {path}")


def run_zemax_batch_psf_zosapi(
    *,
    zmx_path: Path,
    design_count: int,
    random_seed: int,
    pupil_sample_count: int,
    image_sample_count: int,
    output_dir: Path,
    device_argument: str,
    skip_images: bool,
    coordinate_break_pairs: tuple[tuple[int, int], ...] = DEFAULT_COORDINATE_BREAK_PAIRS,
) -> dict[str, object]:
    spec = load_zmx_sequential_system_spec(zmx_path)
    design_records = build_random_assembly_tolerance_design_records(
        int(design_count),
        int(random_seed),
        coordinate_break_pair_count=len(coordinate_break_pairs),
    )
    field_points = tuple((float(field_x), float(field_y)) for field_x, field_y in spec.field_points)
    wavelengths_um = [float(wavelength_um) for wavelength_um in spec.wavelengths_um]

    print(f"zmx 文件: {zmx_path}")
    print(f"device 参数: {device_argument} 仅用于接口对齐，ZOS API 不区分 CPU/GPU")
    print(f"坐标间断面对: {coordinate_break_pairs}")
    print(f"视场点: {field_points}")
    print(f"波长(um): {wavelengths_um}")
    print(f"总设计数: {len(design_records)}")
    print(f"Monte Carlo 随机种子: {random_seed}")

    computed_image_count = 0
    saved_image_count = 0
    compute_elapsed_seconds = 0.0
    export_elapsed_seconds = 0.0
    total_started_at = time.perf_counter()
    zos: PythonStandaloneApplication | None = None
    try:
        zos = PythonStandaloneApplication()
        ZOSAPI = zos.ZOSAPI
        the_system = zos.TheSystem
        load_lens_system(the_system, zmx_path)
        analysis = the_system.Analyses.New_HuygensPsf()

        for design_index, design_record in enumerate(design_records):
            apply_assembly_tolerance_to_zosapi(the_system, design_record, coordinate_break_pairs)
            for field_index in range(len(field_points)):
                compute_started_at = time.perf_counter()
                psf = fetch_zosapi_psf_image(
                    analysis,
                    ZOSAPI,
                    pupil_sample_count=int(pupil_sample_count),
                    image_sample_count=int(image_sample_count),
                    field_index=field_index,
                )
                compute_elapsed_seconds += time.perf_counter() - compute_started_at
                computed_image_count += 1
                if not skip_images:
                    export_started_at = time.perf_counter()
                    save_psf_png(output_dir / f"zemax_psf_design_{design_index:04d}_field_{field_index}.png", psf)
                    export_elapsed_seconds += time.perf_counter() - export_started_at
                    saved_image_count += 1
            if (design_index + 1) % 20 == 0 or design_index + 1 == len(design_records):
                print(f"进度: {design_index + 1}/{len(design_records)}")
    finally:
        if zos is not None:
            del zos

    elapsed_seconds = time.perf_counter() - total_started_at
    total_pupil_sample_count = len(design_records) * len(field_points) * len(wavelengths_um) * int(pupil_sample_count) ** 2
    total_phase_sample_count = total_pupil_sample_count * int(image_sample_count) ** 2
    psf_per_second = saved_image_count / elapsed_seconds if saved_image_count > 0 else 0.0
    compute_psf_per_second = computed_image_count / compute_elapsed_seconds if compute_elapsed_seconds > 0.0 else 0.0

    print("ZOS API Huygens PSF 信息:")
    print(f"design_count={len(design_records)}, field_count={len(field_points)}, wavelength_count={len(wavelengths_um)}")
    print(f"pupil_sample_count={pupil_sample_count}, image_sample_count={image_sample_count}")
    print(f"detected_design_batch_size=1, design_batch_size=1, minibatch_count={computed_image_count}")
    print(f"elapsed_seconds={elapsed_seconds:.6f}, psf_images_per_second={psf_per_second:.3f}")
    print(f"compute_elapsed_seconds={compute_elapsed_seconds:.6f}, compute_psf_images_per_second={compute_psf_per_second:.3f}")
    print(f"export_elapsed_seconds={export_elapsed_seconds:.6f}, skip_images={skip_images}")
    print("peak_allocated_gib=0.000")
    print("peak_reserved_gib=0.000")
    print(f"total_pupil_sample_count={total_pupil_sample_count}")
    print(f"total_phase_sample_count={total_phase_sample_count}")
    print(f"computed_image_count={computed_image_count}")
    print(f"saved_image_count={saved_image_count}")
    print(f"output_dir={output_dir}")

    return {
        "analysis_type": "psf",
        "source": "zosapi.HuygensPSF",
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
        "design_count": len(design_records),
        "field_count": len(field_points),
        "wavelengths_um": wavelengths_um,
        "wavelength_count": len(wavelengths_um),
        "pupil_sample_count": int(pupil_sample_count),
        "image_sample_count": int(image_sample_count),
        "image_delta_um": DEFAULT_IMAGE_DELTA_UM,
        "sample_count": int(pupil_sample_count) if int(pupil_sample_count) == int(image_sample_count) else None,
        "case_name": "sample_count",
        "case_value": int(pupil_sample_count),
        "total_pupil_sample_count": total_pupil_sample_count,
        "total_phase_sample_count": total_phase_sample_count,
        "detected_design_batch_size": 1,
        "design_batch_size": 1,
        "minibatch_count": computed_image_count,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_ms": elapsed_seconds * 1000.0,
        "compute_elapsed_seconds": compute_elapsed_seconds,
        "compute_elapsed_ms": compute_elapsed_seconds * 1000.0,
        "export_elapsed_seconds": export_elapsed_seconds,
        "export_elapsed_ms": export_elapsed_seconds * 1000.0,
        "psf_images_per_second": psf_per_second,
        "compute_psf_images_per_second": compute_psf_per_second,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "computed_image_count": computed_image_count,
        "saved_image_count": saved_image_count,
        "skip_images": bool(skip_images),
        "output_dir": str(output_dir),
    }


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

    summary = run_zemax_batch_psf_zosapi(
        zmx_path=zmx_path,
        design_count=int(args.design_count),
        random_seed=int(args.random_seed),
        pupil_sample_count=int(args.pupil_sample_count),
        image_sample_count=int(args.image_sample_count),
        output_dir=Path(args.output_dir),
        device_argument=str(args.device),
        skip_images=bool(args.skip_images),
        coordinate_break_pairs=coordinate_break_pairs,
    )
    if args.summary_json is not None:
        save_summary_json(Path(args.summary_json), summary)


if __name__ == "__main__":
    main()
