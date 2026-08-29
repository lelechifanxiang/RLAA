from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optics_core as oc
from examples.batch_mtf import DEFAULT_IMAGE_SAMPLE_COUNT, DEFAULT_PUPIL_SAMPLE_COUNT
from examples.batch_spot import resolve_device, save_summary_json
from scripts.batch_tolerance_common import (
    DEFAULT_MONTE_CARLO_DESIGN_COUNT,
    DEFAULT_RANDOM_SEED,
    DOUBLE_GAUSS_CB_ZMX_PATH,
    build_random_assembly_tolerance_multi_system,
    monte_carlo_tolerance_summary_fields,
)


DEFAULT_DEVICE = "cuda:0"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples/output/batch_psf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双高斯装配公差 Monte Carlo 全波长 Huygens PSF 图片导出")
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
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"PSF 图片输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--summary-json", default=None, help="导出本次性能摘要")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="只执行 Huygens PSF 计算，不导出 PNG 图片，用于测量纯计算性能",
    )
    return parser.parse_args()


def synchronized_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]


def save_psf_image(
    path: Path,
    *,
    image_uint8,
) -> None:
    if not cv2.imwrite(str(path), image_uint8, PNG_WRITE_PARAMS):
        raise RuntimeError(f"failed to write PSF png: {path}")


def normalize_psf_stack_to_uint8(psf_stack: torch.Tensor):
    images = torch.as_tensor(psf_stack, dtype=torch.float64)
    min_values = images.amin(dim=(-2, -1), keepdim=True)
    max_values = images.amax(dim=(-2, -1), keepdim=True)
    ranges = max_values - min_values
    normalized = torch.where(
        ranges > torch.finfo(torch.float64).eps,
        (images - min_values) / ranges.clamp_min(torch.finfo(torch.float64).eps),
        torch.zeros_like(images),
    )
    return torch.round(normalized * 255.0).clamp(0.0, 255.0).to(torch.uint8).cpu().contiguous().numpy()


def export_field_psf_images(
    result: oc.PSFResult,
    *,
    field_index: int,
    output_dir: Path,
) -> int:
    psf = torch.as_tensor(result.psf, dtype=torch.float64)
    images_uint8 = normalize_psf_stack_to_uint8(psf[:, 0])
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for design_index in range(psf.shape[0]):
        save_psf_image(
            output_dir / f"psf_design_{design_index:04d}_field_{field_index}.png",
            image_uint8=images_uint8[design_index],
        )
        saved_count += 1
    return saved_count


def run_batch_psf_analysis(
    *,
    device: torch.device,
    design_count: int,
    random_seed: int,
    pupil_sample_count: int,
    image_sample_count: int,
    output_dir: Path,
    skip_images: bool = False,
) -> dict[str, object]:
    system, _design_records, field_points, surface_indices = build_random_assembly_tolerance_multi_system(
        REPO_ROOT / DOUBLE_GAUSS_CB_ZMX_PATH,
        device=device,
        design_count=int(design_count),
        random_seed=int(random_seed),
    )
    system.prepare()
    field_indices = tuple(range(len(system.fields)))
    settings_base = dict(
        pupil_sample_count=int(pupil_sample_count),
        image_sample_count=int(image_sample_count),
        wavelength_index=-1,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_started_at = synchronized_now(device)
    compute_elapsed_seconds = 0.0
    export_elapsed_seconds = 0.0
    saved_image_count = 0
    computed_image_count = 0
    detected_design_batch_size = None
    design_batch_size = None
    minibatch_count = 0
    for field_index in field_indices:
        compute_started_at = synchronized_now(device)
        result = system.analysis.psf(oc.PSFSettings(field_index=field_index, **settings_base)).run()
        compute_elapsed_seconds += synchronized_now(device) - compute_started_at
        computed_image_count += int(torch.as_tensor(result.psf).shape[0])
        if not skip_images:
            export_started_at = time.perf_counter()
            saved_image_count += export_field_psf_images(result, field_index=field_index, output_dir=output_dir)
            export_elapsed_seconds += time.perf_counter() - export_started_at
        detected_design_batch_size = result.detected_design_batch_size
        design_batch_size = result.design_batch_size
        minibatch_count += int(result.minibatch_count)
    elapsed_seconds = synchronized_now(device) - total_started_at

    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    psf_per_second = saved_image_count / elapsed_seconds if saved_image_count > 0 else 0.0
    compute_psf_per_second = computed_image_count / compute_elapsed_seconds
    total_pupil_sample_count = (
        system.system_count * len(field_indices) * len(system.wavelengths) * int(pupil_sample_count) ** 2
    )
    total_phase_sample_count = total_pupil_sample_count * int(image_sample_count) ** 2

    print("并行 Huygens PSF 信息:")
    print(f"design_count={system.system_count}, field_count={len(field_indices)}, wavelength_count={len(system.wavelengths)}")
    print(f"pupil_sample_count={pupil_sample_count}, image_sample_count={image_sample_count}")
    print(
        f"detected_design_batch_size={detected_design_batch_size}, "
        f"design_batch_size={design_batch_size}, minibatch_count={minibatch_count}"
    )
    print(f"elapsed_seconds={elapsed_seconds:.6f}, psf_images_per_second={psf_per_second:.3f}")
    print(
        f"compute_elapsed_seconds={compute_elapsed_seconds:.6f}, "
        f"compute_psf_images_per_second={compute_psf_per_second:.3f}"
    )
    print(f"export_elapsed_seconds={export_elapsed_seconds:.6f}, skip_images={skip_images}")
    print(f"peak_allocated_gib={peak_allocated_bytes / 2**30:.3f}")
    print(f"peak_reserved_gib={peak_reserved_bytes / 2**30:.3f}")
    print(f"total_pupil_sample_count={total_pupil_sample_count}")
    print(f"total_phase_sample_count={total_phase_sample_count}")
    print(f"computed_image_count={computed_image_count}")
    print(f"saved_image_count={saved_image_count}")
    print(f"output_dir={output_dir}")

    return {
        "analysis_type": "psf",
        "device": str(device),
        **monte_carlo_tolerance_summary_fields(int(design_count), int(random_seed)),
        "internal_surface_indices": list(surface_indices),
        "field_points": [list(point) for point in field_points],
        "design_count": system.system_count,
        "field_count": len(field_indices),
        "wavelength_count": len(system.wavelengths),
        "pupil_sample_count": int(pupil_sample_count),
        "image_sample_count": int(image_sample_count),
        "sample_count": int(pupil_sample_count) if int(pupil_sample_count) == int(image_sample_count) else None,
        "case_name": "sample_count",
        "case_value": int(pupil_sample_count),
        "total_pupil_sample_count": total_pupil_sample_count,
        "total_phase_sample_count": total_phase_sample_count,
        "detected_design_batch_size": detected_design_batch_size,
        "design_batch_size": design_batch_size,
        "minibatch_count": minibatch_count,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_ms": elapsed_seconds * 1000.0,
        "compute_elapsed_seconds": compute_elapsed_seconds,
        "compute_elapsed_ms": compute_elapsed_seconds * 1000.0,
        "export_elapsed_seconds": export_elapsed_seconds,
        "export_elapsed_ms": export_elapsed_seconds * 1000.0,
        "psf_images_per_second": psf_per_second,
        "compute_psf_images_per_second": compute_psf_per_second,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "computed_image_count": computed_image_count,
        "saved_image_count": saved_image_count,
        "skip_images": bool(skip_images),
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if int(args.design_count) <= 0:
        raise ValueError("随机设计数量必须为正整数。")
    if int(args.pupil_sample_count) <= 0 or int(args.image_sample_count) <= 0:
        raise ValueError("采样率必须为正整数。")

    summary = run_batch_psf_analysis(
        device=device,
        design_count=int(args.design_count),
        random_seed=int(args.random_seed),
        pupil_sample_count=int(args.pupil_sample_count),
        image_sample_count=int(args.image_sample_count),
        output_dir=Path(args.output_dir),
        skip_images=bool(args.skip_images),
    )
    if args.summary_json is not None:
        save_summary_json(Path(args.summary_json), summary)


if __name__ == "__main__":
    main()
