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
from examples.batch_spot import resolve_device, save_summary_json
from scripts.batch_tolerance_common import (
    DEFAULT_MONTE_CARLO_DESIGN_COUNT,
    DEFAULT_RANDOM_SEED,
    build_random_assembly_tolerance_multi_system,
    monte_carlo_tolerance_summary_fields,
)


DEFAULT_DEVICE = "cuda:0"
DEFAULT_SAMPLE_COUNT = 32
DEFAULT_OUTPUT_DIR = REPO_ROOT / "examples/output/batch_wavefront"
PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]

# 使用自定义系统时修改这两项；面号采用 Zemax LDE 编号。
ZMX_PATH = REPO_ROOT / "tests/zemax/zmx_files/case2_3p_center.zmx"
# 单组必须保留尾逗号，例如 ((1, 4),)；多组写成 ((1, 4), (7, 10))。
COORDINATE_BREAK_PAIRS: tuple[tuple[int, int], ...] = ((1, 4),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="装配公差 Monte Carlo Wavefront Map 图片导出")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="运行设备，默认 cuda:0；CPU 可手动指定 cpu")
    parser.add_argument(
        "--design-count",
        type=int,
        default=DEFAULT_MONTE_CARLO_DESIGN_COUNT,
        help=f"随机装配公差设计数量，默认 {DEFAULT_MONTE_CARLO_DESIGN_COUNT}",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help=f"随机种子，默认 {DEFAULT_RANDOM_SEED}")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Wavefront Map 采样率，默认 {DEFAULT_SAMPLE_COUNT}",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"图片输出目录，默认 {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--summary-json", default=None, help="导出本次性能摘要")
    parser.add_argument("--skip-images", action="store_true", help="只执行 Wavefront Map 计算，不导出 PNG 图片")
    parser.add_argument("--field-indices", nargs="+", type=int, default=None, help="视场索引列表，默认全部视场")
    parser.add_argument("--wavelength-indices", nargs="+", type=int, default=None, help="波长索引列表，默认主波长")
    return parser.parse_args()


def synchronized_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def normalize_wavefront_stack_to_uint8(opd_stack: torch.Tensor):
    images = torch.as_tensor(opd_stack, dtype=torch.float64)
    max_abs = torch.amax(torch.abs(images), dim=(-2, -1), keepdim=True)
    normalized = torch.where(
        max_abs > torch.finfo(torch.float64).eps,
        0.5 + 0.5 * images / max_abs.clamp_min(torch.finfo(torch.float64).eps),
        torch.full_like(images, 0.5),
    )
    return torch.round(normalized * 255.0).clamp(0.0, 255.0).to(torch.uint8).cpu().contiguous().numpy()


def export_wavefront_images(
    result: oc.WavefrontResult,
    *,
    output_dir: Path,
) -> int:
    opd = torch.as_tensor(result.opd, dtype=torch.float64)
    images_uint8 = normalize_wavefront_stack_to_uint8(opd)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for design_index in range(opd.shape[0]):
        for local_field_index, field_index in enumerate(result.field_indices):
            for local_wavelength_index, wavelength_index in enumerate(result.wavelength_indices):
                output_path = output_dir / (
                    f"wavefront_design_{design_index:04d}_field_{field_index}_wave_{wavelength_index}.png"
                )
                image_uint8 = images_uint8[design_index, local_field_index, local_wavelength_index]
                if not cv2.imwrite(str(output_path), image_uint8, PNG_WRITE_PARAMS):
                    raise RuntimeError(f"failed to write wavefront png: {output_path}")
                saved_count += 1
    return saved_count


def run_batch_wavefront_analysis(
    *,
    device: torch.device,
    design_count: int,
    random_seed: int,
    sample_count: int,
    output_dir: Path,
    field_indices: tuple[int, ...] | None,
    wavelength_indices: tuple[int, ...] | None,
    skip_images: bool = False,
) -> dict[str, object]:
    system, _design_records, field_points, surface_indices = build_random_assembly_tolerance_multi_system(
        ZMX_PATH,
        device=device,
        design_count=int(design_count),
        random_seed=int(random_seed),
        coordinate_break_pairs=COORDINATE_BREAK_PAIRS,
    )
    system.prepare()
    resolved_field_indices = tuple(range(len(system.fields))) if field_indices is None else tuple(field_indices)
    resolved_wavelength_indices = (
        (int(system.wavelengths.primary_index),)
        if wavelength_indices is None
        else tuple(int(index) for index in wavelength_indices)
    )
    settings = oc.WavefrontSettings(
        field_indices=resolved_field_indices,
        wavelength_indices=resolved_wavelength_indices,
        sample_count=int(sample_count),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_started_at = synchronized_now(device)
    compute_started_at = synchronized_now(device)
    result = system.analysis.wavefront(settings).run()
    compute_elapsed_seconds = synchronized_now(device) - compute_started_at

    saved_image_count = 0
    export_elapsed_seconds = 0.0
    if not skip_images:
        export_started_at = time.perf_counter()
        saved_image_count = export_wavefront_images(result, output_dir=output_dir)
        export_elapsed_seconds = time.perf_counter() - export_started_at
    elapsed_seconds = synchronized_now(device) - total_started_at

    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    computed_map_count = system.system_count * len(resolved_field_indices) * len(resolved_wavelength_indices)
    maps_per_second = computed_map_count / elapsed_seconds
    compute_maps_per_second = computed_map_count / compute_elapsed_seconds
    wavefront_shape = tuple(torch.as_tensor(result.opd).shape)

    print("并行 Wavefront Map 信息:")
    print(
        f"design_count={system.system_count}, field_count={len(resolved_field_indices)}, "
        f"wavelength_count={len(resolved_wavelength_indices)}"
    )
    print(f"sample_count={sample_count}")
    print(
        f"detected_design_batch_size={result.detected_design_batch_size}, "
        f"design_batch_size={result.design_batch_size}, minibatch_count={result.minibatch_count}"
    )
    print(f"elapsed_seconds={elapsed_seconds:.6f}, wavefront_maps_per_second={maps_per_second:.3f}")
    print(
        f"compute_elapsed_seconds={compute_elapsed_seconds:.6f}, "
        f"compute_wavefront_maps_per_second={compute_maps_per_second:.3f}"
    )
    print(f"export_elapsed_seconds={export_elapsed_seconds:.6f}, skip_images={skip_images}")
    print(f"peak_allocated_gib={peak_allocated_bytes / 2**30:.3f}")
    print(f"peak_reserved_gib={peak_reserved_bytes / 2**30:.3f}")
    print(f"computed_map_count={computed_map_count}")
    print(f"saved_image_count={saved_image_count}")
    print(f"wavefront_shape={wavefront_shape}")
    print(f"output_dir={output_dir}")

    return {
        "analysis_type": "wavefront",
        "device": str(device),
        **monte_carlo_tolerance_summary_fields(
            int(design_count),
            int(random_seed),
            zmx_path=ZMX_PATH,
            coordinate_break_pairs=COORDINATE_BREAK_PAIRS,
        ),
        "internal_surface_indices": list(surface_indices),
        "field_points": [list(point) for point in field_points],
        "design_count": system.system_count,
        "field_count": len(resolved_field_indices),
        "wavelength_count": len(resolved_wavelength_indices),
        "field_indices": list(resolved_field_indices),
        "wavelength_indices": list(resolved_wavelength_indices),
        "sample_count": int(sample_count),
        "case_name": "sample_count",
        "case_value": int(sample_count),
        "detected_design_batch_size": result.detected_design_batch_size,
        "design_batch_size": result.design_batch_size,
        "minibatch_count": result.minibatch_count,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_ms": elapsed_seconds * 1000.0,
        "compute_elapsed_seconds": compute_elapsed_seconds,
        "compute_elapsed_ms": compute_elapsed_seconds * 1000.0,
        "export_elapsed_seconds": export_elapsed_seconds,
        "export_elapsed_ms": export_elapsed_seconds * 1000.0,
        "wavefront_maps_per_second": maps_per_second,
        "compute_wavefront_maps_per_second": compute_maps_per_second,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "computed_map_count": computed_map_count,
        "saved_image_count": saved_image_count,
        "skip_images": bool(skip_images),
        "wavefront_shape": list(wavefront_shape),
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if int(args.design_count) <= 0:
        raise ValueError("随机设计数量必须为正整数。")
    if int(args.sample_count) <= 1:
        raise ValueError("Wavefront Map 采样率必须大于 1。")

    summary = run_batch_wavefront_analysis(
        device=device,
        design_count=int(args.design_count),
        random_seed=int(args.random_seed),
        sample_count=int(args.sample_count),
        output_dir=Path(args.output_dir),
        field_indices=None if args.field_indices is None else tuple(args.field_indices),
        wavelength_indices=None if args.wavelength_indices is None else tuple(args.wavelength_indices),
        skip_images=bool(args.skip_images),
    )
    if args.summary_json is not None:
        save_summary_json(Path(args.summary_json), summary)


if __name__ == "__main__":
    main()
