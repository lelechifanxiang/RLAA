from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SPOT_SCRIPT_PATH = REPO_ROOT / "examples/batch_spot.py"
OUTPUT_DIR = REPO_ROOT / "examples/output/batch_spot_scaling"
DEFAULT_SURFACE_GROUPS = ("1", "1,2", "1,2,10,11")
DEFAULT_RAY_DENSITIES = (3, 5, 7, 9, 11, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量扫描 batch_spot 配置并绘制光线数量-耗时曲线")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=default_devices(),
        help="设备列表，例如 cpu cuda:0，默认自动选择 cpu 和可用 GPU",
    )
    parser.add_argument(
        "--surface-groups",
        nargs="+",
        default=list(DEFAULT_SURFACE_GROUPS),
        help="多个扰动面组合，每组使用逗号分隔，例如 1 1,2 1,2,10,11",
    )
    parser.add_argument(
        "--ray-densities",
        nargs="+",
        type=int,
        default=list(DEFAULT_RAY_DENSITIES),
        help="需要扫描的 spot 六边采样密度列表",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"输出目录，默认 {OUTPUT_DIR}",
    )
    return parser.parse_args()


def default_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    return devices


def parse_surface_group(group_text: str) -> list[int]:
    numbers: list[int] = []
    for part in group_text.split(","):
        stripped = part.strip()
        if stripped:
            numbers.append(int(stripped))
    if not numbers:
        raise ValueError(f"无效的 surface group: {group_text!r}")
    return numbers


def surface_group_label(surface_numbers: list[int]) -> str:
    return f"{len(surface_numbers)} surfaces [{', '.join(str(number) for number in surface_numbers)}]"


def run_slug(device: str, surface_numbers: list[int], ray_density: int) -> str:
    device_slug = device.replace(":", "_")
    surface_slug = "-".join(str(number) for number in surface_numbers)
    return f"{device_slug}_s{surface_slug}_d{ray_density}"


def run_batch_spot_case(
    *,
    device: str,
    surface_numbers: list[int],
    ray_density: int,
    output_dir: Path,
) -> dict[str, object]:
    run_name = run_slug(device, surface_numbers, ray_density)
    run_dir = output_dir / "runs"
    summary_path = run_dir / f"{run_name}.json"
    log_path = run_dir / f"{run_name}.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(BATCH_SPOT_SCRIPT_PATH),
        "--device",
        device,
        "--surfaces",
        *[str(number) for number in surface_numbers],
        "--ray-density",
        str(ray_density),
        "--skip-csv",
        "--summary-json",
        str(summary_path),
    ]

    print(
        f"运行: device={device}, surfaces={surface_numbers}, ray_density={ray_density}"
    )
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    wall_ms = (time.perf_counter() - started_at) * 1000.0

    log_text = completed.stdout
    if completed.stderr:
        log_text += "\n[stderr]\n" + completed.stderr
    log_path.write_text(log_text, encoding="utf-8")

    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise RuntimeError(f"batch_spot.py 运行失败，日志见 {log_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["surface_group"] = list(surface_numbers)
    summary["surface_group_label"] = surface_group_label(surface_numbers)
    summary["run_name"] = run_name
    summary["summary_json_path"] = str(summary_path)
    summary["log_path"] = str(log_path)
    summary["wall_ms"] = wall_ms
    print(
        f"完成: total_ray_count={summary['total_ray_count']}, elapsed_ms={float(summary['elapsed_ms']):.3f}"
    )
    return summary


def save_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_name",
        "device",
        "surface_group_label",
        "surface_count",
        "surface_group",
        "ray_density",
        "design_count",
        "field_count",
        "wavelength_count",
        "pupil_ray_count",
        "total_ray_count",
        "valid_ray_count",
        "elapsed_ms",
        "wall_ms",
        "summary_json_path",
        "log_path",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_name": row["run_name"],
                    "device": row["device"],
                    "surface_group_label": row["surface_group_label"],
                    "surface_count": row["surface_count"],
                    "surface_group": ",".join(str(number) for number in row["surface_group"]),
                    "ray_density": row["ray_density"],
                    "design_count": row["design_count"],
                    "field_count": row["field_count"],
                    "wavelength_count": row["wavelength_count"],
                    "pupil_ray_count": row["pupil_ray_count"],
                    "total_ray_count": row["total_ray_count"],
                    "valid_ray_count": row["valid_ray_count"],
                    "elapsed_ms": f"{float(row['elapsed_ms']):.3f}",
                    "wall_ms": f"{float(row['wall_ms']):.3f}",
                    "summary_json_path": row["summary_json_path"],
                    "log_path": row["log_path"],
                }
            )


def save_summary_json(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    payload = {
        "devices": list(args.devices),
        "surface_groups": list(args.surface_groups),
        "ray_densities": list(args.ray_densities),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_scaling_curve(path: Path, rows: list[dict[str, object]]) -> None:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        label = f"{row['device']} | {row['surface_group_label']}"
        grouped_rows[label].append(row)

    figure, axes = plt.subplots(figsize=(10.5, 6.5), dpi=150)
    for label, series_rows in grouped_rows.items():
        ordered_rows = sorted(
            series_rows,
            key=lambda row: (int(row["total_ray_count"]), int(row["ray_density"])),
        )
        x_values = [int(row["total_ray_count"]) for row in ordered_rows]
        y_values = [float(row["elapsed_ms"]) for row in ordered_rows]
        axes.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            label=label,
        )

    axes.set_title("Batch Spot Ray Count vs Time")
    axes.set_xlabel("Total rays")
    axes.set_ylabel("Spot analysis time (ms)")
    axes.grid(True, linewidth=0.35, alpha=0.45)
    axes.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_groups = [parse_surface_group(group_text) for group_text in args.surface_groups]
    rows: list[dict[str, object]] = []
    for device in args.devices:
        for surface_numbers in surface_groups:
            for ray_density in args.ray_densities:
                if int(ray_density) < 3:
                    raise ValueError("ray_density 必须大于等于 3。")
                rows.append(
                    run_batch_spot_case(
                        device=device,
                        surface_numbers=surface_numbers,
                        ray_density=int(ray_density),
                        output_dir=output_dir,
                    )
                )

    csv_path = output_dir / "batch_spot_scaling.csv"
    json_path = output_dir / "batch_spot_scaling.json"
    plot_path = output_dir / "batch_spot_scaling.png"

    save_summary_csv(csv_path, rows)
    save_summary_json(json_path, rows, args)
    plot_scaling_curve(plot_path, rows)

    print("批量扫描完成。")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"图片: {plot_path}")


if __name__ == "__main__":
    main()
