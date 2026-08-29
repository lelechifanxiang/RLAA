from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


@dataclass(frozen=True)
class AnalysisScalingSpec:
    name: str
    label: str
    script_path: Path
    default_output_dir: Path
    default_case_values: tuple[int, ...]
    case_name: str
    case_label: str
    default_x_key: str
    build_command: Callable[[str, list[int], int, Path], list[str]]


def default_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    return devices


def _build_spot_command(device: str, surface_numbers: list[int], case_value: int, summary_path: Path) -> list[str]:
    return [
        sys.executable,
        str(EXAMPLES_DIR / "batch_spot.py"),
        "--device",
        device,
        "--surfaces",
        *[str(number) for number in surface_numbers],
        "--ray-density",
        str(case_value),
        "--skip-csv",
        "--summary-json",
        str(summary_path),
    ]


def _build_mtf_command(device: str, surface_numbers: list[int], case_value: int, summary_path: Path) -> list[str]:
    return [
        sys.executable,
        str(EXAMPLES_DIR / "batch_mtf.py"),
        "--device",
        device,
        "--pupil-sample-count",
        str(case_value),
        "--image-sample-count",
        str(case_value),
        "--skip-csv",
        "--summary-json",
        str(summary_path),
    ]


ANALYSIS_SPECS: dict[str, AnalysisScalingSpec] = {
    "spot": AnalysisScalingSpec(
        name="spot",
        label="spot",
        script_path=EXAMPLES_DIR / "batch_spot.py",
        default_output_dir=REPO_ROOT / "examples/output/batch_spot_scaling",
        default_case_values=(3, 5, 7, 9, 11, 13),
        case_name="ray_density",
        case_label="ray_density",
        default_x_key="total_ray_count",
        build_command=_build_spot_command,
    ),
    "mtf": AnalysisScalingSpec(
        name="mtf",
        label="mtf",
        script_path=EXAMPLES_DIR / "batch_mtf.py",
        default_output_dir=REPO_ROOT / "examples/output/batch_mtf_scaling",
        default_case_values=(8, 12, 16, 24, 32),
        case_name="sample_count",
        case_label="sample_count",
        default_x_key="total_phase_sample_count",
        build_command=_build_mtf_command,
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量扫描 spot/mtf 配置并绘制性能曲线")
    parser.add_argument(
        "--analysis",
        choices=tuple(ANALYSIS_SPECS),
        required=True,
        help="测评项目类型，例如 spot 或 mtf",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=default_devices(),
        help="设备列表，例如 cpu cuda:0，默认自动选择 cpu 和可用 GPU",
    )
    parser.add_argument(
        "--surface-groups",
        nargs="+",
        default=["1", "1,2", "1,2,10,11"],
        help="多个扰动面组合，每组使用逗号分隔，例如 1 1,2 1,2,10,11",
    )
    parser.add_argument(
        "--case-values",
        nargs="+",
        type=int,
        default=None,
        help="扫描参数列表。spot 对应 ray_density，mtf 对应 pupil/image sample count",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；未提供时按 analysis 使用默认目录",
    )
    parser.add_argument(
        "--x-key",
        default=None,
        help="绘图横坐标字段；未提供时按 analysis 使用默认 workload 字段",
    )
    return parser.parse_args(argv)


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


def run_slug(spec: AnalysisScalingSpec, device: str, surface_numbers: list[int], case_value: int) -> str:
    device_slug = device.replace(":", "_")
    surface_slug = "-".join(str(number) for number in surface_numbers)
    return f"{spec.name}_{device_slug}_s{surface_slug}_{spec.case_name}{case_value}"


def run_batch_analysis_case(
    *,
    spec: AnalysisScalingSpec,
    device: str,
    surface_numbers: list[int],
    case_value: int,
    output_dir: Path,
) -> dict[str, object]:
    run_name = run_slug(spec, device, surface_numbers, case_value)
    run_dir = output_dir / "runs"
    summary_path = run_dir / f"{run_name}.json"
    log_path = run_dir / f"{run_name}.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    command = spec.build_command(device, surface_numbers, int(case_value), summary_path)
    print(f"运行: analysis={spec.name}, device={device}, surfaces={surface_numbers}, {spec.case_label}={case_value}")
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
        raise RuntimeError(f"{spec.script_path.name} 运行失败，日志见 {log_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["analysis_type"] = summary.get("analysis_type", spec.name)
    summary["surface_group"] = list(surface_numbers)
    summary["surface_group_label"] = surface_group_label(surface_numbers)
    summary["run_name"] = run_name
    summary["summary_json_path"] = str(summary_path)
    summary["log_path"] = str(log_path)
    summary["wall_ms"] = wall_ms
    summary["case_name"] = spec.case_name
    summary["case_value"] = int(case_value)
    print(
        f"完成: x={summary.get(spec.default_x_key)}, elapsed_ms={float(summary['elapsed_ms']):.3f}"
    )
    return summary


def _stringify_value(value: object) -> str | int | float:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def save_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    common_fieldnames = [
        "analysis_type",
        "run_name",
        "device",
        "surface_group_label",
        "surface_count",
        "surface_group",
        "case_name",
        "case_value",
        "design_count",
        "field_count",
        "wavelength_count",
        "elapsed_ms",
        "wall_ms",
        "summary_json_path",
        "log_path",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in common_fieldnames
        }
    )
    fieldnames = common_fieldnames + extra_fieldnames

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: _stringify_value(row.get(fieldname)) for fieldname in fieldnames})


def save_summary_json(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    payload = {
        "analysis": args.analysis,
        "devices": list(args.devices),
        "surface_groups": list(args.surface_groups),
        "case_values": list(args.case_values),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def workload_label(x_key: str) -> str:
    labels = {
        "total_ray_count": "Total sampled ray count",
        "pupil_ray_count": "Pupil sampled ray count",
        "total_pupil_sample_count": "Total pupil sample count",
        "total_phase_sample_count": "Total phase sample count",
        "design_count": "Design count",
    }
    return labels.get(x_key, x_key)


def plot_scaling_curve(path: Path, rows: list[dict[str, object]], *, spec: AnalysisScalingSpec, x_key: str) -> None:
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        label = f"{row['device']} | {row['surface_group_label']}"
        grouped_rows[label].append(row)

    figure, axes = plt.subplots(figsize=(10.5, 6.5), dpi=150)
    for label, series_rows in grouped_rows.items():
        ordered_rows = sorted(
            series_rows,
            key=lambda row: (float(row[x_key]), int(row["case_value"])),
        )
        x_values = [float(row[x_key]) for row in ordered_rows]
        y_values = [float(row["elapsed_ms"]) for row in ordered_rows]
        axes.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            label=label,
        )

    axes.set_title(f"Batch {spec.label} workload vs time")
    axes.set_xlabel(workload_label(x_key))
    axes.set_ylabel(f"{spec.label.upper()} analysis time (ms)")
    axes.grid(True, linewidth=0.35, alpha=0.45)
    axes.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None, *, default_analysis: str | None = None) -> None:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if default_analysis is not None and "--analysis" not in effective_argv:
        effective_argv = ["--analysis", default_analysis, *effective_argv]
    args = parse_args(effective_argv)
    spec = ANALYSIS_SPECS[args.analysis]
    case_values = list(spec.default_case_values if args.case_values is None else args.case_values)
    output_dir = Path(args.output_dir) if args.output_dir is not None else spec.default_output_dir
    x_key = spec.default_x_key if args.x_key is None else str(args.x_key)

    rows: list[dict[str, object]] = []
    surface_groups = [parse_surface_group(group_text) for group_text in args.surface_groups]
    for device in args.devices:
        for surface_numbers in surface_groups:
            for case_value in case_values:
                if int(case_value) <= 0:
                    raise ValueError(f"{spec.case_label} 必须为正整数。")
                rows.append(
                    run_batch_analysis_case(
                        spec=spec,
                        device=device,
                        surface_numbers=surface_numbers,
                        case_value=int(case_value),
                        output_dir=output_dir,
                    )
                )

    csv_path = output_dir / f"batch_{spec.name}_scaling.csv"
    json_path = output_dir / f"batch_{spec.name}_scaling.json"
    plot_path = output_dir / f"batch_{spec.name}_scaling.png"

    save_summary_csv(csv_path, rows)
    save_summary_json(json_path, rows, args)
    plot_scaling_curve(plot_path, rows, spec=spec, x_key=x_key)

    print("批量扫描完成。")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"图片: {plot_path}")


if __name__ == "__main__":
    main()
