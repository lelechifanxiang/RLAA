from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_DIR = REPO_ROOT / "examples/output/server_batch_spot_scaling"
DEFAULT_DESKTOP_DIR = REPO_ROOT / "examples/output/desktop_batch_spot_scaling"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "examples/output/batch_spot_scaling_compare.png"

SERIES_ORDER = (
    "Server CPU (8474C)",
    "Server GPU (4090D)",
    "Desktop CPU (i7-12700F)",
    "Desktop GPU (V100 16G)",
    "Desktop Zemax",
)

SERIES_STYLE = {
    "Server CPU (8474C)": {"marker": "o", "linewidth": 2.0},
    "Server GPU (4090D)": {"marker": "s", "linewidth": 2.0},
    "Desktop CPU (i7-12700F)": {"marker": "^", "linewidth": 2.0},
    "Desktop GPU (V100 16G)": {"marker": "D", "linewidth": 2.0},
    "Desktop Zemax": {"marker": "*", "linewidth": 0.0, "markersize": 13},
}

LABEL_OFFSETS = {
    "Server CPU (8474C)": (-8, 10),
    "Server GPU (4090D)": (-8, -14),
    "Desktop CPU (i7-12700F)": (8, -14),
    "Desktop GPU (V100 16G)": (8, 10),
    "Desktop Zemax": (0, 12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制服务器、本机和 Zemax 点列图性能对比曲线")
    parser.add_argument("--server-dir", default=str(DEFAULT_SERVER_DIR), help="服务器 batch_spot_scaling 输出目录")
    parser.add_argument("--desktop-dir", default=str(DEFAULT_DESKTOP_DIR), help="本机 batch_spot_scaling 输出目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="输出图片路径")
    parser.add_argument("--csv-output", default=None, help="可选：导出绘图数据 CSV 路径")
    parser.add_argument("--json-output", default=None, help="可选：导出绘图数据 JSON 路径")
    parser.add_argument(
        "--x-axis",
        choices=("total-rays", "pupil-rays"),
        default="total-rays",
        help="横坐标类型，默认使用总追迹光线数",
    )
    parser.add_argument("--linear-y", action="store_true", help="使用线性纵轴，默认使用对数纵轴")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def device_series_label(platform: str, device: str) -> str:
    if platform == "server":
        if device.startswith("cpu"):
            return "Server CPU (8474C)"
        return "Server GPU (4090D)"

    if device.startswith("cpu"):
        return "Desktop CPU (i7-12700F)"
    return "Desktop GPU (V100 16G)"


def collect_opticscore_rows(base_dir: Path, platform: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for json_path in sorted((base_dir / "runs").glob("*.json")):
        data = load_json(json_path)
        if int(data["surface_count"]) != 4:
            continue

        seconds = float(data["elapsed_ms"]) / 1000.0
        rows.append(
            {
                "series": device_series_label(platform, str(data["device"])),
                "source": "optics_core",
                "platform": platform,
                "device": data["device"],
                "surface_count": int(data["surface_count"]),
                "ray_density": int(data["ray_density"]),
                "pupil_ray_count": int(data["pupil_ray_count"]),
                "total_ray_count": int(data["total_ray_count"]),
                "design_count": int(data["design_count"]),
                "elapsed_s": seconds,
                "elapsed_label": f"{seconds:.1f}s",
                "path": str(json_path),
            }
        )
    return rows


def collect_zemax_row(desktop_dir: Path) -> dict[str, Any]:
    data = load_json(desktop_dir / "zemax_batch_spot_s1-2_d13.json")
    seconds = float(data["elapsed_ms"]) / 1000.0
    return {
        "series": "Desktop Zemax",
        "source": "zemax",
        "platform": "desktop",
        "device": "zemax",
        "surface_count": int(data["surface_count"]),
        "ray_density": int(data["ray_density"]),
        "pupil_ray_count": int(data["pupil_ray_count"]),
        "total_ray_count": int(data["total_ray_count"]),
        "design_count": int(data["design_count"]),
        "elapsed_s": seconds,
        "elapsed_label": f"{seconds:.1f}s",
        "path": str(desktop_dir / "zemax_batch_spot_s1-2_d13.json"),
    }


def collect_rows(server_dir: Path, desktop_dir: Path) -> list[dict[str, Any]]:
    rows = collect_opticscore_rows(server_dir, "server")
    rows.extend(collect_opticscore_rows(desktop_dir, "desktop"))
    rows.append(collect_zemax_row(desktop_dir))

    present_series = {row["series"] for row in rows}
    missing_series = [series for series in SERIES_ORDER if series not in present_series]
    if missing_series:
        raise ValueError(f"缺少绘图数据: {missing_series}")
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series",
        "source",
        "platform",
        "device",
        "surface_count",
        "ray_density",
        "pupil_ray_count",
        "total_ray_count",
        "design_count",
        "elapsed_s",
        "path",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (SERIES_ORDER.index(item["series"]), item["pupil_ray_count"])):
            writer.writerow({name: row[name] for name in fieldnames})


def write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def format_ray_count(value: float, pos: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def plot_rows(rows: list[dict[str, Any]], output_path: Path, *, x_axis: str, linear_y: bool) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["series"]].append(row)

    x_key = "total_ray_count" if x_axis == "total-rays" else "pupil_ray_count"
    x_label = "Total sampled ray count" if x_axis == "total-rays" else "Pupil sampled ray count"

    plt.rcParams.update(
        {
            "figure.figsize": (11.5, 7.0),
            "axes.grid": True,
            "grid.alpha": 0.28,
            "font.size": 10,
        }
    )
    fig, ax = plt.subplots()

    for series in SERIES_ORDER:
        series_rows = sorted(grouped[series], key=lambda item: item[x_key])
        x_values = [row[x_key] for row in series_rows]
        y_values = [row["elapsed_s"] for row in series_rows]
        style = SERIES_STYLE[series]
        ax.plot(x_values, y_values, label=series, **style)

        x_offset, y_offset = LABEL_OFFSETS[series]
        horizontal_align = "left" if x_offset > 0 else "right" if x_offset < 0 else "center"
        vertical_align = "bottom" if y_offset > 0 else "top"
        for row in series_rows:
            ax.annotate(
                row["elapsed_label"],
                xy=(row[x_key], row["elapsed_s"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal_align,
                va=vertical_align,
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.72},
            )

    if not linear_y:
        ax.set_yscale("log")

    ax.set_title("Batch spot runtime scaling")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Elapsed time (s)")
    ax.xaxis.set_major_formatter(FuncFormatter(format_ray_count))
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    server_dir = Path(args.server_dir).resolve()
    desktop_dir = Path(args.desktop_dir).resolve()
    output_path = Path(args.output).resolve()

    rows = collect_rows(server_dir, desktop_dir)
    plot_rows(rows, output_path, x_axis=args.x_axis, linear_y=args.linear_y)

    csv_output = Path(args.csv_output).resolve() if args.csv_output else output_path.with_suffix(".csv")
    json_output = Path(args.json_output).resolve() if args.json_output else output_path.with_suffix(".json")
    write_csv(rows, csv_output)
    write_json(rows, json_output)

    print(f"绘图完成: {output_path}")
    print(f"绘图数据 CSV: {csv_output}")
    print(f"绘图数据 JSON: {json_output}")


if __name__ == "__main__":
    main()
