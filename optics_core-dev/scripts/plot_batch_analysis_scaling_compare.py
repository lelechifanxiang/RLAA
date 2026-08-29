from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.batch_analysis_scaling import ANALYSIS_SPECS, workload_label

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制服务器、本机和 Zemax 的批量测评性能对比曲线")
    parser.add_argument(
        "--analysis",
        choices=tuple(ANALYSIS_SPECS),
        required=True,
        help="测评项目类型，例如 spot 或 mtf",
    )
    parser.add_argument("--server-dir", default=None, help="服务器 batch scaling 输出目录")
    parser.add_argument("--desktop-dir", default=None, help="本机 batch scaling 输出目录")
    parser.add_argument("--zemax-summary", default=None, help="可选：Zemax summary json 路径")
    parser.add_argument("--zemax-label", default="Desktop Zemax", help="Zemax 曲线图例名称")
    parser.add_argument("--output", default=None, help="输出图片路径")
    parser.add_argument("--csv-output", default=None, help="可选：导出绘图数据 CSV 路径")
    parser.add_argument("--json-output", default=None, help="可选：导出绘图数据 JSON 路径")
    parser.add_argument("--surface-count", type=int, default=None, help="可选：仅保留指定 surface_count")
    parser.add_argument("--x-key", default=None, help="横坐标字段；未提供时按 analysis 取默认值")
    parser.add_argument("--linear-y", action="store_true", help="使用线性纵轴，默认使用对数纵轴")
    return parser.parse_args(argv)


def default_server_dir(analysis: str) -> Path:
    return REPO_ROOT / f"examples/output/server_batch_{analysis}_scaling"


def default_desktop_dir(analysis: str) -> Path:
    return REPO_ROOT / f"examples/output/desktop_batch_{analysis}_scaling"


def default_output_path(analysis: str) -> Path:
    return REPO_ROOT / f"examples/output/batch_{analysis}_scaling_compare.png"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def device_series_label(platform: str, device: str) -> str:
    if platform == "server":
        if str(device).startswith("cpu"):
            return "Server CPU (8474C)"
        return "Server GPU (4090D)"

    if str(device).startswith("cpu"):
        return "Desktop CPU (i7-12700F)"
    return "Desktop GPU (V100 16G)"


def collect_opticscore_rows(
    base_dir: Path,
    *,
    platform: str,
    analysis: str,
    surface_count: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for json_path in sorted((base_dir / "runs").glob("*.json")):
        data = load_json(json_path)
        if str(data.get("analysis_type")) != analysis:
            continue
        if surface_count is not None and int(data["surface_count"]) != int(surface_count):
            continue

        seconds = float(data["elapsed_ms"]) / 1000.0
        rows.append(
            {
                "series": device_series_label(platform, str(data["device"])),
                "source": "optics_core",
                "analysis_type": analysis,
                "platform": platform,
                "device": data["device"],
                "surface_count": int(data["surface_count"]),
                "case_name": data.get("case_name"),
                "case_value": int(data.get("case_value", 0)),
                "pupil_ray_count": int(data.get("pupil_ray_count", 0)),
                "total_ray_count": int(data.get("total_ray_count", 0)),
                "total_pupil_sample_count": int(data.get("total_pupil_sample_count", 0)),
                "total_phase_sample_count": int(data.get("total_phase_sample_count", 0)),
                "design_count": int(data["design_count"]),
                "elapsed_s": seconds,
                "elapsed_label": f"{seconds:.1f}s",
                "path": str(json_path),
            }
        )
    return rows


def collect_zemax_rows(
    summary_path: Path,
    *,
    analysis: str,
    surface_count: int | None,
    series_label: str,
) -> list[dict[str, Any]]:
    payload = load_json(summary_path)
    source_rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else [payload]
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        if str(source_row.get("analysis_type", analysis)) != analysis:
            continue
        if surface_count is not None and int(source_row["surface_count"]) != int(surface_count):
            continue
        seconds = float(source_row["elapsed_ms"]) / 1000.0
        rows.append(
            {
                "series": series_label,
                "source": "zemax",
                "analysis_type": analysis,
                "platform": "desktop",
                "device": "zemax",
                "surface_count": int(source_row["surface_count"]),
                "case_name": source_row.get("case_name"),
                "case_value": int(source_row.get("case_value", 0)),
                "pupil_ray_count": int(source_row.get("pupil_ray_count", 0)),
                "total_ray_count": int(source_row.get("total_ray_count", 0)),
                "total_pupil_sample_count": int(source_row.get("total_pupil_sample_count", 0)),
                "total_phase_sample_count": int(source_row.get("total_phase_sample_count", 0)),
                "design_count": int(source_row["design_count"]),
                "elapsed_s": seconds,
                "elapsed_label": f"{seconds:.1f}s",
                "path": str(summary_path),
            }
        )
    return rows


def collect_rows(
    *,
    server_dir: Path,
    desktop_dir: Path,
    analysis: str,
    surface_count: int | None,
    zemax_summary: Path | None,
    zemax_label: str,
) -> list[dict[str, Any]]:
    rows = collect_opticscore_rows(server_dir, platform="server", analysis=analysis, surface_count=surface_count)
    rows.extend(
        collect_opticscore_rows(desktop_dir, platform="desktop", analysis=analysis, surface_count=surface_count)
    )
    if zemax_summary is not None:
        rows.extend(
            collect_zemax_rows(
                zemax_summary,
                analysis=analysis,
                surface_count=surface_count,
                series_label=zemax_label,
            )
        )

    if not rows:
        raise ValueError("未找到可用的绘图数据。")
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series",
        "source",
        "analysis_type",
        "platform",
        "device",
        "surface_count",
        "case_name",
        "case_value",
        "pupil_ray_count",
        "total_ray_count",
        "total_pupil_sample_count",
        "total_phase_sample_count",
        "design_count",
        "elapsed_s",
        "path",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["series"], item["case_value"], item["design_count"])):
            writer.writerow({name: row[name] for name in fieldnames})


def write_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def format_count(value: float, pos: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def plot_rows(rows: list[dict[str, Any]], output_path: Path, *, analysis: str, x_key: str, linear_y: bool) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["series"]].append(row)

    present_series = {row["series"] for row in rows}
    ordered_series = [series for series in SERIES_ORDER if series in present_series]
    unordered_series = sorted(series for series in present_series if series not in SERIES_ORDER)
    series_order = ordered_series + unordered_series

    plt.rcParams.update(
        {
            "figure.figsize": (11.5, 7.0),
            "axes.grid": True,
            "grid.alpha": 0.28,
            "font.size": 10,
        }
    )
    fig, ax = plt.subplots()

    for series in series_order:
        series_rows = [row for row in grouped[series] if float(row.get(x_key, 0.0)) > 0.0]
        series_rows = sorted(series_rows, key=lambda item: float(item[x_key]))
        if not series_rows:
            continue
        x_values = [float(row[x_key]) for row in series_rows]
        y_values = [float(row["elapsed_s"]) for row in series_rows]
        style = SERIES_STYLE.get(series, {"marker": "o", "linewidth": 2.0})
        ax.plot(x_values, y_values, label=series, **style)

        x_offset, y_offset = LABEL_OFFSETS.get(series, (8, 10))
        horizontal_align = "left" if x_offset > 0 else "right" if x_offset < 0 else "center"
        vertical_align = "bottom" if y_offset > 0 else "top"
        for row in series_rows:
            ax.annotate(
                row["elapsed_label"],
                xy=(float(row[x_key]), float(row["elapsed_s"])),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=horizontal_align,
                va=vertical_align,
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.72},
            )

    if not linear_y:
        ax.set_yscale("log")

    ax.set_title(f"Batch {analysis} runtime scaling")
    ax.set_xlabel(workload_label(x_key))
    ax.set_ylabel("Elapsed time (s)")
    ax.xaxis.set_major_formatter(FuncFormatter(format_count))
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None, *, default_analysis: str | None = None) -> None:
    effective_argv = list(argv or [])
    if not effective_argv:
        import sys

        effective_argv = list(sys.argv[1:])
    if default_analysis is not None and "--analysis" not in effective_argv:
        effective_argv = ["--analysis", default_analysis, *effective_argv]

    args = parse_args(effective_argv)
    spec = ANALYSIS_SPECS[args.analysis]
    server_dir = Path(args.server_dir) if args.server_dir is not None else default_server_dir(args.analysis)
    desktop_dir = Path(args.desktop_dir) if args.desktop_dir is not None else default_desktop_dir(args.analysis)
    zemax_summary = Path(args.zemax_summary) if args.zemax_summary is not None else None
    output_path = Path(args.output) if args.output is not None else default_output_path(args.analysis)
    x_key = spec.default_x_key if args.x_key is None else str(args.x_key)

    rows = collect_rows(
        server_dir=server_dir.resolve(),
        desktop_dir=desktop_dir.resolve(),
        analysis=args.analysis,
        surface_count=args.surface_count,
        zemax_summary=zemax_summary.resolve() if zemax_summary is not None else None,
        zemax_label=str(args.zemax_label),
    )
    plot_rows(rows, output_path.resolve(), analysis=args.analysis, x_key=x_key, linear_y=args.linear_y)

    csv_output = Path(args.csv_output).resolve() if args.csv_output else output_path.with_suffix(".csv")
    json_output = Path(args.json_output).resolve() if args.json_output else output_path.with_suffix(".json")
    write_csv(rows, csv_output)
    write_json(rows, json_output)

    print(f"绘图完成: {output_path}")
    print(f"绘图数据 CSV: {csv_output}")
    print(f"绘图数据 JSON: {json_output}")


if __name__ == "__main__":
    main()
