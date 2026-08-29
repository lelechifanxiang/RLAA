from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OPTICSCORE_CSV = REPO_ROOT / "examples/output/batch_mtf.csv"
DEFAULT_ZEMAX_CSV = REPO_ROOT / "examples/output/zemax_batch_mtf_s1-2_p32_i32.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "examples/output/batch_mtf_error_analysis.png"
METRIC_PATTERN = re.compile(r"^field_(?P<field>\d+)_freq_(?P<frequency>[^_]+)_(?P<direction>sagittal|tangential)$")


@dataclass(slots=True)
class MTFErrorPoint:
    parameter_key: tuple[float, ...]
    opticscore_design_index: int
    zemax_design_index: int
    column: str
    field_index: int
    frequency_lp_per_mm: float
    direction: str
    opticscore_mtf: float
    zemax_mtf: float

    @property
    def error(self) -> float:
        return self.opticscore_mtf - self.zemax_mtf

    @property
    def abs_error(self) -> float:
        return abs(self.error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制批量 Huygens MTF 与 Zemax 的误差分布")
    parser.add_argument("--opticscore-csv", default=str(DEFAULT_OPTICSCORE_CSV), help="本项目 batch_mtf.csv 路径")
    parser.add_argument("--zemax-csv", default=str(DEFAULT_ZEMAX_CSV), help="Zemax batch_mtf CSV 路径")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="输出图片路径")
    parser.add_argument("--top-count", type=int, default=10, help="控制台打印最大误差条数")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def parse_frequency(token: str) -> float:
    return float(token.replace("p", "."))


def metric_metadata(column: str) -> tuple[int, float, str] | None:
    match = METRIC_PATTERN.match(column)
    if match is None:
        return None
    return (
        int(match.group("field")),
        parse_frequency(match.group("frequency")),
        match.group("direction"),
    )


def parameter_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("s") and (column.endswith("_thickness_mm") or column.endswith("_radius_mm"))
    ]


def metric_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if metric_metadata(column) is not None]


def row_key(row: dict[str, str], columns: list[str]) -> tuple[float, ...]:
    return tuple(round(float(row[column]), 6) for column in columns)


def unique_rows_by_key(rows: list[dict[str, str]], columns: list[str]) -> dict[tuple[float, ...], dict[str, str]]:
    grouped: dict[tuple[float, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row_key(row, columns)].append(row)

    duplicates = [key for key, keyed_rows in grouped.items() if len(keyed_rows) != 1]
    if duplicates:
        raise ValueError(f"存在重复参数组合，示例: {duplicates[:3]}")
    return {key: keyed_rows[0] for key, keyed_rows in grouped.items()}


def collect_error_points(
    opticscore_rows: list[dict[str, str]],
    zemax_rows: list[dict[str, str]],
) -> tuple[list[MTFErrorPoint], list[str], list[str]]:
    if not opticscore_rows or not zemax_rows:
        raise ValueError("CSV 文件不能为空。")

    opticscore_columns = list(opticscore_rows[0])
    zemax_columns = list(zemax_rows[0])
    shared_columns = [column for column in opticscore_columns if column in zemax_columns]
    params = parameter_columns(shared_columns)
    metrics = metric_columns(shared_columns)
    if not params:
        raise ValueError("未找到可用于对齐的扰动参数列。")
    if not metrics:
        raise ValueError("未找到 MTF 指标列。")

    opticscore_by_key = unique_rows_by_key(opticscore_rows, params)
    zemax_by_key = unique_rows_by_key(zemax_rows, params)
    shared_keys = sorted(set(opticscore_by_key) & set(zemax_by_key))
    if not shared_keys:
        raise ValueError("两份 CSV 没有可配对的扰动参数组合。")

    points: list[MTFErrorPoint] = []
    for key in shared_keys:
        opticscore_row = opticscore_by_key[key]
        zemax_row = zemax_by_key[key]
        for column in metrics:
            metadata = metric_metadata(column)
            if metadata is None:
                continue
            field_index, frequency, direction = metadata
            points.append(
                MTFErrorPoint(
                    parameter_key=key,
                    opticscore_design_index=int(opticscore_row["design_index"]),
                    zemax_design_index=int(zemax_row["design_index"]),
                    column=column,
                    field_index=field_index,
                    frequency_lp_per_mm=frequency,
                    direction=direction,
                    opticscore_mtf=float(opticscore_row[column]),
                    zemax_mtf=float(zemax_row[column]),
                )
            )
    return points, params, metrics


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def direction_label(direction: str) -> str:
    return "S" if direction == "sagittal" else "T"


def plot_scatter(axis, points: list[MTFErrorPoint]) -> None:
    frequencies = sorted({point.frequency_lp_per_mm for point in points})
    colors = plt.get_cmap("tab10")
    color_by_frequency = {frequency: colors(index % 10) for index, frequency in enumerate(frequencies)}
    marker_by_direction = {"sagittal": "o", "tangential": "x"}

    for frequency in frequencies:
        for direction in ("sagittal", "tangential"):
            series = [
                point
                for point in points
                if point.frequency_lp_per_mm == frequency and point.direction == direction
            ]
            if not series:
                continue
            axis.scatter(
                [point.zemax_mtf for point in series],
                [point.opticscore_mtf for point in series],
                s=18,
                alpha=0.68,
                marker=marker_by_direction[direction],
                color=color_by_frequency[frequency],
                label=f"{frequency:g} lp/mm {direction_label(direction)}",
            )

    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, linestyle="--", label="y=x")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Zemax MTF")
    axis.set_ylabel("OpticsCore MTF")
    axis.set_title("MTF Scatter")
    axis.grid(True, linewidth=0.35, alpha=0.35)
    axis.legend(fontsize=7, loc="lower right")


def plot_error_histogram(axis, points: list[MTFErrorPoint]) -> None:
    errors = [point.error for point in points]
    axis.hist(errors, bins=36, color="#4E79A7", alpha=0.82)
    axis.axvline(0.0, color="black", linewidth=1.0)
    axis.axvline(mean(errors), color="#E15759", linewidth=1.2, linestyle="--", label=f"mean={mean(errors):.4f}")
    axis.set_xlabel("OpticsCore - Zemax")
    axis.set_ylabel("Count")
    axis.set_title(f"Error Histogram, mean |e|={mean([abs(error) for error in errors]):.4f}, RMS={rms(errors):.4f}")
    axis.grid(True, linewidth=0.35, alpha=0.35)
    axis.legend(fontsize=8)


def plot_group_boxplot(axis, points: list[MTFErrorPoint]) -> None:
    group_keys = sorted(
        {(point.field_index, point.frequency_lp_per_mm, point.direction) for point in points},
        key=lambda item: (item[0], item[1], item[2] != "sagittal"),
    )
    grouped_errors = [
        [
            point.error
            for point in points
            if (point.field_index, point.frequency_lp_per_mm, point.direction) == key
        ]
        for key in group_keys
    ]
    labels = [f"F{field}\n{frequency:g}{direction_label(direction)}" for field, frequency, direction in group_keys]

    axis.boxplot(grouped_errors, tick_labels=labels, showfliers=False, patch_artist=True)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_ylabel("OpticsCore - Zemax")
    axis.set_title("Grouped Error Distribution")
    axis.grid(True, axis="y", linewidth=0.35, alpha=0.35)
    axis.tick_params(axis="x", labelsize=7)


def plot_mean_abs_heatmap(axis, figure, points: list[MTFErrorPoint]) -> None:
    frequencies = sorted({point.frequency_lp_per_mm for point in points})
    row_keys = sorted(
        {(point.field_index, point.direction) for point in points},
        key=lambda item: (item[0], item[1] != "sagittal"),
    )
    matrix: list[list[float]] = []
    for field_index, direction in row_keys:
        row: list[float] = []
        for frequency in frequencies:
            values = [
                point.abs_error
                for point in points
                if point.field_index == field_index
                and point.direction == direction
                and point.frequency_lp_per_mm == frequency
            ]
            row.append(mean(values))
        matrix.append(row)

    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(frequencies)), [f"{frequency:g}" for frequency in frequencies])
    axis.set_yticks(range(len(row_keys)), [f"F{field} {direction_label(direction)}" for field, direction in row_keys])
    axis.set_xlabel("Frequency (lp/mm)")
    axis.set_title("Mean Absolute Error")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def plot_error_analysis(points: list[MTFErrorPoint], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.0), dpi=160)
    plot_scatter(axes[0, 0], points)
    plot_error_histogram(axes[0, 1], points)
    plot_group_boxplot(axes[1, 0], points)
    plot_mean_abs_heatmap(axes[1, 1], figure, points)
    figure.suptitle("Batch Huygens MTF Error Analysis", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def print_summary(
    *,
    opticscore_rows: list[dict[str, str]],
    zemax_rows: list[dict[str, str]],
    points: list[MTFErrorPoint],
    parameter_names: list[str],
    top_count: int,
    output_path: Path,
) -> None:
    errors = [point.error for point in points]
    abs_errors = [abs(error) for error in errors]
    top_points = sorted(points, key=lambda point: point.abs_error, reverse=True)[:top_count]

    print(f"opticscore_rows={len(opticscore_rows)}, zemax_rows={len(zemax_rows)}")
    print(f"paired_design_count={len({point.parameter_key for point in points})}")
    print(f"parameter_columns={parameter_names}")
    print(f"metric_count={len(points)}")
    print(f"mean_abs_error={mean(abs_errors):.9g}, mean_error={mean(errors):.9g}, rms_error={rms(errors):.9g}")
    print(
        "max_abs_error={:.9g}, column={}, opticscore_design={}, zemax_design={}, opticscore={:.9g}, zemax={:.9g}".format(
            top_points[0].abs_error,
            top_points[0].column,
            top_points[0].opticscore_design_index,
            top_points[0].zemax_design_index,
            top_points[0].opticscore_mtf,
            top_points[0].zemax_mtf,
        )
    )
    print("top_abs_errors:")
    for point in top_points:
        print(
            "  {} opticscore_design={} zemax_design={} opticscore={:.9g} zemax={:.9g} error={:.9g}".format(
                point.column,
                point.opticscore_design_index,
                point.zemax_design_index,
                point.opticscore_mtf,
                point.zemax_mtf,
                point.error,
            )
        )
    print(f"figure_saved={output_path}")


def main() -> None:
    args = parse_args()
    opticscore_csv = Path(args.opticscore_csv)
    zemax_csv = Path(args.zemax_csv)
    output_path = Path(args.output)

    opticscore_rows = load_csv(opticscore_csv)
    zemax_rows = load_csv(zemax_csv)
    points, params, _metrics = collect_error_points(opticscore_rows, zemax_rows)
    plot_error_analysis(points, output_path)
    print_summary(
        opticscore_rows=opticscore_rows,
        zemax_rows=zemax_rows,
        points=points,
        parameter_names=params,
        top_count=int(args.top_count),
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
