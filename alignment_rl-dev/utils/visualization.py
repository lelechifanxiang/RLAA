"""
可视化工具：对准轨迹、算法对比。
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# 统一绘图风格（中文字体自动降级处理）
matplotlib.rcParams.update({
    "axes.unicode_minus": False,
})


# ======================================================================
# 对准轨迹（单次 episode）
# ======================================================================

def plot_trajectory(
    result: dict,
    label: str = "RL",
    ax: plt.Axes | None = None,
    threshold: float = 0.95,
    **plot_kwargs,
) -> plt.Figure:
    """
    绘制单次对准过程中耦合效率随步数的变化曲线。

    result : _run_episode 或 Aligner.run() 返回的字典，包含 'etas' 键
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    etas = np.array(result["etas"])
    steps = np.arange(len(etas))
    ax.plot(steps, etas, label=label, **plot_kwargs)
    ax.axhline(threshold, color="gray", ls="--", lw=1, label=f"Threshold ({threshold:.0%})")
    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel("Coupling Efficiency η", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig


# ======================================================================
# 多算法对比（boxplot / 收敛曲线）
# ======================================================================

def plot_comparison(
    results_dict: dict[str, list[dict]],
    threshold: float = 0.95,
    max_steps_clip: int = 300,
    ylabel: str = "Coupling Efficiency η",
) -> plt.Figure:
    """
    对比多个算法在多次 episode 上的性能。

    参数
    ----
    results_dict : {算法名: [episode_result, ...]} 格式字典
    threshold    : 对准成功阈值
    max_steps_clip: 绘制收敛曲线时对步数上限截断

    绘制两个子图：
        左：收敛速度（每步平均 η）
        右：成功所需步数分布（violinplot）
    """
    fig = plt.figure(figsize=(13, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_curve = fig.add_subplot(gs[0])
    ax_violin = fig.add_subplot(gs[1])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- 左：分位数收敛曲线 ---
    # 显示：均值（实线）+ 25%~75% 分位数带（深色阴影）+ 0%~100% 分位数带（浅色阴影）
    for (name, results), color in zip(results_dict.items(), colors):
        # 对齐各轮耦合序列（用末尾值填充至最大长度）
        max_len = max(len(r["etas"]) for r in results)
        max_len = min(max_len, max_steps_clip)
        padded = []
        for r in results:
            e = np.array(r["etas"])[:max_len]
            pad = np.full(max_len - len(e), e[-1])
            padded.append(np.concatenate([e, pad]))
        arr = np.stack(padded)           # (n_episodes, max_len)
        xs = np.arange(max_len)

        mean_eta = arr.mean(axis=0)
        p25  = np.percentile(arr, 25, axis=0)
        p75  = np.percentile(arr, 75, axis=0)

        # 主均值曲线
        ax_curve.plot(xs, mean_eta, label=name, color=color, lw=2)
        # 25%~75% 分位带（深）
        ax_curve.fill_between(xs, p25, p75, alpha=0.20, color=color)

    # 说明分位数含义的图例代理项
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_extras = [
        Line2D([0], [0], color="gray", lw=1.5, ls="--",
               label=f"Goal {threshold:.0%}"),
        Line2D([0], [0], color="k", lw=1.0, ls="--", alpha=0.8,
               label="100th pct (max)"),
        Line2D([0], [0], color="k", lw=1.0, ls=":",  alpha=0.8,
               label="75th pct"),
        Patch(facecolor="k", alpha=0.20, label="25th–75th pct"),
        # Patch(facecolor="k", alpha=0.10, label="0th–100th pct"),
    ]
    ax_curve.axhline(threshold, color="gray", ls="--", lw=1.5)
    ax_curve.set_xlabel("Steps (env measurements)", fontsize=11)
    ax_curve.set_ylabel(ylabel, fontsize=11)
    ax_curve.set_title("Convergence Curve (Mean + Percentiles)", fontsize=12)
    ax_curve.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    # ax_curve.set_ylim(-0.02, 1.05)
    handles, labels = ax_curve.get_legend_handles_labels()
    ax_curve.legend(handles + legend_extras, labels + [e.get_label() for e in legend_extras],
                    fontsize=8, ncol=2)
    ax_curve.grid(True, alpha=0.3)

    # --- 右：成功步数分布 ---
    labels, data = [], []
    for name, results in results_dict.items():
        success_steps = [
            r["steps"] for r in results if r["success"]
        ]
        if success_steps:
            labels.append(f"{name}\n(succ={len(success_steps)}/{len(results)})")
            data.append(success_steps)

    if data:
        parts = ax_violin.violinplot(data, showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax_violin.set_xticks(np.arange(1, len(labels) + 1))
        ax_violin.set_xticklabels(labels, fontsize=9)
        ax_violin.set_ylabel("Steps to Success", fontsize=11)
        ax_violin.set_title("Steps-to-Align Distribution", fontsize=12)
        ax_violin.grid(True, axis="y", alpha=0.3)
    else:
        ax_violin.text(0.5, 0.5, "No successful episodes", ha="center", va="center",
                       transform=ax_violin.transAxes)

    fig.suptitle("Alignment Algorithm Comparison", fontsize=13, fontweight="bold")
    return fig
