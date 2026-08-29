from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from .visualize_utils import infer_lens_dof_names, render_metric_curve, render_mtf_panel


def _metric_colors(metric_values: np.ndarray):
    q_min = float(np.min(metric_values)) if metric_values.size else 0.0
    q_max = float(np.max(metric_values)) if metric_values.size else 1.0
    span = max(q_max - q_min, 1e-9)
    return cm.RdYlGn((metric_values - q_min) / span)


def render_lens_xy_panel(
    ax: plt.Axes,
    states: list[np.ndarray],
    metric_values: list[float],
    step_idx: int,
    threshold: float,
    landscape: dict | None = None,
) -> None:
    hist = np.asarray(states[: step_idx + 1], dtype=np.float64)
    xs = hist[:, 0]
    ys = hist[:, 1]
    q = np.asarray(metric_values[: step_idx + 1], dtype=np.float64)

    span = max(np.max(np.abs(xs)) if xs.size else 0.0, np.max(np.abs(ys)) if ys.size else 0.0, 0.35)
    if landscape is not None:
        span = max(
            span,
            float(np.max(np.abs(np.asarray(landscape["xs"], dtype=np.float64)))),
            float(np.max(np.abs(np.asarray(landscape["ys"], dtype=np.float64)))),
        )
    span *= 1.35

    ax.set_facecolor("#F6F8FB")
    if landscape is not None:
        q_map = np.asarray(landscape["quality"], dtype=np.float64)
        xs_map = np.asarray(landscape["xs"], dtype=np.float64)
        ys_map = np.asarray(landscape["ys"], dtype=np.float64)
        q_min = float(np.min(q_map))
        q_max = float(np.max(q_map))
        ax.imshow(
            q_map,
            extent=(float(xs_map[0]), float(xs_map[-1]), float(ys_map[0]), float(ys_map[-1])),
            origin="lower",
            cmap="RdYlGn",
            vmin=q_min,
            vmax=q_max,
            alpha=0.62,
            zorder=0,
            interpolation="bicubic",
        )
        contour_levels = np.unique(np.array([threshold, 0.0], dtype=np.float64))
        ax.contour(xs_map, ys_map, q_map, levels=contour_levels, colors=["#1B5E20", "#37474F"][: len(contour_levels)], linewidths=1.1, linestyles=["--", ":"][: len(contour_levels)], alpha=0.85, zorder=1)

    ax.axhline(0.0, color="#9AA4AF", ls=":", lw=0.9)
    ax.axvline(0.0, color="#9AA4AF", ls=":", lw=0.9)

    colors = _metric_colors(q)

    if len(xs) > 1:
        for i in range(len(xs) - 1):
            color = colors[i]
            alpha = 0.18 + 0.55 * (i + 1) / len(xs)
            ax.plot(xs[i:i + 2], ys[i:i + 2], "-", color=color, alpha=alpha, lw=2.0, zorder=2)

    if len(xs) > 0:
        alphas = np.linspace(0.18, 0.80, len(xs))
        for x, y, color, alpha in zip(xs, ys, colors, alphas):
            ax.plot(x, y, "o", color=color, alpha=alpha, markersize=5.0, zorder=3)

    ax.plot(xs[-1], ys[-1], "*", color="#FFD600", markersize=18, markeredgecolor="#333333", markeredgewidth=1.3, zorder=5)
    ax.text(xs[-1], ys[-1], f"  ({xs[-1]:+.3f}, {ys[-1]:+.3f}) mm", ha="left", va="bottom", fontsize=9, color="#37474F")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("dx [mm]", fontsize=10)
    ax.set_ylabel("dy [mm]", fontsize=10)
    ax.set_title("Lens 2D Decenter Trajectory", fontsize=12, fontweight="bold", pad=6)
    ax.grid(True, alpha=0.16)


def render_lens_2d_frame(
    states: list[np.ndarray],
    metric_values: list[float],
    mtf_obs_log: list[np.ndarray],
    step_idx: int,
    method_name: str,
    episode_label: int,
    config: dict,
    threshold: float,
    landscape: dict | None = None,
) -> plt.Figure:
    dof_names = infer_lens_dof_names(config, len(states[step_idx]))
    quality_cur = metric_values[step_idx]
    success = quality_cur >= threshold
    n_total = len(metric_values) - 1

    fig = plt.figure(figsize=(14.5, 9.5))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1.0], height_ratios=[1.2, 1.0], hspace=0.30, wspace=0.25, left=0.06, right=0.97, top=0.90, bottom=0.08)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_mtf = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1])

    render_lens_xy_panel(ax_xy, states, metric_values, step_idx, threshold, landscape=landscape)
    render_mtf_panel(ax_mtf, mtf_obs_log[step_idx], config)
    render_metric_curve(
        ax_q,
        metric_values,
        step_idx,
        threshold,
        ylabel=config.get("metric_label", "Quality"),
        title=config.get("metric_title", "Normalized MTF Quality Progress"),
        value_mode=config.get("metric_value_mode", "percent"),
    )

    state_text = ", ".join(f"{name}={float(val):+.3f}" for name, val in zip(dof_names, states[step_idx]))
    status = "✓ Aligned!" if success else f"Q = {quality_cur:.4f}"
    fig.suptitle(f"Method: {method_name}  ·  Episode {episode_label}  ·  Step {step_idx} / {n_total}  ·  {status}\n{state_text}", fontsize=12.5, fontweight="bold", y=0.965)
    return fig
