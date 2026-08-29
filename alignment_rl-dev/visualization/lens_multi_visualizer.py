from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from .visualize_utils import infer_lens_dof_names, render_metric_curve, render_mtf_panel


def _metric_colors(metric_values: np.ndarray):
    q_min = float(np.min(metric_values)) if metric_values.size else 0.0
    q_max = float(np.max(metric_values)) if metric_values.size else 1.0
    span = max(q_max - q_min, 1e-9)
    return cm.RdYlGn((metric_values - q_min) / span)


def decode_lens_state(state: np.ndarray, dof_names: list[str]) -> tuple[list[str], dict[str, dict[str, float]]]:
    groups: dict[str, dict[str, float]] = {}
    ordered_groups: list[str] = []
    for idx, raw_name in enumerate(dof_names[: len(state)]):
        if "_" in raw_name:
            tag, dof = raw_name.split("_", 1)
        else:
            tag, dof = "L1", raw_name
        if tag not in groups:
            groups[tag] = {}
            ordered_groups.append(tag)
        groups[tag][dof] = float(state[idx])
    return ordered_groups, groups


def render_lens_panel(
    ax: plt.Axes,
    states: list[np.ndarray],
    metric_values: list[float],
    step_idx: int,
    dof_names: list[str],
    axis_label: str,
) -> None:
    current = np.asarray(states[step_idx], dtype=np.float64)
    ordered_groups, current_map = decode_lens_state(current, dof_names)
    n_groups = max(len(ordered_groups), 1)
    z_positions = np.linspace(12.0, 12.0 + 22.0 * (n_groups - 1), n_groups)

    offset_key = "dx" if axis_label == "X" else "dy"
    tilt_key = "rx" if axis_label == "X" else "ry"
    all_offsets = []
    for prev_state in states[: step_idx + 1]:
        _, prev_map = decode_lens_state(np.asarray(prev_state, dtype=np.float64), dof_names)
        for tag in ordered_groups:
            all_offsets.append(prev_map.get(tag, {}).get(offset_key, 0.0))
    y_abs = max(max((abs(v) for v in all_offsets), default=0.0), 0.35)
    y_span = y_abs * 1.4

    ax.set_facecolor("#F6F8FB")
    ax.axhline(0.0, color="#9AA4AF", ls=":", lw=0.9, zorder=0)
    ax.plot([z_positions[0] - 10.0, z_positions[-1] + 12.0], [0.0, 0.0], color="#455A64", lw=1.0, alpha=0.35, zorder=0)

    slab_len = 6.5
    slab_half_thick = y_span * 0.08
    hist_alphas = np.linspace(0.12, 0.50, max(step_idx, 1)) if step_idx > 0 else np.array([])

    for group_idx, tag in enumerate(ordered_groups):
        z0 = z_positions[group_idx]
        offset = current_map.get(tag, {}).get(offset_key, 0.0)
        tilt_deg = current_map.get(tag, {}).get(tilt_key, 0.0)
        tilt_slope = np.tan(np.deg2rad(tilt_deg))
        dz = slab_len / 2.0
        y_left = offset - tilt_slope * dz
        y_right = offset + tilt_slope * dz
        poly = [
            (z0 - dz, y_left - slab_half_thick),
            (z0 + dz, y_right - slab_half_thick),
            (z0 + dz, y_right + slab_half_thick),
            (z0 - dz, y_left + slab_half_thick),
        ]
        patch = MplPolygon(poly, closed=True, facecolor="#D7E9FF", edgecolor="#0D47A1", linewidth=1.5, alpha=0.9, zorder=3)
        ax.add_patch(patch)
        ax.axvline(z0, color="#CFD8DC", lw=0.9, ls="--", zorder=1)
        ax.text(z0, y_span * 0.90, tag, ha="center", va="top", fontsize=9, color="#0D47A1", fontweight="bold")

        if step_idx > 0:
            hist_offsets = []
            for prev_state in states[:step_idx]:
                _, prev_map = decode_lens_state(np.asarray(prev_state, dtype=np.float64), dof_names)
                hist_offsets.append(prev_map.get(tag, {}).get(offset_key, 0.0))
            hist_metric = np.array(metric_values[:step_idx]) if step_idx > 0 else np.array([])
            hist_colors = _metric_colors(hist_metric) if len(hist_metric) == len(hist_offsets) else cm.Blues(np.linspace(0.25, 0.85, len(hist_offsets)))
            for i, hist_offset in enumerate(hist_offsets):
                alpha = hist_alphas[min(i, len(hist_alphas) - 1)] if len(hist_alphas) > 0 else 0.35
                ax.plot(z0, hist_offset, "o", color=hist_colors[i], alpha=alpha, markersize=4.2, zorder=4)
            if len(hist_offsets) > 1:
                ax.plot(np.full(len(hist_offsets), z0), hist_offsets, "-", color="#90A4AE", alpha=0.30, lw=1.0, zorder=2)

        ax.plot(z0, offset, "*", color="#FFD600", markersize=15, markeredgecolor="#333333", markeredgewidth=1.2, zorder=6)
        ax.text(z0, -y_span * 0.92, f"{offset_key}={offset:+.3f} mm\n{tilt_key}={tilt_deg:+.3f} deg", ha="center", va="bottom", fontsize=8.2, color="#37474F")

    ax.set_xlim(z_positions[0] - 12.0, z_positions[-1] + 12.0)
    ax.set_ylim(-y_span, y_span)
    ax.set_xlabel("Optical axis z [schematic mm]", fontsize=10)
    ax.set_ylabel(f"{axis_label} decenter [mm]", fontsize=10)
    ax.set_title(f"{axis_label}Z Lens Alignment View", fontsize=11, fontweight="bold", pad=6)
    ax.grid(True, alpha=0.16)


def render_lens_multi_frame(
    states: list[np.ndarray],
    metric_values: list[float],
    mtf_obs_log: list[np.ndarray],
    step_idx: int,
    method_name: str,
    episode_label: int,
    config: dict,
    threshold: float,
) -> plt.Figure:
    dof_names = infer_lens_dof_names(config, len(states[step_idx]))
    quality_cur = metric_values[step_idx]
    success = quality_cur >= threshold
    n_total = len(metric_values) - 1

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3.1, 2.1], hspace=0.34, wspace=0.24, left=0.06, right=0.97, top=0.90, bottom=0.07)
    ax_xz = fig.add_subplot(gs[0, 0])
    ax_yz = fig.add_subplot(gs[0, 1])
    ax_mtf = fig.add_subplot(gs[1, 0])
    ax_q = fig.add_subplot(gs[1, 1])

    render_lens_panel(ax_xz, states, metric_values, step_idx, dof_names, "X")
    render_lens_panel(ax_yz, states, metric_values, step_idx, dof_names, "Y")
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
