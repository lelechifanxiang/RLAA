from __future__ import annotations

import hashlib
import os
import pickle
import sys
import warnings
from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from config import LensEnvConfig, LensGroupConfig, make_lens_rl_config
from env.lens_env import LensAlignmentEnv
from evaluate import run_lens_rl_episode


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def model_cls(algo: str):
    algo = algo.lower()
    mapping = {"sac": SAC, "td3": TD3, "ppo": PPO}
    if algo not in mapping:
        raise ValueError(f"不支持的算法: {algo}，仅支持 {list(mapping)}")
    return mapping[algo]


def infer_task(results_data: dict[str, Any]) -> str:
    return results_data.get("config", {}).get("task", "lens")


def infer_lens_dof_names(config: dict[str, Any], state_dim: int) -> list[str]:
    if "dof_names" in config:
        return list(config["dof_names"])

    n_groups = max(len(config.get("lens_groups", [])), 1)
    if state_dim % n_groups != 0:
        return [f"dof_{i}" for i in range(state_dim)]

    per_group = state_dim // n_groups
    if per_group == 2:
        base_names = ["dx", "dy"]
    elif per_group == 4:
        base_names = ["dx", "dy", "rx", "ry"]
    elif per_group == 5:
        base_names = ["dx", "dy", "dz", "rx", "ry"]
    else:
        base_names = [f"d{i}" for i in range(per_group)]

    names = []
    for gi in range(n_groups):
        tag = f"L{gi+1}"
        names.extend(f"{tag}_{name}" for name in base_names)
    return names[:state_dim]


def is_single_lens_2d_case(config: dict[str, Any], state_dim: int) -> bool:
    dof_names = infer_lens_dof_names(config, state_dim)
    if len(dof_names) != 2:
        return False
    return dof_names in (["L1_dx", "L1_dy"], ["L2_dx", "L2_dy"], ["dx", "dy"])


def build_lens_config_dict(env: LensAlignmentEnv) -> dict[str, Any]:
    cfg = env.cfg
    return {
        "task": "lens",
        "success_threshold": cfg.success_threshold,
        "metric_label": "Relative Log MTF Gain",
        "metric_value_mode": "linear",
        "metric_title": "Episode-relative MTF Gain Progress",
        "mtf_obs_mode": "log_ratio",
        "mtf_obs_title": "Current Episode-relative Log MTF Observation",
        "mtf_relative_clip": cfg.mtf_relative_clip,
        "mtf_field_coords": [tuple(v) for v in cfg.mtf_field_coords],
        "mtf_frequencies": list(cfg.mtf_frequencies),
        "mtf_field_indices": list(cfg.mtf_field_indices),
        "lens_groups": [
            {
                "surf_front": lg.surf_front,
                "surf_rear": lg.surf_rear,
                "z_source_surf": lg.z_source_surf,
                "active_dofs": list(lg.active_dofs),
            }
            for lg in cfg.lens_groups
        ],
        "dof_names": list(env.dof_names),
    }


def run_lens_episode_compat(model, env: LensAlignmentEnv, seed: int | None = None) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    qualities = [info["quality_metric"]]
    states = [info["state"].copy()]
    actions_log = []
    mtf_obs_log = [info["mtf_obs"].copy()]
    terminated, truncated = False, False
    env_dim = int(env.action_space.shape[0])

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size > env_dim:
            action = action[:env_dim]
        elif action.size < env_dim:
            action = np.pad(action, (0, env_dim - action.size))

        obs, _r, terminated, truncated, info = env.step(action)
        qualities.append(info["quality_metric"])
        states.append(info["state"].copy())
        actions_log.append(action.copy())
        mtf_obs_log.append(info["mtf_obs"].copy())

    return {
        "qualities": qualities,
        "etas": qualities,
        "states": states,
        "actions": actions_log,
        "mtf_obs": mtf_obs_log,
        "steps": len(actions_log),
        "success": info["success"],
        "final_eta": qualities[-1],
    }


def load_manual_results(
    model_path: str,
    algo: str,
    episode_idx: int,
    seed: int,
) -> tuple[dict[str, Any], str, int]:
    algo_class = model_cls(algo)
    model = algo_class.load(model_path)
    method_name = f"RL ({algo.upper()})"

    env = LensAlignmentEnv(cfg=make_lens_rl_config())
    model_dim = int(np.prod(model.action_space.shape))
    env_dim = int(np.prod(env.action_space.shape))
    rollout_fn = run_lens_rl_episode if model_dim == env_dim else run_lens_episode_compat
    config = build_lens_config_dict(env)
    config["model_action_dim"] = model_dim
    config["env_action_dim"] = env_dim
    config["seed"] = seed
    config["episode_seeds"] = [seed + episode_idx]

    selected_episode = None
    try:
        for ep in range(episode_idx + 1):
            selected_episode = rollout_fn(model, env, seed=seed + ep)
    finally:
        env.close()

    if selected_episode is None:
        raise RuntimeError("手动 rollout 失败，未生成任何 episode")

    results_data = {
        "results": {method_name: [{**selected_episode, "seed": seed + episode_idx}]},
        "config": config,
    }
    return results_data, method_name, episode_idx


def _rebuild_lens_env_config(config: dict[str, Any]) -> LensEnvConfig:
    cfg = make_lens_rl_config()

    lens_groups = config.get("lens_groups")
    if lens_groups:
        rebuilt_groups = []
        for raw in lens_groups:
            rebuilt_groups.append(
                LensGroupConfig(
                    surf_front=int(raw["surf_front"]),
                    surf_rear=int(raw["surf_rear"]),
                    z_source_surf=int(raw["z_source_surf"]),
                    active_dofs=list(raw.get("active_dofs", ["dx", "dy"])),
                )
            )
        cfg.lens_groups = rebuilt_groups

    if "mtf_field_coords" in config:
        cfg.mtf_field_coords = [tuple(v) for v in config["mtf_field_coords"]]
    if "mtf_field_indices" in config:
        cfg.mtf_field_indices = list(config["mtf_field_indices"])
    if "mtf_frequencies" in config:
        cfg.mtf_frequencies = list(config["mtf_frequencies"])
    if "success_threshold" in config:
        cfg.success_threshold = float(config["success_threshold"])
    if "mtf_relative_clip" in config:
        cfg.mtf_relative_clip = float(config["mtf_relative_clip"])
    return cfg


def _resolve_episode_seed(config: dict[str, Any], ep: dict[str, Any], episode_idx: int) -> int | None:
    if "seed" in ep and ep["seed"] is not None:
        return int(ep["seed"])

    episode_seeds = config.get("episode_seeds")
    if isinstance(episode_seeds, list) and episode_idx < len(episode_seeds):
        return int(episode_seeds[episode_idx])

    if "seed" in config and config["seed"] is not None:
        return int(config["seed"]) + episode_idx

    if config.get("task") == "lens":
        warnings.warn("结果文件未记录 lens episode seed，默认按 seed=0 推断；如背景形貌与轨迹不一致，请重新运行 evaluate.py 生成新结果。")
        return episode_idx

    return None


def _landscape_cache_path(config: dict[str, Any], episode_seed: int, grid_size: int) -> str:
    cache_payload = {
        "seed": int(episode_seed),
        "grid_size": int(grid_size),
        "lens_groups": config.get("lens_groups", []),
        "mtf_field_coords": config.get("mtf_field_coords", []),
        "mtf_field_indices": config.get("mtf_field_indices", []),
        "mtf_frequencies": config.get("mtf_frequencies", []),
        "metric_mode": config.get("mtf_obs_mode", "log_ratio"),
    }
    digest = hashlib.sha1(repr(cache_payload).encode("utf-8")).hexdigest()[:12]
    cache_dir = os.path.join("results", "landscape_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"lens_landscape_seed{episode_seed}_{digest}.npz")


def get_lens_2d_landscape(config: dict[str, Any], ep: dict[str, Any], episode_idx: int) -> dict[str, Any] | None:
    dof_names = infer_lens_dof_names(config, len(ep.get("states", [[0.0, 0.0]])[0]))
    if not is_single_lens_2d_case(config, len(dof_names)):
        return None

    episode_seed = _resolve_episode_seed(config, ep, episode_idx)
    if episode_seed is None:
        return None

    grid_size = 81
    cache_path = _landscape_cache_path(config, episode_seed, grid_size)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return {
            "xs": cached["xs"],
            "ys": cached["ys"],
            "quality": cached["quality"],
            "seed": int(cached["seed"]),
            "cache_path": cache_path,
        }

    env = LensAlignmentEnv(cfg=_rebuild_lens_env_config(config))
    try:
        env.reset(seed=episode_seed)
        x_lim = float(env._action_limit[0])
        y_lim = float(env._action_limit[1])
        xs = np.linspace(-x_lim, x_lim, grid_size, dtype=np.float64)
        ys = np.linspace(-y_lim, y_lim, grid_size, dtype=np.float64)
        quality = np.empty((grid_size, grid_size), dtype=np.float32)

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                env._mgr.apply_alignment_state(np.array([x, y], dtype=np.float64))
                quality[yi, xi] = env._mgr.quality_metric()
    finally:
        env.close()

    np.savez_compressed(cache_path, xs=xs, ys=ys, quality=quality, seed=np.array(episode_seed, dtype=np.int32))
    return {
        "xs": xs,
        "ys": ys,
        "quality": quality,
        "seed": int(episode_seed),
        "cache_path": cache_path,
    }


def render_metric_curve(
    ax: plt.Axes,
    values: list[float],
    step_idx: int,
    threshold: float,
    ylabel: str,
    title: str,
    value_mode: str = "percent",
) -> None:
    n = len(values)
    xs = np.arange(n)

    if step_idx + 1 < n:
        ax.plot(xs[step_idx:], values[step_idx:], "--", color="#BDBDBD", lw=1.2, zorder=2, label="future (known)")
    ax.plot(xs[: step_idx + 1], values[: step_idx + 1], ".-", color="#1565C0", lw=2.0, zorder=3, label="past")
    ax.scatter([step_idx], [values[step_idx]], s=140, color="#FFD600", edgecolors="#333333", linewidths=1.5, zorder=5)
    if value_mode == "percent":
        ax.axhline(threshold, color="#388E3C", ls="--", lw=1.5, alpha=0.85, label=f"Goal {threshold:.0%}")
        ax.axhspan(threshold, 1.02, alpha=0.06, color="#388E3C")
    else:
        ax.axhline(threshold, color="#388E3C", ls="--", lw=1.5, alpha=0.85, label=f"Goal {threshold:+.3f}")
    ax.set_xlim(-0.5, max(n - 0.5, 10))
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, pad=4)
    if value_mode == "percent":
        ax.set_ylim(-0.03, 1.06)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    else:
        values_arr = np.asarray(values, dtype=np.float64)
        y_min = min(float(np.min(values_arr)), threshold)
        y_max = max(float(np.max(values_arr)), threshold)
        pad = max(0.05 * (y_max - y_min), 0.02)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8.5, loc="lower right", ncol=3)
    ax.grid(True, alpha=0.2)


def render_mtf_panel(ax: plt.Axes, mtf_obs: np.ndarray, config: dict[str, Any]) -> None:
    freq_labels = config.get("mtf_frequencies", [])
    field_indices = config.get("mtf_field_indices", [0, 1, 2])
    n_freq = max(len(freq_labels), 1)
    n_fields = max(len(field_indices), 1)
    expected = n_fields * 2 * n_freq

    mtf_obs = np.asarray(mtf_obs, dtype=np.float64).flatten()
    if mtf_obs.size != expected:
        padded = np.zeros(expected, dtype=np.float64)
        padded[: min(expected, mtf_obs.size)] = mtf_obs[: min(expected, mtf_obs.size)]
        mtf_obs = padded
    grid = mtf_obs.reshape(n_fields * 2, n_freq)

    mtf_obs_mode = config.get("mtf_obs_mode", "normalized")
    if mtf_obs_mode == "log_ratio":
        clip = float(config.get("mtf_relative_clip", 2.0))
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-clip, vmax=clip)
    else:
        im = ax.imshow(grid, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    row_labels = []
    for fi in field_indices:
        row_labels.extend([f"F{fi}-S", f"F{fi}-T"])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(n_freq))
    ax.set_xticklabels([str(int(f)) if float(f).is_integer() else f"{f:g}" for f in freq_labels])
    ax.set_xlabel("Spatial frequency [lp/mm]", fontsize=10)
    ax.set_title(config.get("mtf_obs_title", "Current Normalized MTF Observation"), fontsize=11, fontweight="bold", pad=6)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if mtf_obs_mode == "log_ratio":
                color = "white" if abs(grid[i, j]) > 0.35 * float(config.get("mtf_relative_clip", 2.0)) else "#16324F"
            else:
                color = "white" if grid[i, j] < 0.55 else "#16324F"
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=color)


def normalize_episode_data(
    results_data: dict[str, Any],
    method: str,
    episode_idx: int,
) -> tuple[str, dict[str, Any], dict[str, Any], list[np.ndarray], list[float], list[np.ndarray], float]:
    if method not in results_data["results"]:
        available = list(results_data["results"].keys())
        print(f"错误：未找到方法 '{method}'，可用方法：{available}")
        sys.exit(1)

    episodes = results_data["results"][method]
    if episode_idx >= len(episodes):
        print(f"错误：episode 索引 {episode_idx} 超出范围（共 {len(episodes)} 条）。")
        sys.exit(1)

    ep = episodes[episode_idx]
    config = results_data["config"]
    task = infer_task(results_data)
    threshold = config.get("success_threshold", config.get("threshold", 0.95))

    states = ep.get("states")
    if states is None:
        print("错误：结果文件中缺少 'states' 字段。请使用最新版 evaluate.py 重新生成结果文件。")
        sys.exit(1)

    metric_values = ep.get("qualities", ep.get("etas", []))
    min_len = min(len(states), len(metric_values))
    if task == "lens":
        mtf_obs_log = ep.get("mtf_obs", [])
        if mtf_obs_log:
            min_len = min(min_len, len(mtf_obs_log))
            mtf_obs_log = mtf_obs_log[:min_len]
        else:
            n_fields = max(len(config.get("mtf_field_indices", [0, 1, 2])), 1)
            n_freq = max(len(config.get("mtf_frequencies", [30, 60, 100])), 1)
            mtf_obs_log = [np.zeros(n_fields * 2 * n_freq, dtype=np.float32) for _ in range(min_len)]
    else:
        mtf_obs_log = []

    states = states[:min_len]
    metric_values = metric_values[:min_len]
    return task, config, ep, states, metric_values, mtf_obs_log, threshold


def generate_frames(
    results_data: dict[str, Any],
    method: str,
    episode_idx: int,
    output_dir: str,
    renderer: Callable[..., plt.Figure],
    stride: int = 1,
    dpi: int = 120,
    display_episode_idx: int | None = None,
) -> list[str]:
    task, config, ep, states, metric_values, mtf_obs_log, threshold = normalize_episode_data(
        results_data,
        method,
        episode_idx,
    )

    os.makedirs(output_dir, exist_ok=True)
    n_total = len(metric_values)
    step_indices = list(range(0, n_total, max(stride, 1)))
    if (n_total - 1) not in step_indices:
        step_indices.append(n_total - 1)

    display_episode_idx = episode_idx if display_episode_idx is None else display_episode_idx
    metric_name = "Q" if task == "lens" else "η"
    print(f"\n  任务    : {task}")
    print(f"  方法    : {method}")
    print(f"  Episode : {display_episode_idx}  (success={ep['success']}, final_{metric_name}={ep['final_eta']:.4f})")
    print(f"  总记录步: {n_total - 1}  →  渲染 {len(step_indices)} 帧  (stride={stride})")
    print(f"  输出目录: {output_dir}\n")

    landscape = None
    if task == "lens" and is_single_lens_2d_case(config, len(states[0])):
        landscape = get_lens_2d_landscape(config, ep, episode_idx)
        if landscape is not None:
            print(f"  形貌图  : seed={landscape['seed']}  cache={landscape['cache_path']}")

    frame_paths = []
    for frame_num, step_idx in enumerate(step_indices):
        render_kwargs = {
            "states": states,
            "metric_values": metric_values,
            "mtf_obs_log": mtf_obs_log,
            "step_idx": step_idx,
            "method_name": method,
            "episode_label": display_episode_idx,
            "config": config,
            "threshold": threshold,
        }
        if landscape is not None:
            render_kwargs["landscape"] = landscape
        fig = renderer(**render_kwargs)
        out_path = os.path.join(output_dir, f"frame_{frame_num:04d}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", metadata={"Software": "visualize_episode"})
        plt.close(fig)
        frame_paths.append(out_path)
        if (frame_num + 1) % 20 == 0 or frame_num == len(step_indices) - 1:
            print(f"  [{frame_num + 1:>4d}/{len(step_indices)}] {out_path}")

    return frame_paths


def make_gif(frame_paths: list[str], output_path: str, fps: int = 8) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            import imageio  # type: ignore
        except ImportError:
            print("未安装 imageio，跳过 GIF 生成。可运行：pip install imageio")
            return

    print(f"\n合成 GIF：{output_path}  ({len(frame_paths)} 帧，{fps} fps) ...")
    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(output_path, frames, duration=1.0 / fps, loop=0)
    print(f"GIF 已保存至 {output_path}")


def load_results_file(results_path: str) -> dict[str, Any]:
    print(f"加载结果文件：{results_path}")
    with open(results_path, "rb") as f:
        return pickle.load(f)


def print_available_methods(results_data: dict[str, Any]) -> None:
    task = infer_task(results_data)
    available = list(results_data["results"].keys())
    print(f"\n任务类型：{task}")
    print(f"可用方法（共 {len(available)} 种）：")
    for i, method_name in enumerate(available):
        eps = results_data["results"][method_name]
        n_suc = sum(ep["success"] for ep in eps)
        has_s = "✓" if eps and eps[0].get("states") is not None else "✗ (需重新评估)"
        print(f"  [{i}] {method_name:<30} {len(eps)} episodes, {n_suc} success, states={has_s}")
    print("\n请用 --method <名称> 指定方法，--episode <索引> 指定序列。")
