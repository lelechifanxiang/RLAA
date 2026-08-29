from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv
from evaluate import run_lens_rl_episode
from visualize_episode import select_renderer
from visualization.visualize_utils import generate_frames, make_gif


def parse_steps_from_ckpt(path: str) -> int:
    match = re.search(r"(\d+)_steps", os.path.basename(path))
    if match is None:
        raise ValueError(f"无法从 checkpoint 文件名解析训练步数: {path}")
    return int(match.group(1))


def collect_checkpoints(ckpt_dir: str, selected_steps: set[int] | None = None) -> list[tuple[int, str]]:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoint 目录不存在: {ckpt_dir}")

    ckpts = []
    for name in os.listdir(ckpt_dir):
        if not name.endswith(".zip") or "_steps" not in name:
            continue
        full_path = os.path.join(ckpt_dir, name)
        try:
            steps = parse_steps_from_ckpt(full_path)
        except ValueError:
            continue
        if selected_steps is not None and steps not in selected_steps:
            continue
        ckpts.append((steps, full_path))

    ckpts.sort(key=lambda item: item[0])
    if not ckpts:
        raise RuntimeError(f"在目录中未找到可用 checkpoint: {ckpt_dir}")
    return ckpts


def build_results_data(method_name: str, episodes: list[dict], cfg, seed: int) -> dict:
    meta_env = LensAlignmentEnv(cfg=cfg)
    dof_names = list(meta_env.dof_names)
    meta_env.close()

    return {
        "results": {method_name: episodes},
        "config": {
            "task": "lens",
            "n_episodes": len(episodes),
            "lens_groups": [
                {
                    "surf_front": lg.surf_front,
                    "surf_rear": lg.surf_rear,
                    "z_source_surf": lg.z_source_surf,
                    "active_dofs": list(lg.active_dofs),
                }
                for lg in cfg.lens_groups
            ],
            "dof_names": dof_names,
            "seed": seed,
            "episode_seeds": [seed + ep for ep in range(len(episodes))],
            "mtf_field_coords": cfg.mtf_field_coords,
            "mtf_frequencies": cfg.mtf_frequencies,
            "mtf_field_indices": cfg.mtf_field_indices,
            "success_threshold": cfg.success_threshold,
            "metric_label": "Relative Log MTF Gain",
            "metric_value_mode": "linear",
            "metric_title": "Episode-relative MTF Gain Progress",
            "mtf_obs_mode": "log_ratio",
            "mtf_obs_title": "Current Episode-relative Log MTF Observation",
            "mtf_relative_clip": cfg.mtf_relative_clip,
        },
    }


def save_results_pkl(results_data: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results_data, f)


def evaluate_lens_checkpoint(model_path: str, algo: str, n_episodes: int, seed: int) -> tuple[dict, dict]:
    algo_cls = {"sac": SAC, "td3": TD3, "ppo": PPO}[algo]
    model = algo_cls.load(model_path)
    cfg = make_lens_rl_config()

    env = LensAlignmentEnv(cfg=cfg)
    try:
        episodes = []
        for ep in range(n_episodes):
            result = run_lens_rl_episode(model, env, seed=seed + ep)
            result["seed"] = seed + ep
            episodes.append(result)
    finally:
        env.close()

    method_name = f"RL ({algo.upper()})"
    results_data = build_results_data(method_name, episodes, cfg, seed)

    success_arr = np.array([ep["success"] for ep in episodes], dtype=np.float64)
    steps_arr = np.array([ep["steps"] for ep in episodes], dtype=np.float64)
    final_q_arr = np.array([ep["final_eta"] for ep in episodes], dtype=np.float64)

    stats = {
        "succ_rate": float(np.mean(success_arr)),
        "avg_steps": float(np.mean(steps_arr)),
        "median_steps": float(np.median(steps_arr)),
        "avg_final_q": float(np.mean(final_q_arr)),
        "median_final_q": float(np.median(final_q_arr)),
        "n_episodes": int(n_episodes),
    }
    return results_data, stats


def generate_report_gifs(
    results_data: dict,
    method_name: str,
    gif_root: str,
    gif_episodes: int,
    stride: int,
    dpi: int,
    fps: int,
) -> None:
    renderer = select_renderer(results_data, method_name, 0)
    n_available = len(results_data["results"][method_name])
    n_gifs = min(gif_episodes, n_available)

    for episode_idx in range(n_gifs):
        frame_dir = os.path.join(gif_root, f"episode_{episode_idx:02d}_frames")
        gif_path = os.path.join(gif_root, f"episode_{episode_idx:02d}.gif")
        frame_paths = generate_frames(
            results_data,
            method=method_name,
            episode_idx=episode_idx,
            output_dir=frame_dir,
            renderer=renderer,
            stride=stride,
            dpi=dpi,
            display_episode_idx=episode_idx,
        )
        if frame_paths:
            make_gif(frame_paths, gif_path, fps=fps)


def save_summary_csv(rows: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "train_steps",
                "model_path",
                "succ_rate",
                "avg_steps",
                "median_steps",
                "avg_final_q",
                "median_final_q",
                "results_pkl",
                "gif_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict], output_path: str, title: str) -> None:
    rows_sorted = sorted(rows, key=lambda item: item["train_steps"])
    train_steps = np.array([row["train_steps"] for row in rows_sorted], dtype=np.int64)
    succ_rates = np.array([row["succ_rate"] for row in rows_sorted], dtype=np.float64)
    avg_steps = np.array([row["avg_steps"] for row in rows_sorted], dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(train_steps, succ_rates, marker="o", lw=2.2, color="#1565C0", label="Success Rate")
    ax1.set_xlabel("Training Timesteps", fontsize=11)
    ax1.set_ylabel("Success Rate", color="#1565C0", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(train_steps, avg_steps, marker="s", lw=2.0, ls="--", color="#D84315", label="Avg Steps")
    ax2.set_ylabel("Average Evaluation Steps", color="#D84315", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#D84315")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    ax1.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评估 lens checkpoint，并为汇报生成 GIF 与统计图")
    parser.add_argument("--ckpt_dir", required=True, help="checkpoint 目录，例如 models/sac_lens_xxx_ckpt")
    parser.add_argument("--output_dir", required=True, help="汇报输出目录，例如 results/report_sac_lens_xxx")
    parser.add_argument("--algo", default="sac", choices=["sac", "td3", "ppo"], help="RL 算法类型")
    parser.add_argument("--n_episodes", type=int, default=100, help="每个 checkpoint 的评估 episode 数")
    parser.add_argument("--gif_episodes", type=int, default=10, help="每个 checkpoint 生成 GIF 的 episode 数")
    parser.add_argument("--seed", type=int, default=0, help="评估起始随机种子")
    parser.add_argument("--stride", type=int, default=1, help="生成 GIF 时的帧间隔")
    parser.add_argument("--dpi", type=int, default=120, help="渲染帧分辨率")
    parser.add_argument("--fps", type=int, default=8, help="GIF 帧率")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="仅评估指定训练步数，例如 --steps 50000 200000")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    eval_dir = os.path.join(output_dir, "evaluations")
    gif_dir = os.path.join(output_dir, "gifs")
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    selected_steps = set(args.steps) if args.steps else None
    ckpts = collect_checkpoints(args.ckpt_dir, selected_steps=selected_steps)
    method_name = f"RL ({args.algo.upper()})"
    rows = []

    print(f"共发现 {len(ckpts)} 个 checkpoint，开始评估 lens RL 模型...")
    for idx, (train_steps, model_path) in enumerate(ckpts, start=1):
        print(f"\n[{idx}/{len(ckpts)}] 评估 {os.path.basename(model_path)}")
        results_data, stats = evaluate_lens_checkpoint(
            model_path=model_path,
            algo=args.algo,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )

        step_tag = f"{train_steps:07d}_steps"
        pkl_path = os.path.join(eval_dir, f"eval_lens_{step_tag}.pkl")
        gif_root = os.path.join(gif_dir, step_tag)
        os.makedirs(gif_root, exist_ok=True)

        save_results_pkl(results_data, pkl_path)
        generate_report_gifs(
            results_data=results_data,
            method_name=method_name,
            gif_root=gif_root,
            gif_episodes=args.gif_episodes,
            stride=args.stride,
            dpi=args.dpi,
            fps=args.fps,
        )

        row = {
            "train_steps": train_steps,
            "model_path": os.path.abspath(model_path),
            "succ_rate": stats["succ_rate"],
            "avg_steps": stats["avg_steps"],
            "median_steps": stats["median_steps"],
            "avg_final_q": stats["avg_final_q"],
            "median_final_q": stats["median_final_q"],
            "results_pkl": os.path.abspath(pkl_path),
            "gif_dir": os.path.abspath(gif_root),
        }
        rows.append(row)

        print(
            f"    train_steps={train_steps:,} | succ_rate={stats['succ_rate']:.1%} | "
            f"avg_steps={stats['avg_steps']:.2f} | avg_final_q={stats['avg_final_q']:.4f}"
        )
        print(f"    pkl={pkl_path}")
        print(f"    gifs={gif_root}")

    csv_path = os.path.join(summary_dir, "checkpoint_summary.csv")
    plot_path = os.path.join(summary_dir, "success_rate_vs_training_steps.png")
    save_summary_csv(rows, csv_path)
    plot_summary(rows, plot_path, title=f"{args.algo.upper()} Lens Checkpoint Evaluation")

    readme_path = os.path.join(summary_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Lens checkpoint report bundle\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint dir: {os.path.abspath(args.ckpt_dir)}\n")
        f.write(f"Episodes per checkpoint: {args.n_episodes}\n")
        f.write(f"GIF episodes per checkpoint: {args.gif_episodes}\n")
        f.write(f"Seed start: {args.seed}\n")
        f.write(f"Algorithm: {args.algo}\n")
        f.write(f"Summary CSV: {os.path.abspath(csv_path)}\n")
        f.write(f"Summary figure: {os.path.abspath(plot_path)}\n")

    print("\n全部完成。")
    print(f"汇总 CSV: {csv_path}")
    print(f"汇总图像: {plot_path}")
    print(f"汇报目录: {output_dir}")


if __name__ == "__main__":
    main()