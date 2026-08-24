"""
评估脚本：对比 RL 智能体与传统基准算法在镜头主动对准任务上的性能。

用法
----
1. 仅运行基准算法对比（无需预训练模型）：
    python evaluate.py

2. 加载已训练的 SAC 模型进行完整对比：
    python evaluate.py --model_path models/sac_lens_XXX_final

3. 仅运行 RL 评估（跳过所有 baseline）：
    python evaluate.py --model_path models/sac_lens_XXX_final --only_rl

输出
----
- 控制台：各算法成功率、平均步数、平均质量指标
- figures/lens_comparison.png : 多 episode 统计对比
- results/eval_lens_results_XXX.pkl : 完整轨迹数据
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3, PPO

from config import LensEnvConfig, make_lens_rl_config
from env.lens_env import LensAlignmentEnv
from agents.baseline_lens import LensHillClimbAligner, LensCoordinateSearchAligner
from utils.visualization import plot_comparison


# ======================================================================
# 工具：使用 SB3 模型运行单 lens episode
# ======================================================================

def run_lens_rl_episode(model, env: LensAlignmentEnv, seed: int | None = None) -> dict:
    obs, info = env.reset(seed=seed)
    qualities = [info["quality_metric"]]
    states = [info["state"].copy()]
    actions_log = []
    mtf_obs_log = [info["mtf_obs"].copy()]
    terminated, truncated = False, False
    env_dim = int(np.prod(env.action_space.shape))

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
        "etas": qualities,           # 统一接口（质量指标 ≡ episode 相对 log MTF 增益）
        "states": states,
        "actions": actions_log,
        "mtf_obs": mtf_obs_log,
        "steps": len(actions_log),
        "success": info["success"],
        "seed": seed,
        "final_eta": qualities[-1],
    }


# ======================================================================
# 镜头任务评估主函数
# ======================================================================

def evaluate_lens(
    model_path: str | None = None,
    n_episodes: int = 50,
    algo: str = "sac",
    only_rl: bool = False,
    seed: int = 0,
) -> None:
    """评估镜头主动对准（LensAlignmentEnv）。

    支持 Hill Climbing 基线与 RL 智能体评估。
    """
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    lens_cfg = make_lens_rl_config()
    meta_env = LensAlignmentEnv(cfg=lens_cfg)
    lens_dof_names = list(meta_env.dof_names)
    meta_env.close()
    results: dict[str, list[dict]] = {}

    if only_rl:
        print("已启用 --only_rl，跳过所有 baseline 方法。")
    else:
        print(f"运行 Lens Hill Climbing 基线（{n_episodes} episodes）...")
        hc_aligner = LensHillClimbAligner()
        hc_results = []
        for ep in range(n_episodes):
            if ep % 5 == 0:
                print(f"  Episode {ep + 1}/{n_episodes}...")
            env = LensAlignmentEnv(cfg=lens_cfg)
            hc_results.append(hc_aligner.run(env, seed=seed + ep))
            hc_results[-1]["seed"] = seed + ep
            env.close()
        results["Hill Climbing"] = hc_results

        print(f"运行 Lens Coordinate Search 基线（{n_episodes} episodes）...")
        cs_aligner = LensCoordinateSearchAligner()
        cs_results = []
        for ep in range(n_episodes):
            if ep % 5 == 0:
                print(f"  Episode {ep + 1}/{n_episodes}...")
            env = LensAlignmentEnv(cfg=lens_cfg)
            cs_results.append(cs_aligner.run(env, seed=seed + ep))
            cs_results[-1]["seed"] = seed + ep
            env.close()
        results["Coordinate Search"] = cs_results

    if model_path is not None:
        print(f"加载 RL 模型：{model_path}")
        algo_cls = {"sac": SAC, "td3": TD3, "ppo": PPO}.get(algo, SAC)
        rl_model = algo_cls.load(model_path)

        print(f"运行 RL 镜头对准器（{n_episodes} episodes）...")
        env = LensAlignmentEnv(cfg=lens_cfg)
        rl_results = []
        for ep in range(n_episodes):
            if ep % 5 == 0: print(f"  Episode {ep + 1}/{n_episodes}...")
            rl_results.append(run_lens_rl_episode(rl_model, env, seed=seed + ep))
        results[f"RL ({algo.upper()})"] = rl_results
        env.close()

    # ------ 控制台统计 ------
    header = f"{'Algorithm':<20} {'SuccRate':>8} {'Avg Steps':>10} {'Avg Q':>8} {'Median Q':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, res_list in results.items():
        succ_rate = np.mean([r["success"] for r in res_list])
        avg_steps = np.mean([r["steps"] for r in res_list])
        avg_q = np.mean([r["final_eta"] for r in res_list])
        med_q = np.median([r["final_eta"] for r in res_list])
        print(f"{name:<20} {succ_rate:>7.1%} {avg_steps:>10.1f} {avg_q:>8.4f} {med_q:>9.4f}")
    print("=" * len(header))

    # ------ 保存轨迹 pkl ------
    _ts = time.strftime("%Y%m%d_%H%M%S")
    _path = os.path.join("results", f"eval_lens_results_{_ts}.pkl")
    with open(_path, "wb") as _f:
        pickle.dump({
            "results": results,
            "config": {
                "task": "lens",
                "n_episodes": n_episodes,
                "lens_groups": [
                    {
                        "surf_front": lg.surf_front,
                        "surf_rear": lg.surf_rear,
                        "z_source_surf": lg.z_source_surf,
                        "active_dofs": list(lg.active_dofs),
                    }
                    for lg in lens_cfg.lens_groups
                ],
                "dof_names": lens_dof_names,
                "seed": seed,
                "episode_seeds": [seed + ep for ep in range(n_episodes)],
                "mtf_field_coords": lens_cfg.mtf_field_coords,
                "mtf_frequencies": lens_cfg.mtf_frequencies,
                "mtf_field_indices": lens_cfg.mtf_field_indices,
                "success_threshold": lens_cfg.success_threshold,
                "metric_label": "Relative Log MTF Gain",
                "metric_value_mode": "linear",
                "metric_title": "Episode-relative MTF Gain Progress",
                "mtf_obs_mode": "log_ratio",
                "mtf_obs_title": "Current Episode-relative Log MTF Observation",
                "mtf_relative_clip": lens_cfg.mtf_relative_clip,
            },
        }, _f)
    print(f"完整轨迹已保存至 {_path}\n")

    # ------ 收敛曲线 ------
    fig_cmp = plot_comparison(results, threshold=lens_cfg.success_threshold,
                               ylabel="Relative Log MTF Gain")
    fig_cmp.savefig("figures/lens_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cmp)
    print("对比图已保存到 figures/lens_comparison.png")


# ======================================================================
# CLI 入口
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="镜头主动对准算法评估")
    parser.add_argument("--model_path", default=None,
                        help="已训练 RL 模型路径（不含 .zip 后缀）")
    parser.add_argument("--algo", default="sac", choices=["sac", "td3", "ppo"],
                        help="RL 算法（用于加载模型，default: sac）")
    parser.add_argument(
        "--only_rl",
        action="store_true",
        help="仅评估 RL 模型，跳过所有 baseline",
    )
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="评估 episode 数量 (default: 5)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluate_lens(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        algo=args.algo,
        only_rl=args.only_rl,
        seed=args.seed,
    )
