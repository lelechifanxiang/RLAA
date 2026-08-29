"""
分析4自由度交替对准的运动轨迹和性能。

功能：
    1. 可视化交替运动轨迹（偏心 vs 倾斜）
    2. 对比交替模式和同时模式的收敛速度
    3. 分析各自由度的调整幅度和频率
    4. 绘制MTF质量改进曲线
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stable_baselines3 import SAC, TD3, PPO

from config_4dof import make_4dof_alternating_config, Alternating4DOFConfig
from env.alternating_lens_env import AlternatingLensEnv


def evaluate_episode(
    model,
    env: AlternatingLensEnv,
    deterministic: bool = True,
) -> Dict:
    """评估一个episode并记录详细轨迹。

    Returns:
        包含轨迹数据的字典：
            - states: (n_steps, 4) 对准状态 [dx, dy, rx, ry] (mm, mm, deg, deg)
            - actions: (n_steps, 4) 原始动作
            - masked_actions: (n_steps, 4) 屏蔽后的动作
            - rewards: (n_steps,) 奖励
            - qualities: (n_steps,) 质量指标
            - active_modes: (n_steps,) 激活模式 ('decenter' 或 'tilt')
            - success: bool 是否成功
            - total_reward: float 总奖励
    """
    obs, info = env.reset()

    trajectory = {
        'states': [info['state']],
        'actions': [],
        'masked_actions': [],
        'rewards': [],
        'qualities': [info['quality_metric']],
        'active_modes': [info['active_dofs']],
    }

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        trajectory['states'].append(info['state'])
        trajectory['actions'].append(info.get('original_action', action))
        trajectory['masked_actions'].append(info.get('masked_action', action))
        trajectory['rewards'].append(reward)
        trajectory['qualities'].append(info['quality_metric'])
        trajectory['active_modes'].append(info['active_dofs'])

        total_reward += reward

    # 转换为numpy数组
    trajectory['states'] = np.array(trajectory['states'])
    trajectory['actions'] = np.array(trajectory['actions'])
    trajectory['masked_actions'] = np.array(trajectory['masked_actions'])
    trajectory['rewards'] = np.array(trajectory['rewards'])
    trajectory['qualities'] = np.array(trajectory['qualities'])
    trajectory['active_modes'] = np.array(trajectory['active_modes'])
    trajectory['success'] = terminated
    trajectory['total_reward'] = total_reward

    return trajectory


def plot_trajectory(trajectory: Dict, save_path: str = None):
    """绘制4自由度交替运动轨迹。

    包含4个子图：
        1. 偏心轨迹 (dx, dy)
        2. 倾斜轨迹 (rx, ry)
        3. 各自由度调整历史
        4. MTF质量改进曲线
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    states = trajectory['states']
    qualities = trajectory['qualities']
    active_modes = trajectory['active_modes']
    n_steps = len(qualities)

    # 区分偏心步和倾斜步
    decenter_mask = active_modes == 'decenter'
    tilt_mask = active_modes == 'tilt'

    # === 子图1: 偏心轨迹 (dx, dy) ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(states[:, 0], states[:, 1], 'o-', color='gray', alpha=0.3, label='完整轨迹')
    ax1.scatter(states[decenter_mask, 0], states[decenter_mask, 1],
                c=np.arange(n_steps)[decenter_mask], cmap='Reds',
                s=80, marker='o', edgecolors='black', linewidth=1.5,
                label='偏心调整步', zorder=5)
    ax1.scatter(states[0, 0], states[0, 1], c='green', s=200, marker='*',
                edgecolors='black', linewidth=2, label='起点', zorder=10)
    ax1.scatter(states[-1, 0], states[-1, 1], c='red', s=200, marker='X',
                edgecolors='black', linewidth=2, label='终点', zorder=10)
    ax1.set_xlabel('dx (mm)', fontsize=12)
    ax1.set_ylabel('dy (mm)', fontsize=12)
    ax1.set_title('偏心轨迹 (dx, dy)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # === 子图2: 倾斜轨迹 (rx, ry) ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(states[:, 2], states[:, 3], 'o-', color='gray', alpha=0.3, label='完整轨迹')
    ax2.scatter(states[tilt_mask, 2], states[tilt_mask, 3],
                c=np.arange(n_steps)[tilt_mask], cmap='Blues',
                s=80, marker='s', edgecolors='black', linewidth=1.5,
                label='倾斜调整步', zorder=5)
    ax2.scatter(states[0, 2], states[0, 3], c='green', s=200, marker='*',
                edgecolors='black', linewidth=2, label='起点', zorder=10)
    ax2.scatter(states[-1, 2], states[-1, 3], c='red', s=200, marker='X',
                edgecolors='black', linewidth=2, label='终点', zorder=10)
    ax2.set_xlabel('rx (deg)', fontsize=12)
    ax2.set_ylabel('ry (deg)', fontsize=12)
    ax2.set_title('倾斜轨迹 (rx, ry)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # === 子图3: 各自由度调整历史 ===
    ax3 = fig.add_subplot(gs[1, :])
    steps = np.arange(n_steps)
    ax3.plot(steps, states[:, 0], 'o-', label='dx (mm)', color='C0', markersize=4)
    ax3.plot(steps, states[:, 1], 's-', label='dy (mm)', color='C1', markersize=4)
    ax3.plot(steps, states[:, 2], '^-', label='rx (deg)', color='C2', markersize=4)
    ax3.plot(steps, states[:, 3], 'v-', label='ry (deg)', color='C3', markersize=4)

    # 标注激活区域
    for i in range(n_steps - 1):
        if active_modes[i] == 'decenter':
            ax3.axvspan(i, i+1, alpha=0.1, color='red')
        elif active_modes[i] == 'tilt':
            ax3.axvspan(i, i+1, alpha=0.1, color='blue')

    ax3.set_xlabel('步数', fontsize=12)
    ax3.set_ylabel('对准状态', fontsize=12)
    ax3.set_title('各自由度调整历史（红色=偏心步，蓝色=倾斜步）', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', ncol=4)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # === 子图4: MTF质量改进曲线 ===
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(steps, qualities, 'o-', color='purple', linewidth=2, markersize=5)
    ax4.axhline(0.05, color='green', linestyle='--', linewidth=2, label='成功阈值 (0.05)')
    ax4.fill_between(steps, 0.05, qualities, where=(qualities >= 0.05),
                     alpha=0.3, color='green', label='成功区域')
    ax4.set_xlabel('步数', fontsize=12)
    ax4.set_ylabel('质量指标 (log MTF增益)', fontsize=12)
    ax4.set_title('MTF质量改进曲线', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    # 总标题
    success_str = "成功 ✓" if trajectory['success'] else "未成功 ✗"
    fig.suptitle(
        f'4自由度交替对准轨迹分析 | {success_str} | '
        f'总奖励: {trajectory["total_reward"]:.2f} | 步数: {n_steps-1}',
        fontsize=16, fontweight='bold'
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"轨迹图已保存: {save_path}")

    plt.show()


def compare_modes(
    model_alternating,
    model_simultaneous,
    lens_cfg,
    n_episodes: int = 20,
    save_path: str = None,
):
    """对比交替模式和同时模式的性能。"""
    # 创建环境
    env_alt = AlternatingLensEnv(
        cfg=lens_cfg,
        alternating_cfg=Alternating4DOFConfig(motion_mode='alternating')
    )
    env_sim = AlternatingLensEnv(
        cfg=lens_cfg,
        alternating_cfg=Alternating4DOFConfig(motion_mode='simultaneous')
    )

    results = {'alternating': [], 'simultaneous': []}

    print(f"\n评估交替模式 ({n_episodes} episodes)...")
    for i in range(n_episodes):
        traj = evaluate_episode(model_alternating, env_alt, deterministic=True)
        results['alternating'].append({
            'success': traj['success'],
            'steps': len(traj['qualities']) - 1,
            'final_quality': traj['qualities'][-1],
            'total_reward': traj['total_reward'],
        })
        if (i + 1) % 5 == 0:
            print(f"  完成 {i+1}/{n_episodes}")

    print(f"\n评估同时模式 ({n_episodes} episodes)...")
    for i in range(n_episodes):
        traj = evaluate_episode(model_simultaneous, env_sim, deterministic=True)
        results['simultaneous'].append({
            'success': traj['success'],
            'steps': len(traj['qualities']) - 1,
            'final_quality': traj['final_quality'],
            'total_reward': traj['total_reward'],
        })
        if (i + 1) % 5 == 0:
            print(f"  完成 {i+1}/{n_episodes}")

    # 统计对比
    print(f"\n{'='*60}")
    print("性能对比:")
    print(f"{'='*60}")

    for mode in ['alternating', 'simultaneous']:
        mode_name = "交替模式" if mode == 'alternating' else "同时模式"
        data = results[mode]
        success_rate = np.mean([d['success'] for d in data]) * 100
        avg_steps = np.mean([d['steps'] for d in data])
        avg_quality = np.mean([d['final_quality'] for d in data])
        avg_reward = np.mean([d['total_reward'] for d in data])

        print(f"\n{mode_name}:")
        print(f"  成功率        : {success_rate:.1f}%")
        print(f"  平均步数      : {avg_steps:.1f}")
        print(f"  平均最终质量  : {avg_quality:.4f}")
        print(f"  平均总奖励    : {avg_reward:.2f}")

    print(f"{'='*60}\n")

    env_alt.close()
    env_sim.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="分析4自由度交替对准")
    parser.add_argument(
        "--model", required=True,
        help="模型路径 (.zip)"
    )
    parser.add_argument(
        "--mode", default="alternating", choices=["alternating", "simultaneous"],
        help="运动模式"
    )
    parser.add_argument(
        "--algo", default="sac", choices=["sac", "td3", "ppo"],
        help="算法类型"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5,
        help="评估episode数"
    )
    parser.add_argument(
        "--output-dir", default="analysis_results",
        help="输出目录"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.model}")
    algo_cls = {"sac": SAC, "td3": TD3, "ppo": PPO}[args.algo]
    model = algo_cls.load(args.model)

    # 创建环境
    lens_cfg = make_4dof_alternating_config(fast_mode=False)  # 使用高精度模式
    alt_cfg = Alternating4DOFConfig(motion_mode=args.mode)
    env = AlternatingLensEnv(cfg=lens_cfg, alternating_cfg=alt_cfg)

    # 评估多个episodes
    print(f"\n开始评估 {args.n_episodes} episodes...")
    for i in range(args.n_episodes):
        print(f"\nEpisode {i+1}/{args.n_episodes}")
        trajectory = evaluate_episode(model, env, deterministic=True)

        print(f"  成功: {trajectory['success']}")
        print(f"  步数: {len(trajectory['qualities']) - 1}")
        print(f"  最终质量: {trajectory['qualities'][-1]:.4f}")
        print(f"  总奖励: {trajectory['total_reward']:.2f}")

        # 绘制轨迹
        save_path = os.path.join(
            args.output_dir,
            f"trajectory_{args.mode}_ep{i+1}.png"
        )
        plot_trajectory(trajectory, save_path=save_path)

    env.close()
    print(f"\n分析完成！结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
