#!/usr/bin/env python3
"""分析训练11的评估结果"""
import numpy as np
from pathlib import Path

eval_file = Path("logs/sac_lens_20260827_232144/evaluations.npz")

if eval_file.exists():
    data = np.load(eval_file)

    print("=== 第11次训练评估结果 ===\n")

    # 显示所有可用的键
    print("可用数据:")
    for key in data.files:
        print(f"  - {key}: shape={data[key].shape}")
    print()

    # 提取关键指标
    if 'results' in data.files:
        results = data['results']  # shape: (n_evals, n_episodes)
        timesteps = data['timesteps']

        print(f"评估次数: {len(timesteps)}")
        print(f"每次评估episodes: {results.shape[1]}")
        print()

        print("评估历史:")
        print(f"{'Steps':<10} {'Mean Reward':<15} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)

        for i, step in enumerate(timesteps):
            rewards = results[i]
            print(f"{step:<10} {rewards.mean():<15.3f} {rewards.std():<10.3f} {rewards.min():<10.3f} {rewards.max():<10.3f}")

        print()
        print(f"最终评估 (步数={timesteps[-1]}):")
        final_rewards = results[-1]
        print(f"  平均奖励: {final_rewards.mean():.3f} ± {final_rewards.std():.3f}")
        print(f"  奖励范围: [{final_rewards.min():.3f}, {final_rewards.max():.3f}]")

    if 'ep_lengths' in data.files:
        ep_lengths = data['ep_lengths']
        print(f"\n最终评估 episode长度:")
        final_lengths = ep_lengths[-1]
        print(f"  平均长度: {final_lengths.mean():.1f} ± {final_lengths.std():.1f}")
        print(f"  长度范围: [{final_lengths.min():.0f}, {final_lengths.max():.0f}]")

        # 估算成功率（假设短episode=成功）
        success_threshold_steps = 20  # 假设<20步是快速成功
        failure_threshold_steps = 45  # 假设>45步是失败/超时

        n_success = np.sum(final_lengths < success_threshold_steps)
        n_failure = np.sum(final_lengths > failure_threshold_steps)
        n_moderate = len(final_lengths) - n_success - n_failure

        print(f"\nEpisode分布:")
        print(f"  快速成功(<{success_threshold_steps}步): {n_success}/{len(final_lengths)} ({100*n_success/len(final_lengths):.1f}%)")
        print(f"  中等长度({success_threshold_steps}-{failure_threshold_steps}步): {n_moderate}/{len(final_lengths)} ({100*n_moderate/len(final_lengths):.1f}%)")
        print(f"  失败/超时(>{failure_threshold_steps}步): {n_failure}/{len(final_lengths)} ({100*n_failure/len(final_lengths):.1f}%)")

else:
    print(f"评估文件不存在: {eval_file}")
