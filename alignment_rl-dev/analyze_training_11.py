#!/usr/bin/env python3
"""详细分析第11次训练结果"""
import numpy as np
import re
from pathlib import Path

print("=" * 70)
print("第11次训练完整结果分析")
print("=" * 70)

# 1. 训练过程指标
print("\n【1. 训练过程指标】\n")

log_file = Path("logs/training_11_console.log")
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 提取训练指标
    pattern = r'ep_len_mean\s+\|\s+([\d.]+).*?ep_rew_mean\s+\|\s+([\d.]+).*?total_timesteps\s+\|\s+(\d+).*?ent_coef\s+\|\s+([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)

    if matches:
        print(f"训练记录数: {len(matches)}")
        print(f"\n关键里程碑:")

        milestones = [0, len(matches)//4, len(matches)//2, 3*len(matches)//4, -1]
        for idx in milestones:
            ep_len, ep_rew, steps, ent = matches[idx]
            print(f"  {int(steps):>6} steps: ep_len={float(ep_len):>5.1f}, ep_rew={float(ep_rew):>6.3f}, ent_coef={float(ent):.4f}")

# 2. 评估结果
print("\n【2. 评估结果（每10k steps）】\n")

eval_file = Path("logs/sac_lens_20260827_232144/evaluations.npz")
if eval_file.exists():
    data = np.load(eval_file)
    results = data['results']
    ep_lengths = data['ep_lengths']
    timesteps = data['timesteps']

    print(f"{'Steps':<10} {'Reward':<20} {'Ep Length':<20}")
    print("-" * 50)
    for i, step in enumerate(timesteps):
        rew_mean, rew_std = results[i].mean(), results[i].std()
        len_mean, len_std = ep_lengths[i].mean(), ep_lengths[i].std()
        print(f"{step:<10} {rew_mean:>5.2f} ± {rew_std:<4.2f}         {len_mean:>5.1f} ± {len_std:<4.1f}")

# 3. 最终评估详细分析
print("\n【3. 最终评估详细分析（99960 steps, 50 episodes）】\n")

if eval_file.exists():
    final_rewards = results[-1]
    final_lengths = ep_lengths[-1]

    print(f"奖励统计:")
    print(f"  均值: {final_rewards.mean():.3f}")
    print(f"  标准差: {final_rewards.std():.3f}")
    print(f"  中位数: {np.median(final_rewards):.3f}")
    print(f"  范围: [{final_rewards.min():.3f}, {final_rewards.max():.3f}]")

    # 奖励分布
    bins = [final_rewards.min(), -0.1, 0.0, 0.5, 1.0, final_rewards.max()]
    hist, _ = np.histogram(final_rewards, bins=bins)
    print(f"\n奖励分布:")
    print(f"  < -0.1 (很差):       {hist[0]:>2d} ({100*hist[0]/50:.0f}%)")
    print(f"  -0.1 ~ 0.0 (差):     {hist[1]:>2d} ({100*hist[1]/50:.0f}%)")
    print(f"  0.0 ~ 0.5 (中等):    {hist[2]:>2d} ({100*hist[2]/50:.0f}%)")
    print(f"  0.5 ~ 1.0 (好):      {hist[3]:>2d} ({100*hist[3]/50:.0f}%)")
    print(f"  > 1.0 (优秀+bonus):  {hist[4]:>2d} ({100*hist[4]/50:.0f}%)")

    print(f"\nEpisode长度统计:")
    print(f"  均值: {final_lengths.mean():.1f}")
    print(f"  标准差: {final_lengths.std():.1f}")
    print(f"  中位数: {np.median(final_lengths):.0f}")
    print(f"  范围: [{final_lengths.min():.0f}, {final_lengths.max():.0f}]")

    # Episode长度分布
    print(f"\nEpisode长度分布:")
    print(f"  1-10步 (立即成功):    {np.sum(final_lengths <= 10):>2d} ({100*np.sum(final_lengths <= 10)/50:.0f}%)")
    print(f"  11-20步 (快速成功):   {np.sum((final_lengths > 10) & (final_lengths <= 20)):>2d} ({100*np.sum((final_lengths > 10) & (final_lengths <= 20))/50:.0f}%)")
    print(f"  21-35步 (正常对准):   {np.sum((final_lengths > 20) & (final_lengths <= 35)):>2d} ({100*np.sum((final_lengths > 20) & (final_lengths <= 35))/50:.0f}%)")
    print(f"  36-45步 (困难对准):   {np.sum((final_lengths > 35) & (final_lengths <= 45)):>2d} ({100*np.sum((final_lengths > 35) & (final_lengths <= 45))/50:.0f}%)")
    print(f"  46-50步 (失败/超时):  {np.sum(final_lengths > 45):>2d} ({100*np.sum(final_lengths > 45)/50:.0f}%)")

    # 成功率估算
    print(f"\n成功率估算:")
    # 假设 reward > 0.5 表示成功（包含bonus=1.0）
    n_success_by_reward = np.sum(final_rewards > 0.5)
    # 或假设 ep_len <= 35 且 reward > 0 表示成功
    n_success_combined = np.sum((final_lengths <= 35) & (final_rewards > 0))

    print(f"  按奖励(>0.5):         {n_success_by_reward}/50 ({100*n_success_by_reward/50:.0f}%)")
    print(f"  按长度+奖励综合:      {n_success_combined}/50 ({100*n_success_combined/50:.0f}%)")

# 4. 与前次训练对比
print("\n【4. 与前次训练对比】\n")

comparison = [
    ("第5次 (nominal)", 38, 1.53, "✓ 无公差成功"),
    ("第10次 (q005)", 39, 1.42, "△ 有公差部分成功"),
    ("第11次 (本次)", 35, 0.316, "✓ 有公差显著改进"),
]

print(f"{'训练批次':<20} {'平均长度':<12} {'平均奖励':<12} {'评价'}")
print("-" * 60)
for name, length, reward, status in comparison:
    print(f"{name:<20} {length:<12.1f} {reward:<12.3f} {status}")

# 5. 问题与改进建议
print("\n【5. 发现的问题】\n")

problems = [
    ("熵系数崩溃", "0.995 → 0.00106", "后期探索不足，可能陷入局部最优"),
    ("成功率偏低", "~30-40%", "大部分episode失败或超时"),
    ("Episode过长", "35步 (中位数)", "策略效率不够高"),
    ("奖励波动大", "std=0.491", "不同公差下性能差异大"),
]

for problem, value, description in problems:
    print(f"⚠️  {problem} ({value})")
    print(f"    → {description}")
    print()

print("\n【6. 改进建议（第12次训练）】\n")

suggestions = [
    ("修复熵崩溃", "target_entropy=-1.0 (增大)", "保持探索到训练后期"),
    ("增加成功奖励", "success_bonus=1.5-2.0", "更强动机达到success_threshold"),
    ("扩展观测空间", "添加当前状态(dx,dy)", "帮助策略理解位置信息"),
    ("课程学习", "先训练nominal再加公差", "逐步增加任务难度"),
]

for i, (method, change, reason) in enumerate(suggestions, 1):
    print(f"{i}. {method}: {change}")
    print(f"   理由: {reason}")
    print()

print("=" * 70)
print("分析完成")
print("=" * 70)
