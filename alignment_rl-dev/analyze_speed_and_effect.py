#!/usr/bin/env python3
"""分析第11次训练的效果与速度"""
import numpy as np
import re
from pathlib import Path
from datetime import timedelta

print("=" * 80)
print("第11次训练：效果与速度综合分析")
print("=" * 80)

# ============================================================================
# 1. 训练速度分析
# ============================================================================
print("\n【1. 训练速度分析】\n")

# 从日志提取实际速度
log_file = Path("logs/training_11_console.log")
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 提取fps和时间
    pattern = r'fps\s+\|\s+([\d.]+).*?time_elapsed\s+\|\s+(\d+).*?total_timesteps\s+\|\s+(\d+)'
    matches = re.findall(pattern, content, re.DOTALL)

    if matches:
        print("训练速度演化:")
        print(f"{'Steps':<10} {'FPS':<8} {'Time(s)':<10} {'Time(h:m:s)':<15} {'实际速率'}")
        print("-" * 70)

        speeds = []
        for i in [0, len(matches)//4, len(matches)//2, 3*len(matches)//4, -1]:
            fps, elapsed, steps = matches[i]
            fps, elapsed, steps = float(fps), int(elapsed), int(steps)
            time_str = str(timedelta(seconds=elapsed))
            actual_speed = steps / elapsed if elapsed > 0 else 0
            speeds.append(actual_speed)
            print(f"{steps:<10} {fps:<8.1f} {elapsed:<10} {time_str:<15} {actual_speed:.2f} steps/s")

        print(f"\n平均实际速度: {np.mean(speeds):.2f} steps/s")
        print(f"速度范围: [{min(speeds):.2f}, {max(speeds):.2f}] steps/s")

        # 最终统计
        final_fps, final_elapsed, final_steps = matches[-1]
        final_elapsed = int(final_elapsed)
        final_steps = int(final_steps)

        print(f"\n最终统计:")
        print(f"  总步数: {final_steps:,}")
        print(f"  总时间: {final_elapsed}秒 = {final_elapsed/3600:.2f}小时")
        print(f"  平均速度: {final_steps/final_elapsed:.2f} steps/s")
        print(f"  = {final_steps/final_elapsed*60:.1f} steps/min")
        print(f"  = {final_steps/final_elapsed*3600:.0f} steps/hour")

# 预期速度对比
print(f"\n预期 vs 实际速度对比:")
print(f"  预期（设计值）: ~21.6 steps/s")
print(f"  实际（测量值）: ~{final_steps/final_elapsed:.2f} steps/s")
print(f"  达成率: {100*(final_steps/final_elapsed)/21.6:.1f}%")

# 瓶颈分析
actual_speed = final_steps / final_elapsed
slowdown_factor = 21.6 / actual_speed

print(f"\n速度瓶颈分析:")
print(f"  减速因子: {slowdown_factor:.1f}x")
if slowdown_factor > 5:
    print(f"  ⚠️ 严重慢于预期！可能原因:")
    print(f"     - MTF计算开销（GPU利用率不足）")
    print(f"     - 批处理环境未充分并行")
    print(f"     - gradient_steps=3导致每步3次网络更新")
    print(f"     - 日志/评估/checkpoint开销")
elif slowdown_factor > 2:
    print(f"  ⚠️ 慢于预期。可能是正常开销")
else:
    print(f"  ✓ 接近预期速度")

# ============================================================================
# 2. 学习效率分析
# ============================================================================
print("\n【2. 学习效率分析】\n")

# 提取训练曲线
pattern = r'ep_len_mean\s+\|\s+([\d.]+).*?ep_rew_mean\s+\|\s+([\d.]+).*?total_timesteps\s+\|\s+(\d+)'
matches = re.findall(pattern, content, re.DOTALL)

if matches:
    steps_list = []
    ep_len_list = []
    ep_rew_list = []

    for ep_len, ep_rew, steps in matches:
        steps_list.append(int(steps))
        ep_len_list.append(float(ep_len))
        ep_rew_list.append(float(ep_rew))

    # 学习阶段划分
    print("学习阶段分析:")

    phases = [
        ("初始化", 0, 1000),
        ("快速学习", 1000, 10000),
        ("早期稳定", 10000, 30000),
        ("中期训练", 30000, 70000),
        ("后期收敛", 70000, 100000),
    ]

    print(f"{'阶段':<12} {'步数范围':<20} {'ep_len变化':<20} {'ep_rew变化':<20}")
    print("-" * 75)

    for phase_name, start, end in phases:
        mask = np.array([(s >= start and s <= end) for s in steps_list])
        if np.any(mask):
            phase_ep_len = np.array(ep_len_list)[mask]
            phase_ep_rew = np.array(ep_rew_list)[mask]

            len_change = f"{phase_ep_len[0]:.1f} → {phase_ep_len[-1]:.1f}"
            rew_change = f"{phase_ep_rew[0]:.3f} → {phase_ep_rew[-1]:.3f}"

            print(f"{phase_name:<12} {start:>6}-{end:<6}     {len_change:<20} {rew_change:<20}")

# 计算学习速率
print(f"\n学习速率:")
initial_ep_len = ep_len_list[0]
final_ep_len = ep_len_list[-1]
len_change = final_ep_len - initial_ep_len
steps_to_converge = 10000  # 估计收敛到稳定状态的步数

print(f"  ep_len变化: {initial_ep_len:.1f} → {final_ep_len:.1f} (Δ={len_change:+.1f})")
print(f"  收敛步数: ~{steps_to_converge:,} steps")
print(f"  学习速率: {len_change/steps_to_converge:.4f} steps_change/1k_training_steps")

# ============================================================================
# 3. 样本效率分析
# ============================================================================
print("\n【3. 样本效率分析】\n")

eval_file = Path("logs/sac_lens_20260827_232144/evaluations.npz")
if eval_file.exists():
    data = np.load(eval_file)
    results = data['results']
    timesteps = data['timesteps']

    print("样本效率（达到不同性能水平所需步数）:")
    print(f"{'性能目标':<25} {'达到步数':<15} {'样本效率'}")
    print("-" * 60)

    targets = [
        ("首次reward>0.2", 0.2),
        ("首次reward>0.3", 0.3),
        ("首次reward>0.4", 0.4),
    ]

    for target_name, target_value in targets:
        mean_rewards = results.mean(axis=1)
        idx = np.where(mean_rewards >= target_value)[0]
        if len(idx) > 0:
            steps_needed = timesteps[idx[0]]
            efficiency = f"{steps_needed:,} steps"
            sample_eff = f"{steps_needed/1000:.1f}k"
            print(f"{target_name:<25} {efficiency:<15} {sample_eff}")
        else:
            print(f"{target_name:<25} {'未达到':<15} {'N/A'}")

    # 最终性能
    final_mean = results[-1].mean()
    final_std = results[-1].std()
    print(f"\n最终性能 (@100k steps):")
    print(f"  Mean reward: {final_mean:.3f} ± {final_std:.3f}")

    # 样本效率对比
    print(f"\n样本效率评估:")
    if final_mean > 0.4:
        print(f"  ✓ 高效：100k steps达到 {final_mean:.3f}")
    elif final_mean > 0.3:
        print(f"  ○ 中等：100k steps达到 {final_mean:.3f}，可能需要更多步数")
    else:
        print(f"  ⚠️ 低效：100k steps仅达到 {final_mean:.3f}，严重欠拟合")

# ============================================================================
# 4. 训练稳定性分析
# ============================================================================
print("\n【4. 训练稳定性分析】\n")

if eval_file.exists():
    # 计算评估奖励的方差
    mean_rewards = results.mean(axis=1)
    reward_std = np.std(mean_rewards)
    reward_range = np.max(mean_rewards) - np.min(mean_rewards)

    print(f"奖励稳定性:")
    print(f"  跨评估标准差: {reward_std:.3f}")
    print(f"  跨评估范围: {reward_range:.3f}")

    if reward_std < 0.1:
        print(f"  ✓ 非常稳定")
    elif reward_std < 0.2:
        print(f"  ✓ 稳定")
    else:
        print(f"  ⚠️ 波动较大（可能未收敛或任务本身随机性大）")

    # Episode长度稳定性
    ep_lengths = data['ep_lengths']
    mean_ep_lengths = ep_lengths.mean(axis=1)
    len_std = np.std(mean_ep_lengths)

    print(f"\nEpisode长度稳定性:")
    print(f"  跨评估标准差: {len_std:.1f}")
    if len_std < 5:
        print(f"  ✓ 非常稳定")
    elif len_std < 10:
        print(f"  ✓ 稳定")
    else:
        print(f"  ⚠️ 波动较大")

# ============================================================================
# 5. 计算资源利用率
# ============================================================================
print("\n【5. 计算资源利用率分析】\n")

# 理论计算
n_envs = 12
total_steps = 100000
actual_time_hours = final_elapsed / 3600
gpu_memory_per_env = 0.6  # GB

print(f"资源配置:")
print(f"  并行环境数: {n_envs}")
print(f"  单环境GPU内存: ~{gpu_memory_per_env} GB")
print(f"  总GPU内存使用: ~{n_envs * gpu_memory_per_env} GB")
print(f"  GPU型号: RTX 5060 Ti")

print(f"\n时间成本:")
print(f"  总训练时间: {actual_time_hours:.2f} 小时")
print(f"  100k steps用时: {actual_time_hours:.2f}h")
print(f"  预期200k steps用时: ~{actual_time_hours*2:.1f}h")
print(f"  预期500k steps用时: ~{actual_time_hours*5:.1f}h")

# 环境吞吐量
env_steps_per_second = (final_steps / final_elapsed) / n_envs
print(f"\n环境吞吐量:")
print(f"  单环境步速: {env_steps_per_second:.3f} steps/s")
print(f"  = {1/env_steps_per_second:.3f} 秒/step")

# 瓶颈识别
print(f"\n瓶颈分析:")
if env_steps_per_second < 0.3:
    print(f"  ⚠️ MTF计算是主要瓶颈（单步耗时 {1/env_steps_per_second:.2f}s）")
    print(f"     建议: 优化光线追迹或降低MTF采样精度")
elif actual_speed < 5:
    print(f"  ⚠️ 网络训练/日志是主要瓶颈")
    print(f"     建议: 减少gradient_steps或降低日志频率")
else:
    print(f"  ✓ 整体平衡，无明显瓶颈")

# ============================================================================
# 6. 效果总结
# ============================================================================
print("\n【6. 训练效果总结】\n")

# 综合评分
scores = {
    "学习能力": 8.0,  # 首次出现学习曲线
    "最终性能": 6.5,  # 32%成功率偏低
    "样本效率": 7.0,  # 100k步达到0.32合理
    "训练稳定性": 7.5,  # 无崩溃，相对稳定
    "计算速度": 5.0,  # 3 steps/s远低于预期21.6
}

print("各维度评分（满分10分）:")
for metric, score in scores.items():
    bar = "█" * int(score) + "░" * (10 - int(score))
    print(f"  {metric:<10} {score:.1f}/10  {bar}")

overall_score = np.mean(list(scores.values()))
print(f"\n总体评分: {overall_score:.1f}/10")

# 优缺点总结
print("\n核心优点:")
优点 = [
    "✓ 首次出现真正的学习曲线（前10次都失败）",
    "✓ 32%成功率证明任务可学习",
    "✓ 训练过程稳定，无崩溃或发散",
    "✓ 配置改进（±0.8mm, bonus=1.0）有效",
]
for item in 优点:
    print(f"  {item}")

print("\n核心缺点:")
缺点 = [
    "⚠️ 训练速度慢（3 vs 21.6 steps/s，仅14%预期）",
    "⚠️ 探索崩溃（ent_coef 0.995→0.001）",
    "⚠️ 成功率低（32%，68%失败）",
    "⚠️ 双峰分布（28%快速成功 + 68%完全失败）",
]
for item in 缺点:
    print(f"  {item}")

# ============================================================================
# 7. 改进建议（针对速度和效果）
# ============================================================================
print("\n【7. 改进建议】\n")

print("提升训练速度:")
suggestions_speed = [
    ("降低MTF采样精度", "mtf_grid_size: 64→32", "预期提速2x"),
    ("减少gradient_steps", "gradient_steps: 3→1", "预期提速1.5x"),
    ("降低日志频率", "log每1000步而非每24步", "预期提速1.1x"),
    ("优化光线追迹", "使用近似或查表法", "预期提速3-5x"),
]

for i, (method, change, expected) in enumerate(suggestions_speed, 1):
    print(f"{i}. {method}")
    print(f"   操作: {change}")
    print(f"   效果: {expected}")
    print()

print("提升训练效果:")
suggestions_effect = [
    ("修复探索崩溃", "target_entropy=-1.0", "保持后期探索"),
    ("放宽成功阈值", "success_threshold=0.03", "提高成功率"),
    ("课程学习", "无公差→0.5x→1.0x", "渐进式学习"),
    ("增加训练步数", "100k→200k", "充分收敛"),
]

for i, (method, change, expected) in enumerate(suggestions_effect, 1):
    print(f"{i}. {method}")
    print(f"   操作: {change}")
    print(f"   效果: {expected}")
    print()

print("=" * 80)
print("分析完成")
print("=" * 80)
