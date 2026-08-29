#!/usr/bin/env python3
"""监控训练进度并提取关键指标"""
import re
import sys
from pathlib import Path

def parse_training_log(log_path):
    """解析训练日志，提取关键指标"""
    if not Path(log_path).exists():
        print(f"日志文件不存在: {log_path}")
        return

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 提取所有训练记录块
    pattern = r'-+\n\| rollout/.*?\n\|    ep_len_mean\s+\|\s+([\d.]+).*?\n\|    ep_rew_mean\s+\|\s+([\d.]+).*?total_timesteps\s+\|\s+(\d+)'
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        print("未找到训练记录")
        return

    print("\n=== 训练进度摘要 ===\n")
    print(f"{'Steps':<10} {'ep_len_mean':<15} {'ep_rew_mean':<15}")
    print("-" * 40)

    for match in matches[-10:]:  # 显示最近10条记录
        ep_len, ep_rew, steps = match
        print(f"{steps:<10} {ep_len:<15} {ep_rew:<15}")

    # 提取最新的完整记录（包含train指标）
    train_pattern = r'ep_len_mean\s+\|\s+([\d.]+).*?ep_rew_mean\s+\|\s+([\d.]+).*?total_timesteps\s+\|\s+(\d+).*?ent_coef\s+\|\s+([\d.]+).*?n_updates\s+\|\s+(\d+)'
    train_matches = re.findall(train_pattern, content, re.DOTALL)

    if train_matches:
        print("\n=== 最新训练指标 ===")
        ep_len, ep_rew, steps, ent_coef, n_updates = train_matches[-1]
        print(f"  步数: {steps}")
        print(f"  Episode平均长度: {ep_len}")
        print(f"  Episode平均奖励: {ep_rew}")
        print(f"  熵系数: {ent_coef}")
        print(f"  网络更新次数: {n_updates}")

if __name__ == "__main__":
    log_path = "logs/training_11_console.log"
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    parse_training_log(log_path)
