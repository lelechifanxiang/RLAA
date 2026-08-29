#!/usr/bin/env python3
"""每5分钟自动检查训练进度并更新进度文件"""
import time
import re
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("logs/training_11_console.log")
PROGRESS_PATH = Path("TRAINING_11_PROGRESS.md")

def get_latest_metrics(content):
    """提取最新的训练指标"""
    train_pattern = r'ep_len_mean\s+\|\s+([\d.]+).*?ep_rew_mean\s+\|\s+([\d.]+).*?total_timesteps\s+\|\s+(\d+).*?ent_coef\s+\|\s+([\d.]+)'
    matches = re.findall(train_pattern, content, re.DOTALL)

    if matches:
        ep_len, ep_rew, steps, ent_coef = matches[-1]
        return {
            'steps': int(steps),
            'ep_len': float(ep_len),
            'ep_rew': float(ep_rew),
            'ent_coef': float(ent_coef),
        }
    return None

def update_progress():
    """更新进度文件"""
    if not LOG_PATH.exists():
        print(f"日志文件不存在: {LOG_PATH}")
        return

    with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    metrics = get_latest_metrics(content)
    if not metrics:
        print("未找到训练指标")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    progress_line = f"| {metrics['steps']:<5} | {metrics['ep_len']:<11.2f} | {metrics['ep_rew']:<11.3f} | {metrics['ent_coef']:<8.3f} | {timestamp} |"

    print(f"\n[{timestamp}] Steps: {metrics['steps']}, ep_len: {metrics['ep_len']:.2f}, ep_rew: {metrics['ep_rew']:.3f}, ent_coef: {metrics['ent_coef']:.3f}")

    # 追加到进度文件
    with open("training_progress_log.txt", 'a', encoding='utf-8') as f:
        f.write(f"{timestamp} | {metrics['steps']} | {metrics['ep_len']:.2f} | {metrics['ep_rew']:.3f} | {metrics['ent_coef']:.3f}\n")

if __name__ == "__main__":
    print("开始监控训练进度（每5分钟更新一次）...")
    while True:
        try:
            update_progress()
        except Exception as e:
            print(f"错误: {e}")

        time.sleep(300)  # 5分钟
