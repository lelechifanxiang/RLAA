"""
环境性能基准测试脚本

对比不同配置下的环境执行速度：
- 单环境 vs 多环境
- 不同光线数（32 vs 128）
- CPU vs GPU
"""
import time
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from env.lens_env import LensAlignmentEnv
from config import make_lens_rl_config


def benchmark_env(n_envs: int, num_rays: int, n_steps: int = 100):
    """测试环境性能

    Args:
        n_envs: 并行环境数量
        num_rays: MTF计算光线数
        n_steps: 测试步数
    """
    print(f"\n{'='*60}")
    print(f"配置: {n_envs}个环境, {num_rays}条光线")
    print(f"GPU: {torch.cuda.is_available()}")
    print(f"{'='*60}")

    # 创建环境
    cfg = make_lens_rl_config(fast_mode=(num_rays == 32))
    cfg.mtf_num_rays = num_rays

    def make_env():
        return LensAlignmentEnv(cfg=cfg)

    if n_envs == 1:
        env = make_env()
    else:
        env = DummyVecEnv([make_env for _ in range(n_envs)])

    # 预热
    obs, info = env.reset()
    for _ in range(5):
        action = env.action_space.sample() if n_envs == 1 else np.array([env.action_space.sample() for _ in range(n_envs)])
        obs, reward, terminated, truncated, info = env.step(action)

    # 计时
    start_time = time.time()
    step_count = 0

    obs, info = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample() if n_envs == 1 else np.array([env.action_space.sample() for _ in range(n_envs)])
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += n_envs

        if n_envs == 1:
            if terminated or truncated:
                obs, info = env.reset()
        else:
            # 向量化环境自动重置
            pass

    elapsed = time.time() - start_time
    steps_per_sec = step_count / elapsed
    ms_per_step = 1000 * elapsed / step_count

    print(f"总步数: {step_count}")
    print(f"用时: {elapsed:.2f}秒")
    print(f"速度: {steps_per_sec:.2f} steps/sec")
    print(f"每步: {ms_per_step:.1f} ms")

    env.close()

    return steps_per_sec


if __name__ == "__main__":
    results = {}

    # 测试不同配置
    configs = [
        ("单环境-128光线", 1, 128),
        ("单环境-32光线", 1, 32),
        ("4环境-32光线", 4, 32),
        ("8环境-32光线", 8, 32),
    ]

    for name, n_envs, num_rays in configs:
        results[name] = benchmark_env(n_envs, num_rays, n_steps=50)

    # 汇总对比
    print(f"\n{'='*60}")
    print("性能对比汇总")
    print(f"{'='*60}")
    baseline = results["单环境-128光线"]
    for name, speed in results.items():
        speedup = speed / baseline
        print(f"{name:20s}: {speed:6.2f} steps/sec  (加速 {speedup:.1f}x)")
    print(f"{'='*60}\n")
