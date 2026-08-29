"""测试4环境并行训练的实际速度"""
import time
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

def make_env(cfg):
    def _init():
        return LensAlignmentEnv(cfg=cfg)
    return _init

if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    cfg = make_lens_rl_config(fast_mode=True)
    print(f"\n配置: mtf_num_rays={cfg.mtf_num_rays}, 4个并行环境")

    # 创建4个并行环境
    vec_env = SubprocVecEnv([make_env(cfg) for _ in range(4)])

    # 预热
    vec_env.reset()
    for _ in range(3):
        actions = [vec_env.action_space.sample() for _ in range(4)]
        vec_env.step(actions)

    # 实际测试
    num_steps = 50  # 每个环境50步 = 总共200步
    start = time.time()

    for i in range(num_steps):
        actions = [vec_env.action_space.sample() for _ in range(4)]
        obs, rewards, dones, infos = vec_env.step(actions)

    elapsed = time.time() - start
    total_steps = num_steps * 4  # 4个环境
    steps_per_sec = total_steps / elapsed

    print(f"\n性能测试 (4环境并行):")
    print(f"  每环境步数: {num_steps}")
    print(f"  总步数: {total_steps}")
    print(f"  总时间: {elapsed:.2f}s")
    print(f"  并行速度: {steps_per_sec:.2f} steps/sec")
    print(f"  单步时间: {elapsed/num_steps*1000:.1f}ms")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU峰值显存: {peak_mem:.1f} MB")

    # 预估完成时间
    total_steps_target = 1_000_000
    hours = total_steps_target / steps_per_sec / 3600
    print(f"\n预估1M步完成时间: {hours:.1f}小时")

    vec_env.close()
