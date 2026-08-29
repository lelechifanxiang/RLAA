"""比较不同光线数配置的速度和GPU利用率"""
import time
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

configs = [
    (32, "32 rays (原配置)"),
    (64, "64 rays (新配置)"),
]

for num_rays, label in configs:
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    cfg = make_lens_rl_config(fast_mode=True)
    cfg.mtf_num_rays = num_rays

    env = LensAlignmentEnv(cfg=cfg)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 预热
    env.reset()
    for _ in range(2):
        env.step(env.action_space.sample())

    # 测试
    num_steps = 10
    start = time.time()

    for i in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            env.reset()

    elapsed = time.time() - start
    steps_per_sec = num_steps / elapsed

    print(f"\n性能:")
    print(f"  步数: {num_steps}")
    print(f"  总时间: {elapsed:.2f}s")
    print(f"  速度: {steps_per_sec:.2f} steps/sec")
    print(f"  单步: {elapsed/num_steps*1000:.1f}ms")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU峰值: {peak_mem:.1f} MB")

    # 预估4环境并行
    print(f"\n预估4环境并行:")
    print(f"  总速度: {steps_per_sec * 4:.2f} steps/sec")
    print(f"  1M步需要: {1_000_000 / (steps_per_sec * 4) / 3600:.1f} 小时")

print(f"\n{'='*60}")
print("结论:")
print("  如果64光线更慢,说明chunking优化引入了overhead")
print("  需要回退chunking或调整chunk_size")
print(f"{'='*60}")
