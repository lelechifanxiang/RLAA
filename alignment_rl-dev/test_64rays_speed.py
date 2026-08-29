"""测试64条光线的训练速度。"""
import time
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 测试64条光线
cfg = make_lens_rl_config(fast_mode=True)
print(f"\n配置: mtf_num_rays={cfg.mtf_num_rays}")

env = LensAlignmentEnv(cfg=cfg)

# 清空GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# 预热
env.reset()
for _ in range(3):
    env.step(env.action_space.sample())

# 实际测试
num_steps = 20
start = time.time()

for i in range(num_steps):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        env.reset()

elapsed = time.time() - start
steps_per_sec = num_steps / elapsed

print(f"\n性能测试 (64条光线):")
print(f"  总步数: {num_steps}")
print(f"  总时间: {elapsed:.2f}s")
print(f"  速度: {steps_per_sec:.2f} steps/sec")
print(f"  单步时间: {elapsed/num_steps*1000:.1f}ms")

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  GPU峰值显存: {peak_mem:.1f} MB")

# 预估完成时间
total_steps = 1_000_000
hours = total_steps / steps_per_sec / 3600
print(f"\n预估1M步完成时间: {hours:.1f}小时")
