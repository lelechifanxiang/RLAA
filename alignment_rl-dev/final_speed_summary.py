"""最终性能总结：GPU加速后的训练速度分析"""
import time
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

print("="*60)
print("GPU加速性能总结")
print("="*60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}")
print()

# 测试32光线配置（训练用）
cfg = make_lens_rl_config(fast_mode=True)
print(f"配置: mtf_num_rays={cfg.mtf_num_rays} (训练模式)")

env = LensAlignmentEnv(cfg=cfg)

# 预热
env.reset()
for _ in range(5):
    env.step(env.action_space.sample())

# 测试
num_steps = 30
start = time.time()
for i in range(num_steps):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    if done or truncated:
        env.reset()

elapsed = time.time() - start
steps_per_sec = num_steps / elapsed

print(f"\n单环境性能:")
print(f"  步数: {num_steps}")
print(f"  时间: {elapsed:.2f}s")
print(f"  速度: {steps_per_sec:.2f} steps/sec")
print(f"  单步: {elapsed/num_steps*1000:.1f}ms")

if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  GPU峰值显存: {peak_mem:.1f} MB")

print("\n" + "="*60)
print("训练速度估算")
print("="*60)

# SAC/TD3使用4个环境
n_envs_sac = 4
total_speed_sac = steps_per_sec * n_envs_sac
hours_sac = 1_000_000 / total_speed_sac / 3600

print(f"\nSAC/TD3 (4个并行环境):")
print(f"  单环境: {steps_per_sec:.2f} steps/sec")
print(f"  理论总速度: {total_speed_sac:.2f} steps/sec")
print(f"  1M步需要: {hours_sac:.1f} 小时")
print(f"  备注: 实际可能因进程间通信略慢")

# PPO使用8个环境
n_envs_ppo = 8
total_speed_ppo = steps_per_sec * n_envs_ppo
hours_ppo = 1_000_000 / total_speed_ppo / 3600

print(f"\nPPO (8个并行环境):")
print(f"  单环境: {steps_per_sec:.2f} steps/sec")
print(f"  理论总速度: {total_speed_ppo:.2f} steps/sec")
print(f"  1M步需要: {hours_ppo:.1f} 小时")
print(f"  备注: PPO的n_steps=2048意味着每2048步更新一次")

print("\n" + "="*60)
print("结论")
print("="*60)
print(f"✓ GPU已成功启用: PyTorch {torch.__version__}")
print(f"✓ 训练配置: 32光线（速度优先）")
print(f"✓ 单环境速度: ~{steps_per_sec:.1f} steps/sec")
print(f"✓ 建议使用SAC或TD3进行训练")
print(f"✓ 预计训练时间: {hours_sac:.0f}-{hours_ppo:.0f}小时（1M步）")
print("="*60)
