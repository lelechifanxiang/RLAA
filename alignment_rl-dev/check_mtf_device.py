"""检查MTF计算使用的设备（CPU vs GPU）"""
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# 创建环境
cfg = make_lens_rl_config(fast_mode=True)
env = LensAlignmentEnv(cfg=cfg)

# 检查内部设备设置
print("环境内部设备信息:")
print(f"  env._mgr._device: {env._mgr._device}")

# 检查optics_core系统
system = env._mgr._core_system
if hasattr(system, 'device'):
    print(f"  system.device: {system.device}")
else:
    print(f"  system.device: (属性不存在)")

# 进行一次MTF计算，观察GPU使用
print("\n执行一次MTF计算...")
env.reset()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1024**2

action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

if torch.cuda.is_available():
    mem_after = torch.cuda.memory_allocated() / 1024**2
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\nGPU显存使用:")
    print(f"  计算前: {mem_before:.1f} MB")
    print(f"  计算后: {mem_after:.1f} MB")
    print(f"  峰值: {mem_peak:.1f} MB")

    if mem_peak > 100:
        print("\n[OK] MTF calculation is using GPU (memory usage > 100MB)")
    else:
        print("\n[WARN] MTF calculation may not be using GPU (low memory usage)")
else:
    print("\n[WARN] CUDA not available, using CPU")

# 检查中间张量设备
print("\n检查计算过程中的张量设备...")
print("（需要在optics_core代码中添加调试信息）")
