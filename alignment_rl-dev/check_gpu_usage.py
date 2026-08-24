"""检查MTF计算是否使用GPU"""
import time
import torch
import numpy as np
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

print("测试MTF计算GPU使用情况...")
print("如果使用GPU，应该能看到GPU内存增加\n")

# 重置环境
obs, info = env.reset(seed=42)

# 多次step，观察GPU使用
for i in range(5):
    action = env.action_space.sample()

    # 清空GPU缓存并记录初始内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**2

    start = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    elapsed = time.time() - start

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024**2
        print(f"Step {i+1}: {elapsed:.3f}s, GPU内存: {mem_before:.1f}MB -> {mem_after:.1f}MB")
    else:
        print(f"Step {i+1}: {elapsed:.3f}s (CPU)")

print("\n检查optics_core的设备设置...")
print(f"env._mgr._device: {env._mgr._device if hasattr(env._mgr, '_device') else 'N/A'}")
