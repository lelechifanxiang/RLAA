"""测试MTF计算是否真正在GPU上执行"""
import torch
import time
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

# 临时patch _huygens_integral来监控设备
original_huygens = None

def patched_huygens_integral(*args, **kwargs):
    """监控_huygens_integral的设备使用"""
    image_points = kwargs.get('image_points')
    if image_points is not None:
        print(f'  [Huygens] image_points device: {image_points.device}')
        print(f'  [Huygens] image_points shape: {image_points.shape}')
    return original_huygens(*args, **kwargs)

# Patch函数
import optics_core.huygens_psf as huygens_module
original_huygens = huygens_module._huygens_integral
huygens_module._huygens_integral = patched_huygens_integral

# 创建环境
cfg = make_lens_rl_config(fast_mode=True)
print(f"配置: {cfg.mtf_num_rays}条光线\n")

env = LensAlignmentEnv(cfg=cfg)
env.reset()

# 执行一步
print("执行step...")
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1024**2

start = time.time()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
elapsed = time.time() - start

mem_after = torch.cuda.memory_allocated() / 1024**2
mem_peak = torch.cuda.max_memory_allocated() / 1024**2

print(f"\n结果:")
print(f"  时间: {elapsed*1000:.1f}ms")
print(f"  GPU显存: {mem_before:.1f}MB -> {mem_after:.1f}MB (峰值: {mem_peak:.1f}MB)")
print(f"  速度: {1.0/elapsed:.2f} steps/sec")
