"""测试不同chunk_size对速度和显存的影响"""
import torch
import time
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

# Patch _huygens_integral来改变chunk_size
import optics_core.huygens_psf as huygens_module
original_huygens = huygens_module._huygens_integral

def make_chunked_huygens(chunk_size):
    def chunked(*args, **kwargs):
        kwargs['ray_chunk_size'] = chunk_size
        return original_huygens(*args, **kwargs)
    return chunked

# 测试不同chunk_size
chunk_sizes = [128, 256, 512, 1024, 2048, 4096]  # 4096 = 无分块

for chunk_size in chunk_sizes:
    huygens_module._huygens_integral = make_chunked_huygens(chunk_size)

    cfg = make_lens_rl_config(fast_mode=True)
    env = LensAlignmentEnv(cfg=cfg)

    # 预热
    env.reset()
    for _ in range(2):
        env.step(env.action_space.sample())

    # 测试
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(5):
        start = time.time()
        env.step(env.action_space.sample())
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"chunk_size={chunk_size:4d}: {avg_time*1000:.1f}ms, {peak_mem:.1f}MB, {1.0/avg_time:.2f} steps/sec")
