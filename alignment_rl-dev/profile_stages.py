"""详细分析每个计算阶段的耗时"""
import time
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

# Patch各个关键函数
import optics_core.huygens_psf as huygens_module
import optics_core.huygens_mtf as mtf_module

times = {}

original_trace_pupil = huygens_module.trace_pupil_to_image
def timed_trace_pupil(*args, **kwargs):
    start = time.time()
    result = original_trace_pupil(*args, **kwargs)
    times.setdefault('trace_pupil', []).append(time.time() - start)
    return result
huygens_module.trace_pupil_to_image = timed_trace_pupil

original_huygens = huygens_module._huygens_integral
def timed_huygens(*args, **kwargs):
    start = time.time()
    result = original_huygens(*args, **kwargs)
    times.setdefault('huygens_integral', []).append(time.time() - start)
    return result
huygens_module._huygens_integral = timed_huygens

original_compute_mtf = mtf_module.compute_huygens_mtf
def timed_compute_mtf(*args, **kwargs):
    start = time.time()
    result = original_compute_mtf(*args, **kwargs)
    times.setdefault('compute_mtf', []).append(time.time() - start)
    return result
mtf_module.compute_huygens_mtf = timed_compute_mtf

# 运行测试
cfg = make_lens_rl_config(fast_mode=True)
env = LensAlignmentEnv(cfg=cfg)

print(f"配置: {cfg.mtf_num_rays}条光线\n")

# 预热
env.reset()
times.clear()

# 测试5步
for i in range(5):
    start = time.time()
    env.step(env.action_space.sample())
    total = time.time() - start

    trace_t = sum(times.get('trace_pupil', []))
    huygens_t = sum(times.get('huygens_integral', []))
    mtf_t = sum(times.get('compute_mtf', []))

    print(f"Step {i+1}: {total*1000:.1f}ms total")
    print(f"  trace_pupil:      {trace_t*1000:.1f}ms ({trace_t/total*100:.1f}%)")
    print(f"  huygens_integral: {huygens_t*1000:.1f}ms ({huygens_t/total*100:.1f}%)")
    print(f"  compute_mtf:      {mtf_t*1000:.1f}ms ({mtf_t/total*100:.1f}%)")
    print(f"  other:            {(total-trace_t-huygens_t-mtf_t)*1000:.1f}ms\n")

    times.clear()
