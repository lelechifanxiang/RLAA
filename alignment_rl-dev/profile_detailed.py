"""更细粒度的性能分析"""
import time
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

# Patch更多函数
import optics_core.huygens_psf as huygens_module

times = {}

def make_timer(name, original_func):
    def timed(*args, **kwargs):
        start = time.time()
        result = original_func(*args, **kwargs)
        times.setdefault(name, []).append(time.time() - start)
        return result
    return timed

# Patch主要函数
huygens_module.trace_pupil_to_image = make_timer('trace_pupil', huygens_module.trace_pupil_to_image)
huygens_module.extract_image_wave_data = make_timer('extract_wave', huygens_module.extract_image_wave_data)
huygens_module.compute_huygens_psf = make_timer('compute_psf', huygens_module.compute_huygens_psf)
huygens_module._huygens_integral = make_timer('huygens_integral', huygens_module._huygens_integral)

# 运行测试
cfg = make_lens_rl_config(fast_mode=True)
env = LensAlignmentEnv(cfg=cfg)

# 预热
env.reset()
times.clear()

# 测试一步
start_total = time.time()
env.step(env.action_space.sample())
total = time.time() - start_total

print(f"总时间: {total*1000:.1f}ms\n")
print("详细分解:")
for name in ['trace_pupil', 'extract_wave', 'huygens_integral', 'compute_psf']:
    t = sum(times.get(name, []))
    count = len(times.get(name, []))
    print(f"  {name:20s}: {t*1000:6.1f}ms ({count:2d} calls) = {t/total*100:5.1f}%")

accounted = sum(sum(times.get(name, [])) for name in times.keys())
print(f"  {'other':20s}: {(total-accounted)*1000:6.1f}ms            = {(total-accounted)/total*100:5.1f}%")
