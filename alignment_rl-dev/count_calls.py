"""统计MTF计算中的函数调用次数"""
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

# 统计调用次数
import optics_core.huygens_psf as huygens_module
import optics_core.tracing._core as tracing_module

counts = {}

def make_counter(name, original_func):
    def counted(*args, **kwargs):
        counts[name] = counts.get(name, 0) + 1
        return original_func(*args, **kwargs)
    return counted

# Patch函数
huygens_module.trace_pupil_to_image = make_counter('trace_pupil_to_image', huygens_module.trace_pupil_to_image)
huygens_module.extract_image_wave_data = make_counter('extract_image_wave_data', huygens_module.extract_image_wave_data)
huygens_module._huygens_integral = make_counter('huygens_integral', huygens_module._huygens_integral)
huygens_module.compute_huygens_psf = make_counter('compute_huygens_psf', huygens_module.compute_huygens_psf)

if hasattr(tracing_module, 'SequentialSurfaceRayTracer'):
    original_trace = tracing_module.SequentialSurfaceRayTracer.trace
    def counted_trace(self, *args, **kwargs):
        counts['SequentialRayTracer.trace'] = counts.get('SequentialRayTracer.trace', 0) + 1
        return original_trace(self, *args, **kwargs)
    tracing_module.SequentialSurfaceRayTracer.trace = counted_trace

# 运行测试
cfg = make_lens_rl_config(fast_mode=True)
env = LensAlignmentEnv(cfg=cfg)

print("执行一步env.step()...\n")
env.reset()
counts.clear()

env.step(env.action_space.sample())

print("函数调用次数:")
for name, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {name:40s}: {count:4d} 次")

print(f"\n总计: {sum(counts.values())} 次函数调用")
