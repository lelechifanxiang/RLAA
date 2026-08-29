"""详细分析为什么64光线比32光线慢"""
import time
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

print("GPU信息:")
if torch.cuda.is_available():
    print(f"  设备: {torch.cuda.get_device_name(0)}")
    print(f"  总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
print()

for num_rays in [32, 64]:
    print(f"{'='*60}")
    print(f"测试 {num_rays} 光线")
    print(f"{'='*60}")

    cfg = make_lens_rl_config(fast_mode=True)
    cfg.mtf_num_rays = num_rays

    env = LensAlignmentEnv(cfg=cfg)

    # 计算总光线数
    total_rays = num_rays * num_rays
    n_fields = len(cfg.mtf_field_coords)
    n_wavelengths = 3  # 默认
    total_ray_samples = total_rays * n_fields * n_wavelengths

    print(f"\n计算量:")
    print(f"  瞳面采样: {num_rays} × {num_rays} = {total_rays:,} 光线/视场")
    print(f"  视场数: {n_fields}")
    print(f"  波长数: {n_wavelengths}")
    print(f"  总光线: {total_ray_samples:,}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 单次step测试
    env.reset()

    start = time.time()
    env.step(env.action_space.sample())
    elapsed = time.time() - start

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n性能:")
    print(f"  单步时间: {elapsed*1000:.1f}ms")
    print(f"  速度: {1/elapsed:.2f} steps/sec")
    if torch.cuda.is_available():
        print(f"  GPU峰值显存: {peak_mem:.1f} MB")
        print(f"  每光线显存: {peak_mem*1024 / total_ray_samples:.1f} KB")

    print()

print(f"{'='*60}")
print("结论:")
print("  64光线 = 4倍计算量 (64²/32² = 4)")
print("  如果64光线慢13%, 说明GPU并行效率约为 4.0/1.13 = 3.5x")
print("  这个效率已经很高了，不是瓶颈")
print("  真正的瓶颈可能是:")
print("    1. Ray tracing (38%时间) - CPU序列计算")
print("    2. Python overhead (55%时间) - 无法GPU加速")
print(f"{'='*60}")
