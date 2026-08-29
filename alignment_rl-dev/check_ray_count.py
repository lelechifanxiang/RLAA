"""检查MTF计算实际使用的光线数量。"""
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

# 测试不同配置
configs = [
    (32, "fast_mode (训练)"),
    (128, "standard (评估)"),
    (256, "high quality"),
]

for num_rays, label in configs:
    cfg = make_lens_rl_config(fast_mode=False)
    cfg.mtf_num_rays = num_rays

    env = LensAlignmentEnv(cfg=cfg)
    env.reset()

    # 访问内部系统
    system = env._mgr._core_system

    print(f"\n{label}: mtf_num_rays={num_rays}")
    print(f"  视场数: {len(system.fields)}")
    print(f"  波长数: {len(system.wavelengths)}")
    print(f"  实际光线数: {num_rays * num_rays} (pupil_sample_count={num_rays})")
    print(f"  总计算量: {len(system.fields)} fields x {num_rays}^2 rays = {len(system.fields) * num_rays * num_rays:,} rays")

    if torch.cuda.is_available():
        # 估算显存需求（粗略）
        # 每条光线约需: position(3) + direction(3) + opl(1) + valid(1) = 8 × 8 bytes = 64 bytes
        # 加上中间计算（PSF, FFT等）约10倍
        bytes_per_ray = 64 * 10
        total_bytes = len(system.fields) * num_rays * num_rays * bytes_per_ray
        print(f"  估算显存: {total_bytes / 1024**2:.1f} MB")

print(f"\nGPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
