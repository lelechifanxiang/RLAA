"""详细分析MTF计算的各个阶段性能。"""
import time
import torch
import numpy as np
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

def profile_mtf_stages():
    """逐阶段测量MTF计算时间。"""
    cfg = make_lens_rl_config(fast_mode=True)  # 32条光线
    env = LensAlignmentEnv(cfg=cfg)

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"\n配置: {cfg.mtf_num_rays}条光线, {cfg.mtf_grid_size}网格\n")

    # 预热
    env.reset()
    _ = env._mgr._compute_mtf_obs()

    # 详细测量
    n_samples = 5
    times = {
        'total': [],
        'prepare': [],
        'mtf_call': [],
    }

    for i in range(n_samples):
        # 重置环境状态
        env.reset()

        t0 = time.time()

        # 准备阶段
        t_prep = time.time()
        cfg = env._mgr.cfg
        valid_field_indices = [
            i for i in cfg.mtf_field_indices
            if 0 <= i < len(env._mgr._core_system.fields)
        ]
        settings = env._mgr._core_system.analysis.mtf().settings
        times['prepare'].append(time.time() - t_prep)

        # MTF计算
        t_mtf = time.time()
        result = env._mgr._core_system.analysis.mtf(settings).run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times['mtf_call'].append(time.time() - t_mtf)

        times['total'].append(time.time() - t0)

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"Run {i+1}: {times['total'][-1]:.3f}s, "
                  f"GPU内存: {mem_allocated:.1f}MB (reserved: {mem_reserved:.1f}MB)")
        else:
            print(f"Run {i+1}: {times['total'][-1]:.3f}s")

    print(f"\n{'='*60}")
    print(f"平均耗时统计 (n={n_samples}):")
    print(f"  准备阶段:   {np.mean(times['prepare'])*1000:.1f}ms")
    print(f"  MTF计算:    {np.mean(times['mtf_call'])*1000:.1f}ms")
    print(f"  总计:       {np.mean(times['total'])*1000:.1f}ms")
    print(f"{'='*60}\n")

    # 估算训练速度
    mtf_time = np.mean(times['total'])
    steps_per_sec = 1.0 / mtf_time
    print(f"单环境预估速度: {steps_per_sec:.2f} steps/sec")
    print(f"4环境预估速度:  {steps_per_sec * 4:.2f} steps/sec")
    print(f"完成1M步需要:   {1_000_000 / (steps_per_sec * 4) / 3600:.1f} 小时")

if __name__ == "__main__":
    profile_mtf_stages()
