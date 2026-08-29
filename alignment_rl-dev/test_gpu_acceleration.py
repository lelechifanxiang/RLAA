"""
验证 GPU 加速是否正常工作，并测试 MTF 计算的加速效果。
"""
import time
import torch
import numpy as np
from config import LensEnvConfig, LensGroupConfig
import env.lens_env as el


def test_cuda_available():
    """检查 CUDA 是否可用"""
    print("=" * 60)
    print("CUDA 可用性检查")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM 总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()


def test_torch_performance():
    """测试 PyTorch GPU vs CPU 性能"""
    print("=" * 60)
    print("PyTorch 矩阵运算性能对比")
    print("=" * 60)

    size = 5000

    # CPU
    start = time.perf_counter()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.perf_counter() - start

    # GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        start = time.perf_counter()
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        print(f"矩阵大小: {size}x{size}")
        print(f"  CPU 时间: {cpu_time:.4f}s")
        print(f"  GPU 时间: {gpu_time:.4f}s")
        print(f"  加速比: {cpu_time/gpu_time:.2f}x")
    else:
        print("CUDA 不可用，跳过 GPU 测试")
    print()


def build_test_env():
    """构建测试环境"""
    cfg = LensEnvConfig(
        tol_radius_rel=0.0,
        tol_thickness_mm=0.0,
        tol_decenter_mm=0.0,
        tol_tilt_deg=0.0,
        use_compensator=False,
        mtf_num_rays=128,
        mtf_grid_size=None,
        mtf_field_coords=[(0.0, 0.0), (14.0, 0.0), (0.0, 14.0)],
        mtf_field_indices=[0, 1, 2],
        lens_groups=[
            LensGroupConfig(
                surf_front=3,
                surf_rear=4,
                z_source_surf=2,
                active_dofs=["dx", "dy"],
            )
        ],
    )

    env = el.LensAlignmentEnv(cfg=cfg)
    env.reset(seed=0)
    return env


def benchmark_mtf_computation(repeats=10):
    """Benchmark the active Double Gauss optics_core MTF path."""
    print("=" * 60)
    print("MTF 计算性能对比")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for backend in ("optics_core",):
        print(f"\n测试 {backend} backend (device={device})...")
        env = build_test_env()
        state = np.zeros(env.n_dof, dtype=np.float64)

        timings = []
        for i in range(repeats):
            start = time.perf_counter()
            env._mgr.apply_alignment_state(state)
            obs, _, _, _, _ = env.step(np.zeros(env.n_dof, dtype=np.float64))
            elapsed = (time.perf_counter() - start) * 1000
            timings.append(elapsed)
            if (i + 1) % 3 == 0:
                print(f"  迭代 {i+1}/{repeats}: {elapsed:.2f} ms")

        avg_time = np.mean(timings)
        min_time = np.min(timings)
        max_time = np.max(timings)

        results[f"{backend}_{device}"] = avg_time

        print(f"\n{backend} ({device}) 统计:")
        print(f"  平均: {avg_time:.2f} ms")
        print(f"  最小: {min_time:.2f} ms")
        print(f"  最大: {max_time:.2f} ms")

    print("\n" + "=" * 60)
    print("Double Gauss optics_core 总结")
    print("=" * 60)

    print()


def main():
    test_cuda_available()
    test_torch_performance()
    benchmark_mtf_computation(repeats=10)

    print("=" * 60)
    print("✓ GPU 加速验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
