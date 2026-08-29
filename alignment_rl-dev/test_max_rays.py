"""测试不同光线数配置的显存使用和速度"""
import time
import torch
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"CUDA可用: {torch.cuda.is_available()}\n")

# 测试不同光线数配置
configs = [
    (16, "16 rays (256 total)"),
    (32, "32 rays (1,024 total)"),
    (64, "64 rays (4,096 total)"),
    (96, "96 rays (9,216 total)"),
    (128, "128 rays (16,384 total)"),
]

results = []

for num_rays, label in configs:
    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"{'='*60}")

    try:
        cfg = make_lens_rl_config(fast_mode=False)
        cfg.mtf_num_rays = num_rays

        env = LensAlignmentEnv(cfg=cfg)

        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 预热
        env.reset()
        env.step(env.action_space.sample())

        # 测试
        num_steps = 5
        start = time.time()

        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                env.reset()

        elapsed = time.time() - start
        steps_per_sec = num_steps / elapsed
        ms_per_step = elapsed / num_steps * 1000

        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            peak_mem_gb = peak_mem_mb / 1024
        else:
            peak_mem_mb = 0
            peak_mem_gb = 0

        results.append({
            'rays': num_rays,
            'label': label,
            'speed': steps_per_sec,
            'ms_per_step': ms_per_step,
            'mem_mb': peak_mem_mb,
            'mem_gb': peak_mem_gb,
            'success': True
        })

        print(f"  速度: {steps_per_sec:.2f} steps/sec")
        print(f"  单步: {ms_per_step:.1f}ms")
        print(f"  显存: {peak_mem_mb:.1f} MB ({peak_mem_gb:.2f} GB)")
        print(f"  状态: OK")

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"  状态: OOM (显存不足)")
            results.append({
                'rays': num_rays,
                'label': label,
                'success': False,
                'error': 'OOM'
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"  状态: ERROR - {e}")
            results.append({
                'rays': num_rays,
                'label': label,
                'success': False,
                'error': str(e)
            })
    except Exception as e:
        print(f"  状态: ERROR - {e}")
        results.append({
            'rays': num_rays,
            'label': label,
            'success': False,
            'error': str(e)
        })

# 总结
print(f"\n{'='*60}")
print("测试总结")
print(f"{'='*60}\n")

successful = [r for r in results if r['success']]
if successful:
    print("成功配置:")
    print(f"{'光线数':<10} {'速度':<15} {'单步时间':<12} {'显存使用'}")
    print("-" * 60)
    for r in successful:
        print(f"{r['rays']:<10} {r['speed']:>6.2f} steps/sec  {r['ms_per_step']:>7.1f} ms   {r['mem_gb']:>6.2f} GB")

    # 找出最快的配置
    fastest = max(successful, key=lambda x: x['speed'])
    print(f"\n最快配置: {fastest['rays']}条光线 ({fastest['speed']:.2f} steps/sec)")

    # 找出显存使用最高但仍成功的配置
    max_mem = max(successful, key=lambda x: x['mem_gb'])
    print(f"最大可用配置: {max_mem['rays']}条光线 ({max_mem['mem_gb']:.2f} GB显存)")

failed = [r for r in results if not r['success']]
if failed:
    print(f"\n失败配置:")
    for r in failed:
        print(f"  {r['rays']}条光线: {r.get('error', 'Unknown')}")

print(f"\n建议:")
if successful:
    # 计算速度/精度权衡
    baseline = [r for r in successful if r['rays'] == 32]
    if baseline:
        base_speed = baseline[0]['speed']
        print(f"  - 训练用: 32条光线 (基准速度)")
        faster = [r for r in successful if r['speed'] > base_speed * 1.2 and r['rays'] < 32]
        if faster:
            best_fast = max(faster, key=lambda x: x['speed'])
            speedup = best_fast['speed'] / base_speed
            print(f"  - 快速训练: {best_fast['rays']}条光线 ({speedup:.1f}x加速)")
        slower = [r for r in successful if r['rays'] > 32]
        if slower:
            best_quality = max(slower, key=lambda x: x['rays'])
            slowdown = base_speed / best_quality['speed']
            print(f"  - 高精度评估: {best_quality['rays']}条光线 ({slowdown:.1f}x变慢)")
