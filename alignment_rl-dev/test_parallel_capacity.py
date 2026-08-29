"""测试GPU并行计算多个环境的能力"""
import time
import torch
import multiprocessing as mp
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"CPU核心数: {mp.cpu_count()}")
print()

# 测试单个环境的GPU使用
print("="*60)
print("1. 单环境GPU使用分析")
print("="*60)

cfg = make_lens_rl_config(fast_mode=True)  # 32条光线
env = LensAlignmentEnv(cfg=cfg)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

env.reset()
action = env.action_space.sample()

# 记录初始显存
mem_before = torch.cuda.memory_allocated() / 1024**2

start = time.time()
obs, reward, done, truncated, info = env.step(action)
elapsed = time.time() - start

mem_after = torch.cuda.memory_allocated() / 1024**2
mem_peak = torch.cuda.max_memory_allocated() / 1024**2

print(f"单步计算:")
print(f"  时间: {elapsed*1000:.1f}ms")
print(f"  显存使用: {mem_after:.1f} MB")
print(f"  显存峰值: {mem_peak:.1f} MB")
print(f"  GPU利用率: {mem_peak / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100:.1f}%")

# 估算可并行数量
total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
mem_per_env_gb = mem_peak / 1024
max_parallel_by_mem = int(total_mem_gb * 0.8 / mem_per_env_gb)  # 留20%余量

print(f"\n基于显存估算:")
print(f"  单环境峰值显存: {mem_per_env_gb:.2f} GB")
print(f"  理论最大并行数: {max_parallel_by_mem} 个环境")

# 测试顺序执行多个step
print(f"\n{'='*60}")
print("2. 顺序执行多个step")
print("="*60)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

num_steps = 4
start = time.time()

for i in range(num_steps):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        env.reset()

elapsed = time.time() - start
steps_per_sec = num_steps / elapsed
mem_peak_sequential = torch.cuda.max_memory_allocated() / 1024**2

print(f"顺序执行 {num_steps} 步:")
print(f"  总时间: {elapsed:.2f}s")
print(f"  速度: {steps_per_sec:.2f} steps/sec")
print(f"  显存峰值: {mem_peak_sequential:.1f} MB")
print(f"  单步平均: {elapsed/num_steps*1000:.1f}ms")

# 测试批量并行（如果可能）
print(f"\n{'='*60}")
print("3. 多环境并行能力分析")
print("="*60)

# 创建多个环境测试显存占用
test_parallel = [2, 4, 8, 16]
for n_envs in test_parallel:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 创建多个环境
        envs = [LensAlignmentEnv(cfg=cfg) for _ in range(n_envs)]

        # 重置所有环境
        for env in envs:
            env.reset()

        mem_after_create = torch.cuda.memory_allocated() / 1024**2

        # 执行一步
        for env in envs:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

        mem_peak_multi = torch.cuda.max_memory_allocated() / 1024**2

        print(f"{n_envs}个环境:")
        print(f"  创建后显存: {mem_after_create:.1f} MB")
        print(f"  执行后峰值: {mem_peak_multi:.1f} MB")
        print(f"  平均每环境: {mem_peak_multi/n_envs:.1f} MB")
        print(f"  GPU利用率: {mem_peak_multi / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100:.1f}%")

        # 清理
        del envs

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"{n_envs}个环境: OOM (显存不足)")
            break
        else:
            print(f"{n_envs}个环境: ERROR - {e}")
            break
    except Exception as e:
        print(f"{n_envs}个环境: ERROR - {e}")
        break

print(f"\n{'='*60}")
print("4. GPU算力分配模式")
print("="*60)

print("""
当前GPU计算流程:
1. 光线追迹 (Ray Tracing)
   - 逐面计算交点和折射/反射
   - 并行度: 1024条光线 × 3波长 = 3,072条光线并行

2. Huygens积分 (PSF计算)
   - 计算复振幅叠加
   - 并行度: 最多4096条光线/块 (ray_chunk_size)

3. FFT (MTF计算)
   - 2D傅里叶变换
   - 并行度: 整个图像网格并行

GPU算力分配特点:
- 单个step内部: GPU处理数千条光线的并行计算
- 多个step之间: 顺序执行（一个step完成后才开始下一个）
- 多个环境之间: 可以在不同CUDA流中并行，但受显存限制

建议配置:
- 训练时: 使用多进程并行环境（SubprocVecEnv）
  * 每个进程独立GPU上下文
  * Windows上可能有进程通信开销
  * Linux上效率更高

- 单进程内: 难以实现真正的step级并行
  * 因为每个step需要修改光学系统状态
  * 状态修改是顺序的，无法批量并行
""")

print(f"\n最终建议:")
print(f"  单环境显存: ~0.6 GB")
print(f"  理论最大并行: {max_parallel_by_mem} 个环境")
print(f"  推荐配置: 4-8个环境（平衡速度和稳定性）")
print(f"  预期加速: 3-6倍（考虑进程通信开销）")
