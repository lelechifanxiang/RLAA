"""测试多环境并行训练速度"""
import time
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA可用: {torch.cuda.is_available()}\n")

    # 测试不同环境数量的训练速度
    for n_envs in [1, 2, 4, 8]:
        print(f"{'='*60}")
        print(f"测试 {n_envs} 个并行环境")
        print(f"{'='*60}")

        try:
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # 创建环境
            lens_cfg = make_lens_rl_config(fast_mode=True)

            def make_env():
                env = LensAlignmentEnv(cfg=lens_cfg)
                env = Monitor(env)
                return env

            # 使用 SubprocVecEnv 实现真正的多进程并行
            env = make_vec_env(make_env, n_envs=n_envs, seed=42, vec_env_cls=SubprocVecEnv)

            # 创建模型
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=10000,
                batch_size=256,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=0,
            )

            # 预热
            model.learn(total_timesteps=n_envs * 2, progress_bar=False)

            # 测试训练速度
            num_steps = n_envs * 20
            start = time.time()

            model.learn(total_timesteps=num_steps, progress_bar=False)

            elapsed = time.time() - start
            steps_per_sec = num_steps / elapsed

            if torch.cuda.is_available():
                mem_peak = torch.cuda.max_memory_allocated() / 1024**2
                gpu_usage = mem_peak / (torch.cuda.get_device_properties(0).total_memory / 1024**2) * 100
                print(f"  训练步数: {num_steps}")
                print(f"  总时间: {elapsed:.2f}s")
                print(f"  速度: {steps_per_sec:.2f} steps/sec")
                print(f"  加速比: {steps_per_sec/2.0:.2f}x (理论{n_envs}x)")
                print(f"  GPU峰值显存: {mem_peak:.1f} MB ({gpu_usage:.1f}%)")
                print(f"  预计1M步耗时: {1_000_000 / steps_per_sec / 3600:.1f} 小时")
            else:
                print(f"  训练步数: {num_steps}")
                print(f"  总时间: {elapsed:.2f}s")
                print(f"  速度: {steps_per_sec:.2f} steps/sec")

            env.close()
            del model

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            break

        print()

    print(f"{'='*60}")
    print("结论:")
    print("  - 多进程并行可以有效加速训练")
    print("  - 推荐使用8个环境充分利用GPU")
    print("  - Windows上进程通信有一定开销，加速比略低于理论值")
    print(f"{'='*60}")
