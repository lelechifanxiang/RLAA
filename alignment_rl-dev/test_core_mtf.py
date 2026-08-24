"""
测试使用 optics_core 进行 MTF 计算的镜头环境。

验证：
1. 环境能够正常初始化
2. optics_core MTF 计算能够运行
3. reset 和 step 能够正常工作
"""
import sys
from pathlib import Path

# 确保能找到 optics_core
optics_core_path = Path(__file__).parent.parent / "optics_core-dev"
if str(optics_core_path) not in sys.path:
    sys.path.insert(0, str(optics_core_path))

import numpy as np
from env.lens_env import LensAlignmentEnv
from config import LensEnvConfig


def test_env_creation():
    """测试环境创建"""
    print("=" * 60)
    print("测试 1: 环境创建")
    print("=" * 60)

    cfg = LensEnvConfig(
        mtf_field_coords=[(0.0, 0.0)],
        mtf_field_indices=[0],
        mtf_frequencies=[20.0, 30.0, 50.0],
        mtf_num_rays=32,  # 降低采样率以加快测试
        obs_history_len=3,
        max_episode_steps=10,
    )

    try:
        env = LensAlignmentEnv(cfg)
        print("✓ 环境创建成功")
        print(f"  - 动作空间: {env.action_space}")
        print(f"  - 观测空间: {env.observation_space}")
        print(f"  - DOF 数量: {env.n_dof}")
        print(f"  - DOF 名称: {env.dof_names}")
        return env
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_env_reset(env):
    """测试环境 reset"""
    print("\n" + "=" * 60)
    print("测试 2: 环境 Reset")
    print("=" * 60)

    try:
        obs, info = env.reset(seed=42)
        print("✓ Reset 成功")
        print(f"  - 观测形状: {obs.shape}")
        print(f"  - 观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  - 质量指标: {info['quality_metric']:.6f}")
        print(f"  - 初始状态: {info['state']}")
        print(f"  - MTF 观测维度: {info['mtf_obs'].shape}")
        print(f"  - MTF 观测范围: [{info['mtf_obs'].min():.3f}, {info['mtf_obs'].max():.3f}]")
        print(f"  - 原始 MTF 范围: [{info['raw_mtf_obs'].min():.3f}, {info['raw_mtf_obs'].max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Reset 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_step(env):
    """测试环境 step"""
    print("\n" + "=" * 60)
    print("测试 3: 环境 Step")
    print("=" * 60)

    try:
        # 执行几步动作
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"\n步骤 {i+1}:")
            print(f"  - 动作: {action}")
            print(f"  - 奖励: {reward:.6f}")
            print(f"  - 质量指标: {info['quality_metric']:.6f}")
            print(f"  - 状态: {info['state']}")
            print(f"  - Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                print(f"  - Episode 结束")
                break

        print("\n✓ Step 测试成功")
        return True
    except Exception as e:
        print(f"\n✗ Step 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mtf_comparison(env):
    """测试 MTF 计算的一致性"""
    print("\n" + "=" * 60)
    print("测试 4: MTF 一致性检查")
    print("=" * 60)

    try:
        # Reset 环境
        env.reset(seed=123)

        # 获取标称 MTF
        nominal_mtf = env.nominal_mtf_obs()
        print(f"  - 标称 MTF 维度: {nominal_mtf.shape}")
        print(f"  - 标称 MTF 范围: [{nominal_mtf.min():.3f}, {nominal_mtf.max():.3f}]")
        print(f"  - 标称 MTF 均值: {nominal_mtf.mean():.3f}")

        # 检查 MTF 值是否合理（应该在 [0, 1] 范围内）
        if np.all(nominal_mtf >= 0) and np.all(nominal_mtf <= 1):
            print("✓ MTF 值在合理范围内 [0, 1]")
        else:
            print(f"✗ 警告: MTF 值超出范围")

        # 检查 MTF 不全为零
        if np.any(nominal_mtf > 0):
            print("✓ MTF 包含非零值")
        else:
            print("✗ 警告: MTF 全为零")

        return True
    except Exception as e:
        print(f"✗ MTF 一致性检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("测试 optics_core MTF 集成到镜头 RL 环境")
    print("=" * 60)

    # 测试 1: 创建环境
    env = test_env_creation()
    if env is None:
        print("\n环境创建失败，终止测试")
        return

    # 测试 2: Reset
    if not test_env_reset(env):
        print("\nReset 失败，终止测试")
        return

    # 测试 3: Step
    if not test_env_step(env):
        print("\nStep 失败，终止测试")
        return

    # 测试 4: MTF 一致性
    test_mtf_comparison(env)

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
