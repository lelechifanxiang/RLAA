"""
测试 lens_env.py 是否能正常使用 optics_core 进行 MTF 计算
"""
import sys
from pathlib import Path
import io

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from env.lens_env import LensAlignmentEnv
from config import LensEnvConfig


def main():
    print("=" * 60)
    print("测试 LensAlignmentEnv (使用 optics_core MTF)")
    print("=" * 60)

    # 创建配置
    cfg = LensEnvConfig()
    print(f"\n配置:")
    print(f"  - MTF 频率: {cfg.mtf_frequencies}")
    print(f"  - MTF 视场: {cfg.mtf_field_coords}")
    print(f"  - MTF 光线数: {cfg.mtf_num_rays}")

    # 创建环境
    print(f"\n1. 创建环境...")
    try:
        env = LensAlignmentEnv(cfg)
        print(f"   ✓ 环境创建成功")
        print(f"   - 动作空间: {env.action_space}")
        print(f"   - 观测空间: {env.observation_space}")
        print(f"   - 自由度: {env.n_dof}")
        print(f"   - DOF 名称: {env.dof_names}")
    except Exception as e:
        print(f"   ✗ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 重置环境
    print(f"\n2. 重置环境...")
    try:
        obs, info = env.reset(seed=42)
        print(f"   ✓ 环境重置成功")
        print(f"   - 观测形状: {obs.shape}")
        print(f"   - 初始质量: {info['quality_metric']:.4f}")
        print(f"   - 初始状态: {info['state']}")
        print(f"   - MTF 观测形状: {info['mtf_obs'].shape}")
        print(f"   - MTF 观测范围: [{info['mtf_obs'].min():.4f}, {info['mtf_obs'].max():.4f}]")
    except Exception as e:
        print(f"   ✗ 环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 执行几步动作
    print(f"\n3. 执行随机动作...")
    try:
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   步骤 {step + 1}:")
            print(f"     动作: {action}")
            print(f"     奖励: {reward:.4f}")
            print(f"     质量: {info['quality_metric']:.4f}")
            print(f"     状态: {info['state']}")
            if terminated:
                print(f"     ✓ 达到成功阈值！")
                break
        print(f"   ✓ 动作执行成功")
    except Exception as e:
        print(f"   ✗ 动作执行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✓ 测试成功！LensAlignmentEnv 使用 optics_core 正常工作")
    print("=" * 60)


if __name__ == "__main__":
    main()
