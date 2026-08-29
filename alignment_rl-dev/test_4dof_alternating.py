"""
快速测试4自由度交替对准系统。

验证：
    1. 环境创建和配置正确性
    2. 交替屏蔽机制工作正常
    3. 动作空间和观测空间维度正确
    4. 一个完整episode能正常运行
"""

import numpy as np
from config_4dof import make_4dof_alternating_config, Alternating4DOFConfig
from env.alternating_lens_env import AlternatingLensEnv


def test_basic_functionality():
    """测试基本功能"""
    print("="*70)
    print("测试4自由度交替对准系统")
    print("="*70)

    # 1. 创建配置
    print("\n1. 创建配置...")
    lens_cfg = make_4dof_alternating_config(fast_mode=True)
    alt_cfg = Alternating4DOFConfig(motion_mode='alternating')

    print(f"   镜组数量: {len(lens_cfg.lens_groups)}")
    active_groups = [i for i, lg in enumerate(lens_cfg.lens_groups) if lg.active_dofs]
    print(f"   激活镜组: {active_groups}")
    print(f"   自由度: {lens_cfg.lens_groups[1].active_dofs}")
    print(f"   倾斜范围: +/-{lens_cfg.lens_groups[1].limit_rx_deg} deg")
    print("   [OK] 配置创建成功")

    # 2. 创建环境
    print("\n2. 创建环境...")
    try:
        env = AlternatingLensEnv(cfg=lens_cfg, alternating_cfg=alt_cfg)
        print(f"   动作空间: {env.action_space}")
        print(f"   观测空间: {env.observation_space}")
        print("   [OK] 环境创建成功")
    except Exception as e:
        print(f"   [FAIL] 环境创建失败: {e}")
        return False

    # 3. 测试reset
    print("\n3. 测试reset...")
    try:
        obs, info = env.reset(seed=42)
        print(f"   观测维度: {obs.shape}")
        print(f"   初始状态: {info['state']}")
        print(f"   初始质量: {info['quality_metric']:.4f}")
        print(f"   激活模式: {info['active_dofs']}")
        print("   [OK] Reset成功")
    except Exception as e:
        print(f"   [FAIL] Reset失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 测试交替屏蔽机制
    print("\n4. 测试交替屏蔽机制...")
    test_action = np.array([0.5, 0.5, 0.3, 0.3])  # [dx, dy, rx, ry]

    print(f"   原始动作: {test_action}")

    for step in range(1, 5):
        obs, reward, terminated, truncated, info = env.step(test_action)

        masked = info['masked_action']
        active = info['active_dofs']
        state = info['state']

        print(f"\n   步数 {step} ({active}):")
        print(f"     屏蔽后动作: {masked}")
        print(f"     当前状态: {state}")
        print(f"     质量指标: {info['quality_metric']:.4f}")

        # 验证屏蔽正确性
        if step % 2 == 1:  # 奇数步应调偏心
            if alt_cfg.decenter_on_odd_steps:
                assert active == 'decenter', f"奇数步应为偏心模式，实际为{active}"
                assert masked[2] == 0.0 and masked[3] == 0.0, "倾斜动作应被屏蔽"
                print(f"     [OK] 偏心模式正确，倾斜已屏蔽")
        else:  # 偶数步应调倾斜
            if alt_cfg.decenter_on_odd_steps:
                assert active == 'tilt', f"偶数步应为倾斜模式，实际为{active}"
                assert masked[0] == 0.0 and masked[1] == 0.0, "偏心动作应被屏蔽"
                print(f"     [OK] 倾斜模式正确，偏心已屏蔽")

        if terminated or truncated:
            print(f"\n   Episode在第{step}步结束")
            break

    print("\n   [OK] 交替屏蔽机制工作正常")

    # 5. 测试完整episode
    print("\n5. 测试完整episode...")
    obs, info = env.reset(seed=123)
    initial_quality = info['quality_metric']

    step_count = 0
    total_reward = 0

    while step_count < 50:  # 最多50步
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        total_reward += reward

        if terminated or truncated:
            break

    final_quality = info['quality_metric']

    print(f"   初始质量: {initial_quality:.4f}")
    print(f"   最终质量: {final_quality:.4f}")
    print(f"   质量改进: {final_quality - initial_quality:.4f}")
    print(f"   总步数: {step_count}")
    print(f"   总奖励: {total_reward:.2f}")
    print(f"   成功: {info['success']}")
    print("   [OK] 完整episode运行成功")

    env.close()

    # 6. 测试同时模式（对比）
    print("\n6. 测试同时模式...")
    alt_cfg_sim = Alternating4DOFConfig(motion_mode='simultaneous')
    env_sim = AlternatingLensEnv(cfg=lens_cfg, alternating_cfg=alt_cfg_sim)

    obs, info = env_sim.reset(seed=42)
    test_action = np.array([0.5, 0.5, 0.3, 0.3])
    obs, reward, terminated, truncated, info = env_sim.step(test_action)

    masked = info['masked_action']
    print(f"   原始动作: {test_action}")
    print(f"   屏蔽后动作: {masked}")
    print(f"   激活模式: {info['motion_mode']}")

    # 同时模式不应屏蔽
    assert np.allclose(masked, test_action), "同时模式不应屏蔽动作"
    print("   [OK] 同时模式工作正常（无屏蔽）")

    env_sim.close()

    print("\n" + "="*70)
    print("所有测试通过！[OK]")
    print("="*70)

    return True


def test_batch_env():
    """测试批量环境"""
    print("\n" + "="*70)
    print("测试批量交替环境")
    print("="*70)

    from env.alternating_lens_env import AlternatingBatchLensEnv

    lens_cfg = make_4dof_alternating_config(fast_mode=True)
    alt_cfg = Alternating4DOFConfig(motion_mode='alternating')

    print("\n创建批量环境（4个并行环境）...")
    try:
        batch_env = AlternatingBatchLensEnv(
            cfg=lens_cfg,
            n_envs=4,
            seed=42,
            alternating_cfg=alt_cfg,
        )
        print(f"   环境数量: {batch_env.num_envs}")
        print(f"   动作空间: {batch_env.action_space}")
        print(f"   观测空间: {batch_env.observation_space}")
        print("   [OK] 批量环境创建成功")
    except Exception as e:
        print(f"   [FAIL] 批量环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n测试批量step...")
    obs = batch_env.reset()
    print(f"   观测shape: {obs.shape}")

    # 批量动作
    actions = np.random.uniform(-1, 1, size=(4, 4))
    obs, rewards, dones, infos = batch_env.step(actions)

    print(f"   观测shape: {obs.shape}")
    print(f"   奖励shape: {rewards.shape}")
    print(f"   Done标志: {dones}")
    print("   [OK] 批量step成功")

    batch_env.close()

    print("\n" + "="*70)
    print("批量环境测试通过！[OK]")
    print("="*70)

    return True


if __name__ == "__main__":
    print("\n开始测试4自由度交替对准系统...\n")

    # 测试基本功能
    success1 = test_basic_functionality()

    if success1:
        # 测试批量环境
        success2 = test_batch_env()

        if success2:
            print("\n" + "="*70)
            print("所有测试完成！系统工作正常。")
            print("="*70)
            print("\n可以开始训练：")
            print("  python train_4dof_alternating.py --mode alternating")
            print("  python train_4dof_alternating.py --mode simultaneous")
            print("="*70 + "\n")
        else:
            print("\n批量环境测试失败！")
    else:
        print("\n基本功能测试失败！")
