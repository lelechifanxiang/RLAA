"""测试episode长度，验证配置修改是否有效"""
import time
import numpy as np
from config import make_lens_rl_config
from env.lens_env import LensAlignmentEnv

def test_episode_length(num_episodes=5):
    """测试多个episode，统计平均步数"""
    cfg = make_lens_rl_config(fast_mode=True)
    env = LensAlignmentEnv(cfg=cfg)

    print(f"环境配置:")
    print(f"  success_threshold: {cfg.success_threshold}")
    print(f"  max_episode_steps: {cfg.max_episode_steps}")
    print(f"  初始偏移: ±{cfg.lens_groups[0].init_dx_mm}mm")
    print()

    episode_lengths = []
    episode_returns = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        total_reward = 0
        steps = 0

        print(f"Episode {ep+1}:")
        print(f"  初始quality: {info['quality_metric']:.4f}")

        start_time = time.time()
        while True:
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        elapsed = time.time() - start_time
        episode_lengths.append(steps)
        episode_returns.append(total_reward)

        print(f"  步数: {steps}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  最终quality: {info['quality_metric']:.4f}")
        print(f"  耗时: {elapsed:.1f}秒")
        print(f"  速度: {steps/elapsed:.2f} steps/sec")
        print()

    print(f"\n统计结果（{num_episodes}个episodes）:")
    print(f"  平均步数: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  平均回报: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")

if __name__ == "__main__":
    test_episode_length()
