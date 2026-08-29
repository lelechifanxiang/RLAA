"""
4自由度交替对准训练脚本。

特性：
    - 4自由度：偏心（dx, dy）+ 倾斜（rx, ry）
    - 交替运动：奇数步调偏心，偶数步调倾斜
    - 倾斜范围：±1°

运行示例：
    python train_4dof_alternating.py                           # 默认SAC，交替模式
    python train_4dof_alternating.py --mode simultaneous       # 同时调整4自由度（对比实验）
    python train_4dof_alternating.py --timesteps 2000000       # 自定义训练步数
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from config import TrainingConfig
from config_4dof import make_4dof_alternating_config, Alternating4DOFConfig
from env.alternating_lens_env import AlternatingLensEnv, AlternatingBatchLensEnv


def train_4dof_alternating(
    algo: str = "sac",
    motion_mode: str = "alternating",
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    resume_from: str = None,
) -> None:
    """训练4自由度交替对准策略。

    Args:
        algo: 算法选择 ('sac', 'td3', 'ppo')
        motion_mode: 运动模式 ('alternating' 或 'simultaneous')
        total_timesteps: 总训练步数
        seed: 随机种子
        resume_from: 断点续训路径
    """
    cfg = TrainingConfig(algorithm=algo, total_timesteps=total_timesteps, seed=seed)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_4dof_{motion_mode}_{timestamp}"
    tb_path = os.path.join(cfg.log_dir, run_name)
    model_path = os.path.join(cfg.model_dir, run_name)

    # 打印配置信息
    print(f"\n{'='*70}")
    print(f"  4自由度主动对准训练")
    print(f"{'='*70}")
    print(f"  算法        : {algo.upper()}")
    print(f"  运动模式    : {motion_mode.upper()}")
    if motion_mode == "alternating":
        print(f"    - 奇数步  : 调整偏心（dx, dy）")
        print(f"    - 偶数步  : 调整倾斜（rx, ry）")
    else:
        print(f"    - 所有步  : 同时调整4自由度")
    print(f"  自由度      : dx, dy, rx, ry")
    print(f"  倾斜范围    : ±1.0°")
    print(f"  总步数      : {total_timesteps:,}")
    if resume_from:
        print(f"  断点续训    : {resume_from}")
    print(f"  随机种子    : {seed}")
    print(f"  模型保存    : {model_path}")
    print(f"{'='*70}\n")

    # 并行训练环境数
    n_train_envs = 12
    sac_gradient_steps = max(n_train_envs // 4, 1)

    print(f"GPU并行配置:")
    print(f"  训练环境数  : {n_train_envs}")
    print(f"  单环境显存  : ~0.6 GB")
    print(f"  总显存预估  : ~{n_train_envs * 0.6:.1f} GB")
    print(f"  预期速度    : ~{n_train_envs * 1.8:.1f} env-steps/sec")
    print(f"  预计时间    : ~{total_timesteps / (n_train_envs * 1.8) / 3600:.1f} 小时")
    if algo == "sac":
        print(f"  SAC梯度步数 : {sac_gradient_steps}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 构建环境配置
    # ------------------------------------------------------------------
    lens_cfg = make_4dof_alternating_config(fast_mode=True)  # 快速模式（32光线）
    alt_cfg = Alternating4DOFConfig(motion_mode=motion_mode)

    # 构建训练和评估环境
    train_env = AlternatingBatchLensEnv(
        cfg=lens_cfg,
        n_envs=n_train_envs,
        seed=seed,
        alternating_cfg=alt_cfg,
    )
    train_env = VecMonitor(train_env)

    eval_env = AlternatingLensEnv(cfg=lens_cfg, alternating_cfg=alt_cfg)
    eval_env = Monitor(eval_env)

    # ------------------------------------------------------------------
    # 回调：评估 + 检查点保存
    # ------------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path + "_best",
        log_path=tb_path,
        eval_freq=max(cfg.eval_freq // n_train_envs, 1),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // n_train_envs, 1),
        save_path=model_path + "_ckpt",
        name_prefix="rl_model",
        save_replay_buffer=True,
        verbose=0,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ------------------------------------------------------------------
    # 设备选择
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"训练设备: {device}")
    if device == "cuda":
        print(f"GPU型号 : {torch.cuda.get_device_name(0)}\n")

    # ------------------------------------------------------------------
    # 策略网络配置
    # ------------------------------------------------------------------
    policy_kwargs = dict(net_arch=cfg.net_arch)

    # ------------------------------------------------------------------
    # 创建或加载模型
    # ------------------------------------------------------------------
    common_kwargs = dict(
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_path,
        device=device,
        seed=seed,
        verbose=1,
    )

    if resume_from:
        print(f"加载checkpoint: {resume_from}")
        algo_cls = {"sac": SAC, "td3": TD3, "ppo": PPO}[algo]
        model = algo_cls.load(
            resume_from,
            env=train_env,
            device=device,
            tensorboard_log=tb_path,
        )
        if algo == "sac":
            model.gradient_steps = sac_gradient_steps

        # 加载经验回放池
        replay_path = resume_from.replace(".zip", "_replay_buffer.pkl")
        if hasattr(model, "load_replay_buffer") and os.path.exists(replay_path):
            model.load_replay_buffer(replay_path)
            print(f"已加载经验回放池: {replay_path}\n")
        else:
            print("未找到经验回放池，从空buffer开始。\n")

    elif algo == "sac":
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            tau=cfg.tau,
            ent_coef=cfg.ent_coef,
            gradient_steps=sac_gradient_steps,
            **common_kwargs,
        )
    elif algo == "td3":
        from stable_baselines3.common.noise import NormalActionNoise
        n_actions = train_env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        model = TD3(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            tau=cfg.tau,
            action_noise=action_noise,
            **common_kwargs,
        )
    elif algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            **common_kwargs,
        )
    else:
        raise ValueError(f"不支持的算法: {algo}")

    # ------------------------------------------------------------------
    # 开始训练
    # ------------------------------------------------------------------
    print("开始训练...\n")
    latest_path = model_path + "_latest"

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(resume_from),
        )
    except KeyboardInterrupt:
        print("\n收到Ctrl+C，正在保存...")
    finally:
        # 保存最终模型
        model.save(latest_path)
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(latest_path + "_replay_buffer.pkl")

        final_path = model_path + "_final"
        model.save(final_path)
        print(f"\n训练完成！")
        print(f"  最终模型: {final_path}.zip")
        print(f"  最新模型: {latest_path}.zip")

        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4自由度交替对准训练")
    parser.add_argument(
        "--algo", default="sac", choices=["sac", "td3", "ppo"],
        help="RL算法 (default: sac)"
    )
    parser.add_argument(
        "--mode", default="alternating", choices=["alternating", "simultaneous"],
        help="运动模式 (default: alternating)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="总训练步数 (default: 1,000,000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (default: 42)"
    )
    parser.add_argument(
        "--resume_from", default=None, metavar="CKPT_PATH",
        help="从checkpoint断点续训"
    )

    args = parser.parse_args()

    train_4dof_alternating(
        algo=args.algo,
        motion_mode=args.mode,
        total_timesteps=args.timesteps,
        seed=args.seed,
        resume_from=args.resume_from,
    )
