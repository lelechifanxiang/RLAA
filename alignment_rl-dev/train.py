"""
强化学习训练脚本（镜头主动对准）。

支持算法：SAC（默认）、TD3、PPO。
- SAC/TD3：离策略，样本效率高，适合连续动作空间。
- PPO：在策略，并行环境加速，适合快速原型。

运行示例：
    python train.py                    # 使用默认配置（SAC）
    python train.py --algo ppo         # 切换 PPO
    python train.py --timesteps 2000000
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
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from config import TrainingConfig, LensEnvConfig, make_lens_rl_config
from env.lens_env import LensAlignmentEnv
from env.batch_lens_env import BatchLensAlignmentVecEnv


# ======================================================================
# 辅助：从 checkpoint 路径解析已完成步数
# ======================================================================

def _parse_steps_from_ckpt(path: str) -> int:
    """从文件名 rl_model_900000_steps.zip 中解析步数，解析失败返回 0。"""
    import re
    basename = os.path.basename(path)
    m = re.search(r'(\d+)_steps', basename)
    return int(m.group(1)) if m else 0


class _StopFileCallback(BaseCallback):
    """Stop after the current rollout when a sentinel file is created."""

    def __init__(self, stop_file: str):
        super().__init__(verbose=0)
        self.stop_file = stop_file

    def _on_step(self) -> bool:
        return not os.path.exists(self.stop_file)


# ======================================================================
# 辅助：构建单个环境（供 EvalCallback 使用）
# ======================================================================

def make_lens_env(lens_cfg: LensEnvConfig | None = None, seed: int = 0):
    """工厂：带 Monitor 的 LensAlignmentEnv。"""
    if lens_cfg is None:
        lens_cfg = make_lens_rl_config()

    def _init():
        env = LensAlignmentEnv(cfg=lens_cfg)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ======================================================================
# 主训练函数
# ======================================================================

def train(
    algo: str = "sac",
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    resume_from: str = None,
    forever: bool = False,
    chunk_timesteps: int = 10_000,
) -> None:
    cfg = TrainingConfig(algorithm=algo, total_timesteps=total_timesteps, seed=seed)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_lens_{timestamp}"
    tb_path = os.path.join(cfg.log_dir, run_name)
    model_path = os.path.join(cfg.model_dir, run_name)

    steps_done = _parse_steps_from_ckpt(resume_from) if resume_from else 0
    remaining_timesteps = total_timesteps - steps_done

    # The optical path is one process with a batched MultiOpticalSystem.
    # RTX 5060 Ti scaling benchmark: 12 designs reach ~28.8 logical
    # env-steps/s and reserve ~10 GiB, leaving room for SAC/eval state.
    n_train_envs = 12
    sac_gradient_steps = max(n_train_envs // 4, 1)
    print("  optical execution: single-process MultiOpticalSystem design batch")
    print(f"\n{'='*60}")
    print(f"  算法      : {algo.upper()}")
    print(f"  任务      : lens (主动对准)")
    print(f"  训练环境数: {n_train_envs}")
    print(f"  总步数    : {'无限' if forever else f'{total_timesteps:,}'}")
    if resume_from:
        print(f"  断点续训  : {resume_from}")
        print(f"  已完成步数: {steps_done:,}")
        print(f"  剩余步数  : {remaining_timesteps:,}")
    print(f"  随机种子  : {seed}")
    print(f"  模型保存  : {model_path}")
    print(f"{'='*60}")
    print(f"\nGPU并行配置:")
    print(f"  单环境显存: ~0.6 GB")
    print(f"  {n_train_envs}环境总显存: ~{n_train_envs * 0.6:.1f} GB")
    print(f"  预期速度  : ~{n_train_envs * 1.8:.1f} logical env-steps/sec (batch benchmark)")
    print(f"  预计完成时间: ~{total_timesteps / (n_train_envs * 1.8) / 3600:.1f} 小时（不含评估）")
    if algo == "sac":
        print(f"  SAC gradient_steps: {sac_gradient_steps}（保持每样本更新比例）")
    print(f"{'='*60}\n")

    if not forever and resume_from and remaining_timesteps <= 0:
        print("已达到目标步数，无需继续训练。")
        return

    # ------------------------------------------------------------------
    # 构建训练 / 评估环境
    # ------------------------------------------------------------------
    lens_cfg = make_lens_rl_config(fast_mode=True)  # 使用快速模式（32条光线）
    # The vector wrapper keeps four logical episodes and one optics batch.
    # One process owns the complete design batch.
    train_env = VecMonitor(BatchLensAlignmentVecEnv(lens_cfg, n_envs=n_train_envs, seed=seed))
    eval_env = Monitor(LensAlignmentEnv(cfg=lens_cfg))

    # ------------------------------------------------------------------
    # 回调：定期评估 + 定期保存检查点
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
        save_replay_buffer=True,   # 保存经验回放池，支持断点续训
        verbose=0,
    )
    callbacks_list = [eval_callback, checkpoint_callback]
    stop_file = model_path + ".STOP"
    if forever:
        callbacks_list.append(_StopFileCallback(stop_file))
        print(f"  后台停止方式: 创建 {stop_file}")
    callbacks = CallbackList(callbacks_list)

    # ------------------------------------------------------------------
    # 设备选择
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  训练设备 : {device}")
    if device == "cuda":
        print(f"  GPU 型号  : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 策略网络配置（共享）
    # ------------------------------------------------------------------
    policy_kwargs = dict(net_arch=cfg.net_arch)

    # ------------------------------------------------------------------
    # 选择算法
    # ------------------------------------------------------------------
    # ----------------------------------------------------------------
    # 公共构造参数（新建 or 续训均使用）
    # ----------------------------------------------------------------
    common_kwargs = dict(
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_path,
        device=device,
        seed=seed,
        verbose=1,
    )

    if resume_from:
        # ------------------------------------------------------------------
        # 断点续训：从 checkpoint 加载模型权重（及经验池，如果存在）
        # ------------------------------------------------------------------
        print(f"\n  加载 checkpoint: {resume_from}")
        algo_cls = {"sac": SAC, "td3": TD3, "ppo": PPO}[algo]
        model = algo_cls.load(
            resume_from,
            env=train_env,
            device=device,
            tensorboard_log=tb_path,
        )
        if algo == "sac":
            # A checkpoint created with four designs has one update per
            # vector step.  Twelve designs collect three times as many
            # transitions per vector step, so scale updates to preserve the
            # original update-to-sample ratio.
            model.gradient_steps = sac_gradient_steps
        # 尝试加载同名经验回放池（SB3 >= 1.4 checkpoint 保存时带 _replay_buffer.pkl）
        replay_path = resume_from.replace(".zip", "_replay_buffer.pkl")
        if hasattr(model, "load_replay_buffer") and os.path.exists(replay_path):
            model.load_replay_buffer(replay_path)
            print(f"  已加载经验回放池: {replay_path}")
        else:
            print("  未找到经验回放池文件，将从空 buffer 重新收集数据。")
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
        raise ValueError(f"不支持的算法: {algo}，可选 sac / td3 / ppo")

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    latest_path = model_path + "_latest"
    try:
        if forever:
            while True:
                model.learn(
                    total_timesteps=chunk_timesteps,
                    callback=callbacks,
                    progress_bar=False,
                    reset_num_timesteps=False,
                )
                model.save(latest_path)
                if hasattr(model, "save_replay_buffer"):
                    model.save_replay_buffer(latest_path + "_replay_buffer.pkl")
                print(f"\n已保存持续训练状态: {latest_path}.zip (steps={model.num_timesteps})")
                if os.path.exists(stop_file):
                    os.remove(stop_file)
                    break
        else:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=not bool(resume_from),  # 续训时保持步数连续
            )
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在保存当前训练状态...")
    finally:
        model.save(latest_path)
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(latest_path + "_replay_buffer.pkl")
        if not forever:
            final_path = model_path + "_final"
            model.save(final_path)
            print(f"\n训练完成！最终模型已保存至 {final_path}.zip")
        else:
            print(f"\n持续训练已停止，最新模型已保存至 {latest_path}.zip")
        train_env.close()
        eval_env.close()


# ======================================================================
# CLI 入口
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="镜头主动对准强化学习训练")
    parser.add_argument("--algo", default="sac", choices=["sac", "td3", "ppo"],
                        help="RL 算法 (default: sac)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="总训练步数 (default: 1_000_000)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")
    parser.add_argument(
        "--resume_from", default=None, metavar="CKPT_PATH",
        help="从指定 checkpoint（.zip）断点续训",
    )
    parser.add_argument(
        "--forever", action="store_true",
        help="持续训练；创建模型路径同名的 .STOP 文件可安全停止并保存",
    )
    parser.add_argument(
        "--chunk-timesteps", type=int, default=10_000,
        help="持续训练每轮步数 (default: 10000)",
    )
    args = parser.parse_args()

    train(
        algo=args.algo,
        total_timesteps=args.timesteps,
        seed=args.seed,
        resume_from=args.resume_from,
        forever=args.forever,
        chunk_timesteps=args.chunk_timesteps,
    )
