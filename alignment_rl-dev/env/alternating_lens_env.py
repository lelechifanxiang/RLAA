"""
交替运动模式的镜头对准环境包装器。

实现策略：
    - 奇数步（step_count为奇数）：仅执行偏心调整（dx, dy），屏蔽倾斜动作
    - 偶数步（step_count为偶数）：仅执行倾斜调整（rx, ry），屏蔽偏心动作

这种模式模拟实际主动对准工艺流程：
    1. 先调整透镜位置（偏心），粗调光轴
    2. 再调整透镜角度（倾斜），精调成像质量
    3. 交替进行，逐步收敛到最优状态

相比同时调整4自由度，交替模式：
    - 优点：降低动作空间复杂度，更接近实际工艺，可能更容易学习
    - 缺点：收敛速度可能较慢，需要更多步数
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.lens_env import LensAlignmentEnv
from config import LensEnvConfig
from config_4dof import Alternating4DOFConfig


class AlternatingLensEnv(gym.Wrapper):
    """交替运动模式的4自由度对准环境包装器。

    动作空间：
        仍然是4维 [-1, 1]^4，对应 [dx, dy, rx, ry]
        但在每步执行时，根据步数奇偶性屏蔽一半的动作：
        - 奇数步：仅执行 action[0:2]（dx, dy），action[2:4] 被置零
        - 偶数步：仅执行 action[2:4]（rx, ry），action[0:2] 被置零

    观测空间：
        与基础环境相同（MTF历史 + 动作历史）

    Info字段：
        新增 'active_dofs': str  # 当前步激活的自由度 'decenter' 或 'tilt'
    """

    def __init__(
        self,
        cfg: LensEnvConfig,
        alternating_cfg: Alternating4DOFConfig | None = None,
    ):
        """
        Args:
            cfg: 镜头环境配置（需要包含4自由度：dx, dy, rx, ry）
            alternating_cfg: 交替模式配置（默认使用标准配置）
        """
        # 验证配置：至少一个镜组需要4个自由度
        active_dofs_list = [lg.active_dofs for lg in cfg.lens_groups if lg.active_dofs]
        if not active_dofs_list:
            raise ValueError("至少需要一个镜组有激活的自由度")

        self._all_active_dofs = active_dofs_list[0]  # 取第一个激活镜组的DOF列表
        if len(self._all_active_dofs) != 4:
            raise ValueError(
                f"交替模式需要4自由度 ['dx', 'dy', 'rx', 'ry']，"
                f"当前配置为 {self._all_active_dofs}"
            )

        # 创建基础环境
        base_env = LensAlignmentEnv(cfg=cfg)
        super().__init__(base_env)

        # 交替模式配置
        if alternating_cfg is None:
            alternating_cfg = Alternating4DOFConfig()
        self.alt_cfg = alternating_cfg

        # 验证索引配置
        n_action = base_env.action_space.shape[0]
        if n_action != 4:
            raise ValueError(f"基础环境动作维度应为4，实际为 {n_action}")

        for idx in self.alt_cfg.decenter_indices + self.alt_cfg.tilt_indices:
            if idx >= n_action:
                raise ValueError(f"自由度索引 {idx} 超出动作空间范围 [0, {n_action})")

        # 内部状态
        self._step_count = 0

        # 用于统计的累计信息
        self._cumulative_decenter_action = np.zeros(2, dtype=np.float32)
        self._cumulative_tilt_action = np.zeros(2, dtype=np.float32)

    def _get_active_mode(self) -> str:
        """返回当前步激活的模式：'decenter' 或 'tilt'"""
        if self.alt_cfg.motion_mode != 'alternating':
            return 'simultaneous'

        is_odd_step = (self._step_count % 2) == 1
        if self.alt_cfg.decenter_on_odd_steps:
            return 'decenter' if is_odd_step else 'tilt'
        else:
            return 'tilt' if is_odd_step else 'decenter'

    def _mask_action(self, action: np.ndarray) -> np.ndarray:
        """根据当前步数对动作进行屏蔽。

        Args:
            action: 原始动作 [dx, dy, rx, ry]

        Returns:
            屏蔽后的动作（被屏蔽的维度置为0）
        """
        if self.alt_cfg.motion_mode != 'alternating':
            return action  # 同时模式：不屏蔽

        action = np.asarray(action, dtype=np.float64)
        masked_action = action.copy()

        active_mode = self._get_active_mode()

        if active_mode == 'decenter':
            # 偏心模式：屏蔽倾斜动作
            for idx in self.alt_cfg.tilt_indices:
                masked_action[idx] = 0.0
        elif active_mode == 'tilt':
            # 倾斜模式：屏蔽偏心动作
            for idx in self.alt_cfg.decenter_indices:
                masked_action[idx] = 0.0

        return masked_action

    def reset(self, **kwargs):
        """重置环境并初始化步数计数器。"""
        self._step_count = 0
        self._cumulative_decenter_action = np.zeros(2, dtype=np.float32)
        self._cumulative_tilt_action = np.zeros(2, dtype=np.float32)

        obs, info = self.env.reset(**kwargs)

        # 添加交替模式信息
        info['active_dofs'] = self._get_active_mode()
        info['step_count'] = self._step_count
        info['motion_mode'] = self.alt_cfg.motion_mode

        return obs, info

    def step(self, action):
        """执行一步，应用动作屏蔽。"""
        self._step_count += 1

        # 记录原始动作
        original_action = np.asarray(action).copy()

        # 屏蔽动作
        masked_action = self._mask_action(action)

        # 累计统计
        active_mode = self._get_active_mode()
        if active_mode == 'decenter':
            self._cumulative_decenter_action += np.abs(masked_action[self.alt_cfg.decenter_indices])
        elif active_mode == 'tilt':
            self._cumulative_tilt_action += np.abs(masked_action[self.alt_cfg.tilt_indices])

        # 执行屏蔽后的动作
        obs, reward, terminated, truncated, info = self.env.step(masked_action)

        # 添加交替模式信息到info
        info['active_dofs'] = active_mode
        info['step_count'] = self._step_count
        info['original_action'] = original_action
        info['masked_action'] = masked_action
        info['motion_mode'] = self.alt_cfg.motion_mode

        # 添加累计动作统计
        info['cumulative_decenter_action'] = self._cumulative_decenter_action.copy()
        info['cumulative_tilt_action'] = self._cumulative_tilt_action.copy()

        return obs, reward, terminated, truncated, info


class AlternatingBatchLensEnv:
    """批量交替运动环境（用于并行训练）。

    包装 BatchLensAlignmentVecEnv，为每个逻辑环境独立应用交替屏蔽。
    """

    def __init__(
        self,
        cfg: LensEnvConfig,
        n_envs: int = 4,
        seed: int = 0,
        alternating_cfg: Alternating4DOFConfig | None = None,
    ):
        """
        Args:
            cfg: 镜头环境配置
            n_envs: 并行环境数量
            seed: 随机种子
            alternating_cfg: 交替模式配置
        """
        from env.batch_lens_env import BatchLensAlignmentVecEnv

        self._base_env = BatchLensAlignmentVecEnv(cfg=cfg, n_envs=n_envs, seed=seed)

        if alternating_cfg is None:
            alternating_cfg = Alternating4DOFConfig()
        self.alt_cfg = alternating_cfg

        # 每个逻辑环境的步数计数器
        self._step_counts = np.zeros(n_envs, dtype=np.int32)

        # 验证动作空间
        n_action = self._base_env.action_space.shape[0]
        if n_action != 4:
            raise ValueError(f"基础环境动作维度应为4，实际为 {n_action}")

        # 转发属性
        self.observation_space = self._base_env.observation_space
        self.action_space = self._base_env.action_space
        self.num_envs = n_envs

    def _get_active_modes(self) -> np.ndarray:
        """返回所有环境的激活模式（0=偏心, 1=倾斜）"""
        if self.alt_cfg.motion_mode != 'alternating':
            return np.full(self.num_envs, -1, dtype=np.int32)  # -1表示同时模式

        is_odd_steps = (self._step_counts % 2) == 1
        if self.alt_cfg.decenter_on_odd_steps:
            return is_odd_steps.astype(np.int32) * 0  # 奇数→0(偏心), 偶数→1(倾斜)
        else:
            return is_odd_steps.astype(np.int32)  # 奇数→1(倾斜), 偶数→0(偏心)

    def _mask_actions(self, actions: np.ndarray) -> np.ndarray:
        """批量屏蔽动作。

        Args:
            actions: shape=(n_envs, 4)

        Returns:
            屏蔽后的动作
        """
        if self.alt_cfg.motion_mode != 'alternating':
            return actions

        actions = np.asarray(actions, dtype=np.float64)
        masked_actions = actions.copy()

        is_odd_steps = (self._step_counts % 2) == 1

        for env_idx in range(self.num_envs):
            if self.alt_cfg.decenter_on_odd_steps:
                active_decenter = is_odd_steps[env_idx]
            else:
                active_decenter = not is_odd_steps[env_idx]

            if active_decenter:
                # 屏蔽倾斜
                for idx in self.alt_cfg.tilt_indices:
                    masked_actions[env_idx, idx] = 0.0
            else:
                # 屏蔽偏心
                for idx in self.alt_cfg.decenter_indices:
                    masked_actions[env_idx, idx] = 0.0

        return masked_actions

    def reset(self):
        """重置所有环境。"""
        self._step_counts[:] = 0
        return self._base_env.reset()

    def step(self, actions):
        """执行一批动作。"""
        self._step_counts += 1

        # 屏蔽动作
        masked_actions = self._mask_actions(actions)

        # 执行
        return self._base_env.step(masked_actions)

    def close(self):
        """关闭环境。"""
        self._base_env.close()

    def __getattr__(self, name):
        """转发未定义的属性到基础环境。"""
        return getattr(self._base_env, name)
