"""
4自由度交替对准配置：偏心（dx, dy）+ 倾斜（rx, ry）

运动模式：
    - 奇数步：仅调整偏心（dx, dy），倾斜动作被屏蔽
    - 偶数步：仅调整倾斜（rx, ry），偏心动作被屏蔽

这种交替模式模拟实际对准流程：先粗调位置，再微调角度，逐步收敛。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import LensEnvConfig, LensGroupConfig


def make_4dof_alternating_config(fast_mode: bool = True) -> LensEnvConfig:
    """返回4自由度交替对准的环境配置。

    Args:
        fast_mode: True=快速模式（32条光线），False=标准模式（128条光线）

    配置说明：
        - 激活4个自由度：dx, dy, rx, ry
        - 倾斜范围：±1.0°（init_rx/ry_deg, limit_rx/ry_deg）
        - 倾斜步长：0.05° (step_rx/ry_deg)
        - 偏心范围保持：±0.8mm

    使用方法：
        env_cfg = make_4dof_alternating_config()
        env = AlternatingLensEnv(cfg=env_cfg)  # 使用交替包装器
    """
    lens_groups = [
        LensGroupConfig(
            surf_front=-1,
            surf_rear=-1,
            z_source_surf=-1,
            # 第2组（索引1）激活4自由度，顺序为 [dx, dy, rx, ry]
            active_dofs=["dx", "dy", "rx", "ry"] if group_index == 1 else [],
            # 偏心参数（保持原有配置）
            init_dx_mm=0.8,
            init_dy_mm=0.8,
            step_dx_mm=0.05,
            step_dy_mm=0.05,
            limit_dx_mm=0.8,
            limit_dy_mm=0.8,
            # 倾斜参数（新增，范围±1°）
            init_rx_deg=1.0,   # 初始错位范围 ±1°
            init_ry_deg=1.0,
            step_rx_deg=0.05,  # 单步最大调整 0.05°
            step_ry_deg=0.05,
            limit_rx_deg=1.0,  # 行程上限 ±1°
            limit_ry_deg=1.0,
        )
        for group_index in range(4)
    ]

    return LensEnvConfig(
        lens_groups=lens_groups,
        mtf_field_coords=[(0.0, 0.0)],  # 仅使用中心视场加速训练
        mtf_field_indices=[0],
        mtf_num_rays=32 if fast_mode else 128,
        # 制造公差配置
        tol_radius_rel=0.001,
        tol_thickness_mm=0.010,
        tol_decenter_mm=0.015,
        tol_tilt_deg=0.03,
        tol_lens_decenter_mm=0.015,
        tol_lens_tilt_deg=0.03,
        # 成功标准
        success_threshold=0.05,  # 质量增益 ≥5% 视为成功
        success_bonus=1.0,
        max_episode_steps=100,  # 4自由度需要更多步数
    )


@dataclass
class Alternating4DOFConfig:
    """交替运动模式配置"""
    # 运动模式：'alternating' 或 'simultaneous'
    # - alternating: 奇数步调偏心，偶数步调倾斜
    # - simultaneous: 所有自由度同时调整（标准4D对准）
    motion_mode: str = 'alternating'

    # 偏心自由度索引（在 active_dofs 中的位置）
    decenter_indices: List[int] = field(default_factory=lambda: [0, 1])  # dx, dy

    # 倾斜自由度索引
    tilt_indices: List[int] = field(default_factory=lambda: [2, 3])  # rx, ry

    # 是否在奇数步调整偏心（True）还是倾斜（False）
    # True: 奇数步→偏心，偶数步→倾斜
    # False: 奇数步→倾斜，偶数步→偏心
    decenter_on_odd_steps: bool = True
