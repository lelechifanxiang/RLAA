"""
全局配置：主动对准（LensAlignmentEnv）配置。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Tuple


@dataclass
class TrainingConfig:
    """强化学习训练超参数。"""
    algorithm: Literal['sac', 'td3', 'ppo'] = 'sac'
    total_timesteps: int = 1_000_000

    # SAC / TD3 公共参数
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 300_000
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = 'auto'              # SAC 自动调节熵系数

    # PPO 专用
    n_envs: int = 8
    n_steps: int = 2048

    # 策略网络隐藏层
    net_arch: list = field(default_factory=lambda: [256, 256])

    # 日志与模型保存
    log_dir: str = 'logs/'
    model_dir: str = 'models/'
    eval_freq: int = 10_000
    n_eval_episodes: int = 50
    seed: int = 42


# ── 默认实例（直接 import 使用） ──────────────────────────────
train_cfg = TrainingConfig()


# ======================================================================
# 主动对准（Active Alignment）配置
# ======================================================================

@dataclass
class LensGroupConfig:
    """Double Gauss 中一个 Coordinate Break 镜组的对准自由度配置。

    ``surf_front/surf_rear/z_source_surf`` 保留用于报告和旧结果元数据；
    实际运行时，镜组按其在 ZMX 中的 Coordinate Break pair 顺序映射。
    """
    surf_front: int          # 报告中的入口 Coordinate Break 序号
    surf_rear: int           # 报告中的返回 Coordinate Break 序号
    z_source_surf: int       # 轴向运动的报告字段（当前 Double Gauss 保留接口）

    # 激活的对准自由度（列表中的顺序即动作向量顺序）
    # 当前 Double Gauss Coordinate Break 支持：["dx", "dy", "rx", "ry"]
    #   2D 偏心对准：["dx", "dy"]
    #   4D 偏心+倾斜：["dx", "dy", "rx", "ry"]
    active_dofs: List[str] = field(default_factory=lambda: ["dx", "dy"])

    # 初始错位范围（均匀分布 ±range）
    init_dx_mm: float = 0.3          # x 偏心范围 [mm]
    init_dy_mm: float = 0.3          # y 偏心范围 [mm]
    init_dz_mm: float = 0.3          # z 轴向偏移范围 [mm]
    init_rx_deg: float = 0.5         # 绕 x 轴倾斜范围 [deg]
    init_ry_deg: float = 0.5         # 绕 y 轴倾斜范围 [deg]

    # 每步最大调整量
    step_dx_mm: float = 0.05         # 单步 x 偏心上限 [mm]
    step_dy_mm: float = 0.05         # 单步 y 偏心上限 [mm]
    step_dz_mm: float = 0.05         # 单步 z 轴向上限 [mm]
    step_rx_deg: float = 0.02        # 单步 x 倾斜上限 [deg]
    step_ry_deg: float = 0.02        # 单步 y 倾斜上限 [deg]

    # 调整行程上限（±limit）
    limit_dx_mm: float = 0.8
    limit_dy_mm: float = 0.8
    limit_dz_mm: float = 0.8
    limit_rx_deg: float = 1.5
    limit_ry_deg: float = 1.5


@dataclass
class LensEnvConfig:
    """镜头主动对准 Gymnasium 环境参数。

    扩展方式：修改 lens_groups 中各 LensGroupConfig 的 active_dofs 或追加条目
    即可自动调整动作维度，无需修改环境代码。
    例：
        2D 偏心对准（默认）→ lens_groups=[LGC(3,4,2)] → 2D 动作
        4D 偏心+倾斜        → active_dofs=["dx","dy","rx","ry"]
        多镜组对准          → 为多个 Coordinate Break group 配置 active_dofs
    """
    # 被对准的镜片组列表（可扩展）
    # Keep all four coordinate-break pairs in the imported prescription, but
    # expose only the most sensitive second group to the RL action space.
    # The inactive groups remain fixed at zero alignment state.
    lens_groups: List[LensGroupConfig] = field(
        default_factory=lambda: [
            LensGroupConfig(surf_front=-1, surf_rear=-1, z_source_surf=-1, active_dofs=[]),
            LensGroupConfig(surf_front=-1, surf_rear=-1, z_source_surf=-1, active_dofs=["dx", "dy"]),
            LensGroupConfig(surf_front=-1, surf_rear=-1, z_source_surf=-1, active_dofs=[]),
            LensGroupConfig(surf_front=-1, surf_rear=-1, z_source_surf=-1, active_dofs=[]),
        ]
    )

    # Sensor（像面）配置——由补偿器自动对焦，不作为 RL 动作维度
    sensor_surf: int = 22            # imported Double-Gauss image surface
    sensor_limit_dz_mm: float = 3.0  # 补偿器对焦搜索范围 ±limit [mm]（供扩展用）

    # 补偿器接口：当前 Double Gauss MTF 路径不执行显式焦面扫描。
    use_compensator: bool = True     # 预留：供替换为需显式对焦控制的 MTF 后端使用

    # MTF 观测配置：5 个旋转对称视场 × sag/tang × N 频点
    # 视场坐标使用 ZMX 处方的角度视场定义，格式为 (x, y) [deg]
    # 默认：中心场 + X/Y 方向 ±14° 共 5 个视场
    mtf_field_coords: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.0, 0.0),
            (14.0, 0.0),
            (-14.0, 0.0),
            (0.0, 14.0),
            (0.0, -14.0),
        ]
    )
    mtf_frequencies: List[float] = field(
        default_factory=lambda: [20.0, 30.0, 50.0]  # [lp/mm]
    )
    # 视场序号（对应 mtf_field_coords / lens.fields.fields 的索引）
    mtf_field_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    obs_history_len: int = 10        # 历史窗口步数
    action_history_len: int = 10     # 历史动作窗口步数（策略空间 [-1, 1]；0 表示不拼接动作历史）

    # 质量指标阈值（episode 基线相对 log 增益均值 ≥ threshold 视为对准成功）
    # q=0 对应达到当前 episode 的零偏置参考性能；考虑数值误差，
    # 默认允许轻微负裕量，避免把“回到参考附近”误判为失败。
    success_threshold: float = 0.05
    success_bonus: float = 5.0

    # Optional reset-state filter.  A sampled initial state is accepted only
    # when q is below this ceiling, leaving a finite margin to the success q.
    initial_quality_ceiling: float | None = None
    initial_quality_sampling_attempts: int = 8

    max_episode_steps: int = 100

    # 制造公差（用于 domain randomization，每 episode reset 时随机施加）
    # 对非对准元件的面，施加以下尺度的高斯扰动
    tol_radius_rel: float = 0.001    # 曲率半径相对误差标准差（0.1%）
    tol_thickness_mm: float = 0.010  # 厚度公差标准差 [mm]
    # Surface-level manufacturing errors (independent per optical surface).
    tol_decenter_mm: float = 0.015
    tol_tilt_deg: float = 0.03
    # Lens-group assembly errors (common rigid-body error for each non-target
    # Coordinate Break group). Target groups keep only their face-level errors.
    tol_lens_decenter_mm: float = 0.015
    tol_lens_tilt_deg: float = 0.03
    # 不施加制造公差的面（0=物面, 最后面=像面）
    tol_exclude_surfs: List[int] = field(default_factory=lambda: [0, 22])

    # MTF 计算精度（影响速度）
    # num_rays: 射线数量（越大越精确，越慢）。12 ≈ 30ms/call，20 ≈ 34ms/call
    # grid_size: FFT 网格大小（64/128/256），对速度影响较小
    mtf_num_rays: int = 128
    mtf_grid_size: int = None

    # 观测与奖励使用 episode 基线相对 log MTF：log((actual + eps) / (ref + eps))
    mtf_log_epsilon: float = 1e-6
    mtf_relative_clip: float = 2.0


# 默认实例（Lens 2 单片 2D 偏心对准，含像面自动补偿器）
lens_env_cfg = LensEnvConfig()


def make_lens_rl_config(fast_mode: bool = True) -> LensEnvConfig:
    """返回用于 RL 训练/评估的轻量镜头环境配置。

    Args:
        fast_mode: True=快速模式（32条光线），False=标准模式（128条光线）

    光线数选择说明（实测RTX 5060 Ti）：
    - 32条光线：1024 rays/场，~610MB显存，~2.1 steps/sec（最快，用于训练）
    - 64条光线：4096 rays/场，~2.3GB显存，~1.8 steps/sec（慢16%，计算量4倍）
    - 128条光线：16384 rays/场，高精度，用于最终评估

    实测表明GPU并行效率已达85%，瓶颈在ray tracing和Python overhead。
    训练时应使用32光线以最大化速度。

    改进点（第11次训练）：
    - 扩大初始范围：0.5mm → 0.8mm（覆盖完整动作空间）
    - 降低成功奖励：5.0 → 1.0（平衡质量改进与成功奖励）
    """
    # 扩大初始偏移范围到limit（±0.8mm），强制策略学习长距离对准
    lens_groups = [
        LensGroupConfig(
            surf_front=-1,
            surf_rear=-1,
            z_source_surf=-1,
            active_dofs=["dx", "dy"] if group_index == 1 else [],
            init_dx_mm=0.8,  # 改进：0.5 → 0.8mm
            init_dy_mm=0.8,  # 改进：0.5 → 0.8mm
        )
        for group_index in range(4)
    ]

    return LensEnvConfig(
        lens_groups=lens_groups,
        mtf_field_coords=[(0.0, 0.0)],
        mtf_field_indices=[0],
        mtf_num_rays=32 if fast_mode else 128,
        tol_radius_rel=0.001,
        tol_thickness_mm=0.010,
        tol_decenter_mm=0.015,
        tol_tilt_deg=0.03,
        tol_lens_decenter_mm=0.015,
        tol_lens_tilt_deg=0.03,
        # Require a 5% geometric-mean MTF gain over the episode's zero-pose reference.
        success_threshold=0.05,
        success_bonus=1.0,  # 改进：降低成功奖励，平衡质量改进信号
        # Disable initial-quality filtering: accept the first sampled pose.
        # initial_quality_ceiling=0.005,
        # initial_quality_sampling_attempts=8,
        max_episode_steps=50,   # 减少episode长度加速训练
    )
