"""
主动对准（Active Alignment）Gymnasium 环境。

默认场景：对 Double Gauss 六片镜头的第二个可动镜组进行 2 自由度偏心对准（dx, dy）。
像面（sensor）Z 轴由"补偿器"方法在每步动作后自动优化，不作为 RL 动作维度。

可扩展方式：
  · 4D 对准（偏心+倾斜）：LensGroupConfig(active_dofs=["dx","dy","rx","ry"])
  · 多片对准：在 LensEnvConfig.lens_groups 中追加 LensGroupConfig 即可，
    动作、obs 维度自动适配，无需修改本文件。

观测：
    MTF 曲线（视场 × sag/tang × N 频点）的历史窗口（默认 10 步）。
    使用当前 episode 基线参考的对数比值：
        log((MTF_actual + eps) / (MTF_ref + eps))
    其中 MTF_ref 为注入制造公差后、零对准偏置位置的参考 MTF。

奖励：
    r_t = (q_t - q_{t-1}) + success_bonus × 𝟙[q_t ≥ threshold]

光学建模与 MTF 计算：
    使用 optics_core 加载带 Coordinate Break 的 Double Gauss ZMX 处方。
    对准状态、制造公差、光线追迹和 MTF 均作用于同一个 Double Gauss 模型。
"""
from __future__ import annotations

import sys
import copy
import os
from typing import List
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 添加 optics_core 路径以导入 optics_core 和 zemax_utils
_OPTICS_CORE_PATH = Path(
    os.environ.get(
        "ALIGNMENT_RL_OPTICS_CORE_PATH",
        str(Path(__file__).resolve().parents[2] / "optics_core-dev"),
    )
)
if str(_OPTICS_CORE_PATH) not in sys.path:
    sys.path.insert(0, str(_OPTICS_CORE_PATH))

# 添加 zemax_utils 路径（zemax_utils 是 optics_core-dev 下的独立模块）
_ZEMAX_UTILS_PATH = _OPTICS_CORE_PATH / "zemax_utils"
if _ZEMAX_UTILS_PATH.exists() and str(_ZEMAX_UTILS_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_ZEMAX_UTILS_PATH.parent))

# Use optics_core for MTF calculation
import torch
import optics_core as oc

from config import LensEnvConfig, LensGroupConfig


# DOF 属性映射：name → (init_attr, step_attr, limit_attr)（对应 LensGroupConfig 字段名）
_DOF_ATTRS: dict[str, tuple[str, str, str]] = {
    "dx": ("init_dx_mm",  "step_dx_mm",  "limit_dx_mm"),
    "dy": ("init_dy_mm",  "step_dy_mm",  "limit_dy_mm"),
    "dz": ("init_dz_mm",  "step_dz_mm",  "limit_dz_mm"),
    "rx": ("init_rx_deg", "step_rx_deg", "limit_rx_deg"),
    "ry": ("init_ry_deg", "step_ry_deg", "limit_ry_deg"),
}


# ======================================================================
# Double Gauss 光学系统状态管理器
# ======================================================================

class _LensManager:
    """管理一个由 optics_core 表示的 Double Gauss 六片光学系统。"""

    DOF_NAMES_PER_LENS = ["dx", "dy", "dz", "rx", "ry"]
    DOF_SENSOR = ["sensor_z"]

    def __init__(self, cfg: LensEnvConfig):
        self.cfg = cfg
        unsupported = {
            dof
            for group in cfg.lens_groups
            for dof in group.active_dofs
            if dof not in {"dx", "dy", "rx", "ry"}
        }
        if unsupported:
            raise ValueError(
                "Double Gauss Coordinate Break alignment supports dx/dy/rx/ry; "
                f"unsupported DOFs: {sorted(unsupported)}"
            )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._core_system = None  # optics_core system for MTF computation
        # optics_core 的材料表和准备态与大多数对准动作无关。默认 dx/dy
        # 不改变当前后端使用的轴向 frame，因此可以复用 prepare() 的结果。
        self._core_prepared = False
        self._core_prepare_dirty = True
        self._core_surface_state: dict[int, tuple[float, ...]] = {}
        self._core_cb_pairs: tuple[tuple[int, int], ...] = ()
        self._target_group_indices: set[int] = {
            index for index, group in enumerate(cfg.lens_groups) if group.active_dofs
        }
        self._group_tolerance_states = [
            {dof: 0.0 for dof in self.DOF_NAMES_PER_LENS}
            for _ in cfg.lens_groups
        ]
        self._surface_group_indices: dict[int, int | None] = {}
        self._surface_tolerance_indices: set[int] = set()
        self._group_alignment_states = [
            {dof: 0.0 for dof in self.DOF_NAMES_PER_LENS}
            for _ in cfg.lens_groups
        ]
        self._build_system()

    def _build_system(self):
        """加载 Double Gauss ZMX 并计算标称 MTF。"""
        self._init_core_system()

        # 计算标称 MTF（用于分析 / 调试）
        self._nominal_mtf_obs = self._compute_mtf_obs()
        self._episode_ref_mtf_obs = self._nominal_mtf_obs.copy()

    def _init_core_system(self):
        """从 ZMX 处方初始化 Double Gauss optics_core 系统。"""
        # This prescription contains an entrance/return Coordinate Break pair
        # around the movable cemented group.  Ordinary surface.frame values are
        # intentionally not used: optics_core only propagates CoordinateBreak
        # frames during sequential tracing.
        zmx_path = (
            _OPTICS_CORE_PATH
            / "tests"
            / "zemax"
            / "zmx_files"
            / "Double Gauss 28 degree field with CB.ZMX"
        )

        if zmx_path.exists():
            from zemax_utils import load_zmx_sequential_system_spec, build_optics_core_system_from_zmx_spec
            spec = load_zmx_sequential_system_spec(str(zmx_path))
            base_system = build_optics_core_system_from_zmx_spec(spec)

            # 使用正确的 MultiOpticalSystem 初始化方式
            self._core_system = oc.MultiOpticalSystem(
                architecture=base_system.architecture,
                name=base_system.name,
                parameter_schema=oc.ParameterSchema([]),
                parameters=[{}],
                config=copy.deepcopy(base_system.config),
                tracer=base_system.tracer,
                materials=base_system.materials,
                fields=copy.deepcopy(list(base_system.fields)),
                wavelengths=copy.deepcopy(list(base_system.wavelengths)),
                aperture=copy.deepcopy(base_system.aperture),
            )

            # 设置设备
            if self._device.type != "cpu":
                self._core_system.config.backend.device = str(self._device)

            cb_indices = tuple(
                index
                for index, surface in enumerate(self._core_system.architecture.surfaces)
                if isinstance(surface, oc.CoordinateBreak)
            )
            required_breaks = 2 * len(self.cfg.lens_groups)
            if len(cb_indices) < required_breaks:
                raise ValueError(
                    f"The core prescription has {len(cb_indices)} CoordinateBreak surfaces, "
                    f"but {required_breaks} are required for {len(self.cfg.lens_groups)} movable groups."
                )
            self._core_cb_pairs = tuple(
                (cb_indices[2 * index], cb_indices[2 * index + 1])
                for index in range(len(self.cfg.lens_groups))
            )

            # The bundled ZMX is also a non-zero Zemax regression fixture.
            # Alignment episodes need a zero-pose reference, so retain its
            # topology and return-break order while clearing its sample pose.
            for entrance_index, return_index in self._core_cb_pairs:
                entrance = self._core_system.architecture.surfaces[entrance_index]
                return_break = self._core_system.architecture.surfaces[return_index]
                for surface in (entrance, return_break):
                    surface.frame.x = 0.0
                    surface.frame.y = 0.0
                    surface.frame.rx = 0.0
                    surface.frame.ry = 0.0
                    surface.frame.rz = 0.0
                entrance.order_flag = 0
                return_break.order_flag = 1

            # Parameterize fixed-surface optical tolerances. Coordinate Break
            # frames represent lens-group motion; ordinary surface frames are
            # used for independent surface-level decenter/tilt errors.
            specs = []
            dynamic = {i for pair in self._core_cb_pairs for i in pair}
            excluded = {len(self._core_system.architecture.surfaces) - 1} | set(self.cfg.tol_exclude_surfs) | dynamic
            self._surface_group_indices = {}
            for group_index, (entrance_index, return_index) in enumerate(self._core_cb_pairs):
                for surface_index in range(entrance_index + 1, return_index):
                    self._surface_group_indices[surface_index] = group_index
            self._surface_tolerance_indices = set()
            for i, surface in enumerate(self._core_system.architecture.surfaces):
                if i in dynamic:
                    for attr in ("x", "y", "rx", "ry"):
                        specs.append(oc.ParameterSpec(
                            name=f"surface_{i}_frame_{attr}",
                            path=f"surface[{i}].frame.{attr}",
                            default=float(getattr(surface.frame, attr)),
                        ))
                    continue
                if i in excluded:
                    continue
                # Surface-level errors also apply to the target lens.  Only
                # lens-level rigid-body errors are excluded for target groups;
                # RL must compensate the target's independent face errors.
                self._surface_tolerance_indices.add(i)
                for attr, kind in (("x", "surface_decenter"), ("y", "surface_decenter"), ("rx", "surface_tilt"), ("ry", "surface_tilt")):
                    specs.append(oc.ParameterSpec(
                        name=f"surface_{i}_frame_{attr}",
                        path=f"surface[{i}].frame.{attr}",
                        default=float(getattr(surface.frame, attr)),
                        metadata={"tolerance_kind": kind, "surface_index": i},
                    ))
                radius = getattr(surface.geometry, "radius", None)
                if radius is not None and np.isfinite(float(radius)) and float(radius) != 0.0:
                    specs.append(oc.ParameterSpec(name=f"s{i}_r", path=f"surface[{i}].geometry.radius", default=float(radius), metadata={"tolerance_kind": "radius"}))
                thickness = getattr(surface.gap, "thickness", None)
                if thickness is not None and np.isfinite(float(thickness)):
                    specs.append(oc.ParameterSpec(name=f"s{i}_t", path=f"surface[{i}].gap.thickness", default=float(thickness), metadata={"tolerance_kind": "thickness"}))
            schema = oc.ParameterSchema(specs)
            self._core_system.parameters = oc.ParameterVectorBatch(schema=schema, vectors=[schema.default_vector()], grid_shape=(1,))
            self._core_tolerance_values = {}
            self._core_parameter_indices = {s.name: schema.index_of(s.name) for s in schema}

            self._sync_state_to_core()
        else:
            raise FileNotFoundError(f"无法找到 Double Gauss ZMX 文件: {zmx_path}")

    def _sync_state_to_core(self):
        """Map logical alignment states to paired core Coordinate Breaks."""
        if self._core_system is None:
            return

        core_surfaces = self._core_system.architecture.surfaces
        prepare_dirty = not self._core_prepared
        vector = list(self._core_system.parameter_schema.default_vector())
        for name, value in getattr(self, "_core_tolerance_values", {}).items():
            if name in self._core_parameter_indices:
                vector[self._core_parameter_indices[name]] = value

        # A logical movable group maps explicitly to one entrance/return
        # Coordinate Break pair in the Double Gauss prescription.
        for group_index, (entrance_index, return_index) in enumerate(self._core_cb_pairs):
            values = self._group_alignment_states[group_index]
            manufacturing = self._group_tolerance_states[group_index]
            entrance_state = (
                float(values["dx"] + manufacturing["dx"]),
                float(values["dy"] + manufacturing["dy"]),
                float(values["rx"] + manufacturing["rx"]),
                float(values["ry"] + manufacturing["ry"]),
            )
            return_state = tuple(-value for value in entrance_state)

            for surface_index, state in (
                (entrance_index, entrance_state),
                (return_index, return_state),
            ):
                if self._core_surface_state.get(surface_index) == state:
                    continue
                self._core_surface_state[surface_index] = state
                surface = core_surfaces[surface_index]
                surface.frame.x, surface.frame.y, surface.frame.rx, surface.frame.ry = state
                for attr, value in zip(("x", "y", "rx", "ry"), state):
                    name = f"surface_{surface_index}_frame_{attr}"
                    if name in self._core_parameter_indices:
                        vector[self._core_parameter_indices[name]] = value
                prepare_dirty = True

        self._core_system.parameters.vectors = [vector]
        self._core_prepare_dirty = self._core_prepare_dirty or prepare_dirty

    def _ensure_core_prepared(self) -> None:
        """按需构建 optics_core 准备态，复用静态材料和几何缓存。"""
        if self._core_system is None:
            raise RuntimeError("optics_core system has not been initialized.")
        if not self._core_prepared or self._core_prepare_dirty:
            # ZMX 材料和波长表在 episode/动作期间不变；即使 dz 触发
            # 几何准备态重建，也复用已经编译好的材料张量。
            try:
                self._core_system.prepare(recompute_materials=False)
            except TypeError as exc:
                if "recompute_materials" not in str(exc):
                    raise
                self._core_system.prepare()
            self._core_prepared = True
            self._core_prepare_dirty = False

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------

    def reset_to_nominal(self):
        """恢复 Double Gauss 标称制造参数和零对准状态。"""
        for values in self._group_alignment_states:
            values.update({dof: 0.0 for dof in self.DOF_NAMES_PER_LENS})
        for values in self._group_tolerance_states:
            values.update({dof: 0.0 for dof in self.DOF_NAMES_PER_LENS})
        self._core_tolerance_values = {}
        self._core_prepare_dirty = True
        self._sync_state_to_core()

    def apply_mfg_tolerances(self, rng: np.random.Generator):
        """对非对准面施加制造公差随机扰动（domain randomization）。

        各面公差类型：
          - 曲率半径相对误差（高斯，σ=tol_radius_rel）
          - 厚度误差（高斯，σ=tol_thickness_mm）

        Coordinate Break 和配置排除面不施加公差。
        """
        cfg = self.cfg
        self._core_tolerance_values = {}
        for values in self._group_tolerance_states:
            values.update({dof: 0.0 for dof in self.DOF_NAMES_PER_LENS})
        for group_index, values in enumerate(self._group_tolerance_states):
            if group_index in self._target_group_indices:
                continue
            values["dx"] = float(rng.normal(0.0, cfg.tol_lens_decenter_mm))
            values["dy"] = float(rng.normal(0.0, cfg.tol_lens_decenter_mm))
            values["rx"] = float(rng.normal(0.0, cfg.tol_lens_tilt_deg))
            values["ry"] = float(rng.normal(0.0, cfg.tol_lens_tilt_deg))
            self._core_tolerance_values.update({
                f"lens_group_{group_index}_dx_mm": values["dx"],
                f"lens_group_{group_index}_dy_mm": values["dy"],
                f"lens_group_{group_index}_rx_deg": values["rx"],
                f"lens_group_{group_index}_ry_deg": values["ry"],
            })
        if self._core_system is not None:
            for spec in self._core_system.parameter_schema:
                kind = spec.metadata.get("tolerance_kind")
                if kind == "radius":
                    base = float(spec.default_value())
                    self._core_tolerance_values[spec.name] = float(base + rng.normal(0.0, abs(base) * cfg.tol_radius_rel))
                elif kind == "thickness":
                    base = float(spec.default_value())
                    self._core_tolerance_values[spec.name] = float(base + rng.normal(0.0, cfg.tol_thickness_mm))
                elif kind == "surface_decenter":
                    base = float(spec.default_value())
                    self._core_tolerance_values[spec.name] = float(base + rng.normal(0.0, cfg.tol_decenter_mm))
                elif kind == "surface_tilt":
                    base = float(spec.default_value())
                    self._core_tolerance_values[spec.name] = float(base + rng.normal(0.0, cfg.tol_tilt_deg))
        self._core_prepare_dirty = True
        self._sync_state_to_core()

    def apply_alignment_state(self, state: np.ndarray):
        """将逻辑对准状态写入 Double Gauss Coordinate Break。

        Args:
            state: shape=(sum(len(lg.active_dofs) for lg in cfg.lens_groups),)
                   各 lens_group 的 active_dofs 值依次排列（mm / deg）。
        """
        state = np.asarray(state, dtype=np.float64)
        expected_size = sum(len(lg.active_dofs) for lg in self.cfg.lens_groups)
        if state.shape != (expected_size,):
            raise ValueError(f"alignment state must have shape ({expected_size},), got {state.shape}")

        offset = 0
        for group_index, lg in enumerate(self.cfg.lens_groups):
            active = lg.active_dofs
            n = len(active)
            lg_state = state[offset: offset + n]
            offset += n

            # 将 active_dofs 映射为完整 5D 向量（未激活的 DOF 保持 0）
            vals: dict[str, float] = {d: 0.0 for d in ("dx", "dy", "dz", "rx", "ry")}
            for i, d in enumerate(active):
                vals[d] = float(lg_state[i])
            self._group_alignment_states[group_index] = vals.copy()

        self._sync_state_to_core()

    # ------------------------------------------------------------------
    # 补偿器（自动对焦）
    # ------------------------------------------------------------------

    def _apply_best_focus(self) -> float:
        """在每步动作后寻找最优对焦位置（补偿器接口）。

        当前 Double Gauss MTF 路径不执行显式焦面扫描，本方法保留补偿器接口并
        返回 0.0。需要像面补偿时，应在 optics_core 参数模型中加入 sensor_z。

        若将来替换为需要显式对焦控制的 MTF 后端（如几何 MTF、实测数据拟合），
        可在此实现 sensor_z 扫描逻辑（参考 config.sensor_limit_dz_mm）。

        Returns:
            补偿器施加的 sensor_z 偏移量（mm）；当前始终为 0.0。
        """
        return 0.0

    # ------------------------------------------------------------------
    # MTF 计算
    # ------------------------------------------------------------------

    def set_episode_reference(self):
        """记录当前 tolerance realization 下、零对准偏置位置的参考 MTF。"""
        self._episode_ref_mtf_obs = self._compute_mtf_obs()

    def _relative_log_mtf_obs(self, raw: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """按 reference 计算裁剪后的对数比值观测。"""
        eps = self.cfg.mtf_log_epsilon
        rel = np.log((raw + eps) / (reference + eps))
        return np.clip(rel, -self.cfg.mtf_relative_clip, self.cfg.mtf_relative_clip).astype(np.float32)

    def _compute_mtf_obs(self) -> np.ndarray:
        """计算当前镜头状态的 MTF 观测向量（未归一化）。

        使用 optics_core 进行 MTF 计算。

        Returns:
            shape=(n_fields * 2 * n_freqs,)  sag 和 tang 交替排列
        """
        cfg = self.cfg

        try:
            # 仅首次计算或动态几何真正失效时重建准备态。
            self._ensure_core_prepared()

            # 检查视场数量，确保 field_indices 不超出范围
            n_fields = len(self._core_system.fields)
            valid_field_indices = [fi for fi in cfg.mtf_field_indices if fi < n_fields]

            if not valid_field_indices:
                # 如果没有有效的视场索引，使用第一个视场
                valid_field_indices = [0]

            if len(valid_field_indices) != len(cfg.mtf_field_indices):
                print(f"警告: 部分视场索引超出范围。系统有 {n_fields} 个视场，"
                      f"配置要求 {cfg.mtf_field_indices}，使用 {valid_field_indices}")

            # 配置 MTF 设置
            settings = oc.MTFSettings(
                pupil_sample_count=cfg.mtf_num_rays,
                image_sample_count=cfg.mtf_grid_size if cfg.mtf_grid_size else 64,
                frequencies_lp_per_mm=tuple(cfg.mtf_frequencies),
                field_indices=tuple(valid_field_indices),
                wavelength_index=-1,  # 使用所有波长
            )

            # 计算 MTF
            result = self._core_system.analysis.mtf(settings).run()

            # 提取结果
            sagittal = torch.as_tensor(result.sagittal, dtype=torch.float64).detach().cpu().numpy()
            tangential = torch.as_tensor(result.tangential, dtype=torch.float64).detach().cpu().numpy()

            # 格式化输出：[design=0, field, freq] -> 展平为 [field * (sag+tang) * freq]
            # sagittal/tangential shape: (n_designs=1, n_fields, n_freqs)
            if sagittal.ndim == 3:
                sagittal = sagittal[0]  # 取第一个设计
                tangential = tangential[0]

            result_vec = []
            for fi in range(sagittal.shape[0]):  # 遍历视场
                for ori in range(2):  # 0=sag, 1=tang
                    if ori == 0:
                        vals = np.clip(sagittal[fi], 0.0, 1.0)
                    else:
                        vals = np.clip(tangential[fi], 0.0, 1.0)
                    result_vec.extend(vals.tolist())

            return np.array(result_vec, dtype=np.float32)

        except Exception as e:
            # 计算失败（极度失焦/畸变导致光瞳采样失败）→ 返回零
            print(f"MTF 计算失败: {e}")
            import traceback
            traceback.print_exc()
            n = len(cfg.mtf_field_indices) * 2 * len(cfg.mtf_frequencies)
            return np.zeros(n, dtype=np.float32)

    def get_normalized_mtf_obs(self) -> np.ndarray:
        """返回相对全局标称系统的归一化 MTF（用于分析 / 调试）。"""
        raw = self._compute_mtf_obs()
        denom = np.where(self._nominal_mtf_obs > 1e-6, self._nominal_mtf_obs, 1e-6)
        return np.clip(raw / denom, 0.0, 1.0)

    def get_episode_relative_mtf_obs(self) -> np.ndarray:
        """返回当前 episode 基线相对的 log MTF 观测。"""
        raw = self._compute_mtf_obs()
        return self._relative_log_mtf_obs(raw, self._episode_ref_mtf_obs)

    def quality_metric(self) -> float:
        """计算标量质量指标 q（episode 基线相对 log MTF 增益均值）。"""
        return float(np.mean(self.get_episode_relative_mtf_obs()))


# ======================================================================
# Gymnasium 环境
# ======================================================================

class LensAlignmentEnv(gym.Env):
    """主动对准（Active Alignment）Gymnasium 环境。

    动作空间（连续，每步 [-1,1]^n 内部缩放）：
        各 lens_group 仅包含 active_dofs 中指定的自由度
        总计：sum(len(lg.active_dofs) for lg in lens_groups)
        默认（2D 偏心）：2 维 [dx, dy]

    观测空间（partial 模式）：
        历史 obs_history_len 步的 episode 基线相对 log MTF 向量拼接
        再额外拼接最近 action_history_len 步的归一化动作历史
        shape = (obs_history_len * n_mtf_values + action_history_len * n_action,)
        n_mtf_values = n_fields * 2 * n_frequencies

    补偿器：
        每次 step()/reset() 后自动扫描 sensor_z 以最大化 MTF（use_compensator=True）。
        sensor_z 不在动作空间内，由环境自主优化。

    Info 字段（env.step/reset 返回）：
        quality_metric  : float   当前标量质量指标 q（episode 基线相对 log 增益均值）
        state           : ndarray 对准状态向量（激活 DOF，mm/deg）
        step            : int     当前步数
        success         : bool    是否达成对准目标
        mtf_obs         : ndarray 当前 episode 基线相对 log MTF 观测向量
        raw_mtf_obs     : ndarray 当前原始 MTF 观测向量
        episode_ref_mtf_obs : ndarray 当前 episode 基线参考 MTF 观测向量
        nominal_mtf_obs : ndarray 全局标称系统 MTF 观测向量
        compensator_z   : float   补偿器找到的最优 sensor_z 偏移（mm）
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: LensEnvConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = LensEnvConfig()
        self.cfg = cfg
        self._mgr = _LensManager(cfg)

        self._n_action = sum(len(lg.active_dofs) for lg in cfg.lens_groups)  # 总动作维度

        # 初始化 MTF 维度（实际大小在 _mgr 初始化后更新）
        self._mtf_history_len = cfg.obs_history_len
        self._action_history_len = max(int(cfg.action_history_len), 0)

        # 获取实际 MTF 观测大小（考虑视场验证）
        actual_mtf_size = len(self._mgr._nominal_mtf_obs)
        self._n_mtf = actual_mtf_size
        self._obs_dim = self._mtf_history_len * self._n_mtf + self._action_history_len * self._n_action

        # 动作 / 观测空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_action,), dtype=np.float32
        )
        obs_low = [np.full(self._mtf_history_len * self._n_mtf, -cfg.mtf_relative_clip, dtype=np.float32)]
        obs_high = [np.full(self._mtf_history_len * self._n_mtf, cfg.mtf_relative_clip, dtype=np.float32)]
        if self._action_history_len > 0:
            obs_low.append(np.full(self._action_history_len * self._n_action, -1.0, dtype=np.float32))
            obs_high.append(np.full(self._action_history_len * self._n_action, 1.0, dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.concatenate(obs_low), high=np.concatenate(obs_high), shape=(self._obs_dim,), dtype=np.float32
        )

        # 动作缩放向量（物理单位，mm/deg）
        self._action_scale = self._build_action_scale()
        # 行程边界（±limit）
        self._action_limit = self._build_action_limit()

        # 内部状态
        self._alignment_state: np.ndarray | None = None   # 当前对准状态 [mm/deg]
        self._prev_quality: float = 0.0
        self._step_count: int = 0
        self._mtf_obs_buffer = np.zeros((self._mtf_history_len, self._n_mtf), dtype=np.float32)
        self._action_buffer = np.zeros((self._action_history_len, self._n_action), dtype=np.float32)
        self._np_random: np.random.Generator = np.random.default_rng(None)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _build_action_scale(self) -> np.ndarray:
        """返回每个激活 DOF 的单步物理量上限（mm/deg）。"""
        scales = []
        for lg in self.cfg.lens_groups:
            for d in lg.active_dofs:
                scales.append(getattr(lg, _DOF_ATTRS[d][1]))
        return np.array(scales, dtype=np.float64)

    def _build_action_limit(self) -> np.ndarray:
        """返回每个激活 DOF 的行程上限（±limit，mm/deg）。"""
        limits = []
        for lg in self.cfg.lens_groups:
            for d in lg.active_dofs:
                limits.append(getattr(lg, _DOF_ATTRS[d][2]))
        return np.array(limits, dtype=np.float64)

    def _sample_init_state(self) -> np.ndarray:
        """从均匀分布采样初始错位状态（仅激活 DOF）。"""
        state = []
        for lg in self.cfg.lens_groups:
            for d in lg.active_dofs:
                rng = getattr(lg, _DOF_ATTRS[d][0])
                state.append(self._np_random.uniform(-rng, rng))
        return np.array(state, dtype=np.float64)

    def _reset_history_buffers(self, mtf_obs: np.ndarray) -> None:
        """重置观测历史。

        MTF 历史全部填充第一帧真实观测，避免用 0 人为引入一个并不存在的初始跃迁；
        动作历史则清零，因为 reset 之前确实没有执行过控制动作。
        """
        self._mtf_obs_buffer[...] = mtf_obs.astype(np.float32)
        if self._action_history_len > 0:
            self._action_buffer.fill(0.0)

    def _append_history(self, mtf_obs: np.ndarray, action: np.ndarray) -> None:
        """将最新 MTF 观测与动作写入历史窗口。"""
        self._mtf_obs_buffer = np.roll(self._mtf_obs_buffer, -1, axis=0)
        self._mtf_obs_buffer[-1] = mtf_obs.astype(np.float32)
        if self._action_history_len > 0:
            self._action_buffer = np.roll(self._action_buffer, -1, axis=0)
            self._action_buffer[-1] = action.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        obs_parts = [self._mtf_obs_buffer.flatten()]
        if self._action_history_len > 0:
            obs_parts.append(self._action_buffer.flatten())
        return np.concatenate(obs_parts).astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium 接口
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # 1. 恢复标称状态
        self._mgr.reset_to_nominal()

        # 2. 施加制造公差随机扰动（domain randomization）
        self._mgr.apply_mfg_tolerances(self._np_random)

        # 2b. 记录当前 tolerance realization 下零偏置位置的参考 MTF
        ref_state = np.zeros(self._n_action, dtype=np.float64)
        self._mgr.apply_alignment_state(ref_state)
        self._mgr.set_episode_reference()

        # 3. Keep the tolerance realization and zero-pose reference fixed,
        # while resampling an initial state outside the success region.
        ceiling = self.cfg.initial_quality_ceiling
        attempts = max(int(self.cfg.initial_quality_sampling_attempts), 1)
        for _ in range(attempts):
            self._alignment_state = self._sample_init_state()
            self._mgr.apply_alignment_state(self._alignment_state)
            _comp_z = self._mgr._apply_best_focus() if self.cfg.use_compensator else 0.0
            raw_mtf_obs = self._mgr._compute_mtf_obs()
            mtf_obs = self._mgr._relative_log_mtf_obs(raw_mtf_obs, self._mgr._episode_ref_mtf_obs)
            q = float(np.mean(mtf_obs))
            if ceiling is None or q < ceiling:
                break
        if ceiling is not None and q >= ceiling:
            # Some tolerance realizations have q=0 as the sampled map's
            # minimum.  The zero-pose reference is always a valid, non-success
            # starting point and avoids retaining an already successful pose.
            self._alignment_state = np.zeros(self._n_action, dtype=np.float64)
            self._mgr.apply_alignment_state(self._alignment_state)
            _comp_z = self._mgr._apply_best_focus() if self.cfg.use_compensator else 0.0
            raw_mtf_obs = self._mgr._compute_mtf_obs()
            mtf_obs = self._mgr._relative_log_mtf_obs(raw_mtf_obs, self._mgr._episode_ref_mtf_obs)
            q = float(np.mean(mtf_obs))
        self._prev_quality = q

        # 5. 初始化历史观测 buffer（全部填充第一帧）
        self._reset_history_buffers(mtf_obs)

        self._step_count = 0

        info = {
            "quality_metric": q,
            "state": self._alignment_state.copy(),
            "step": 0,
            "success": q >= self.cfg.success_threshold,
            "mtf_obs": mtf_obs,
            "raw_mtf_obs": raw_mtf_obs,
            "episode_ref_mtf_obs": self._mgr._episode_ref_mtf_obs.copy(),
            "nominal_mtf_obs": self._mgr._nominal_mtf_obs.copy(),
            "core_tolerances": dict(self._mgr._core_tolerance_values),
            "compensator_z": _comp_z,
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        assert self._alignment_state is not None, "Call reset() before step()"

        # 1. 动作解码：[-1,1] → 物理量（mm/deg），并裁剪到行程边界
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        delta = action * self._action_scale
        new_state = np.clip(
            self._alignment_state + delta,
            -self._action_limit,
            self._action_limit,
        )
        self._alignment_state = new_state

        # 2. 写入镜面
        self._mgr.apply_alignment_state(self._alignment_state)

        # 2b. 补偿器：自动优化像面对焦位置
        _comp_z = self._mgr._apply_best_focus() if self.cfg.use_compensator else 0.0

        # 3. 计算新质量
        raw_mtf_obs = self._mgr._compute_mtf_obs()
        mtf_obs = self._mgr._relative_log_mtf_obs(raw_mtf_obs, self._mgr._episode_ref_mtf_obs)
        q = float(np.mean(mtf_obs))

        # 4. 计算奖励
        terminated = bool(q >= self.cfg.success_threshold)
        reward = float(
            (q - self._prev_quality)
            + self.cfg.success_bonus * float(terminated)
        )
        self._prev_quality = q
        self._step_count += 1

        # 5. 更新历史 buffer（滑动窗口，旧帧前移，最新帧追加到末尾）
        self._append_history(mtf_obs, action)

        truncated = self._step_count >= self.cfg.max_episode_steps

        info = {
            "quality_metric": q,
            "state": self._alignment_state.copy(),
            "step": self._step_count,
            "success": terminated,
            "mtf_obs": mtf_obs,
            "raw_mtf_obs": raw_mtf_obs,
            "episode_ref_mtf_obs": self._mgr._episode_ref_mtf_obs.copy(),
            "nominal_mtf_obs": self._mgr._nominal_mtf_obs.copy(),
            "core_tolerances": dict(self._mgr._core_tolerance_values),
            "compensator_z": _comp_z,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass  # 可视化由 visualize_episode.py 单独处理

    def close(self):
        pass

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @property
    def n_dof(self) -> int:
        return self._n_action

    @property
    def dof_names(self) -> List[str]:
        names = []
        for i, lg in enumerate(self.cfg.lens_groups):
            tag = f"L{i+1}"
            names.extend(f"{tag}_{d}" for d in lg.active_dofs)
        return names

    def nominal_mtf_obs(self) -> np.ndarray:
        """返回标称（完美对准）MTF 观测向量（未归一化原始值）。"""
        return self._mgr._nominal_mtf_obs.copy()

    def episode_reference_mtf_obs(self) -> np.ndarray:
        """返回当前 episode 基线参考 MTF 观测向量（未归一化原始值）。"""
        return self._mgr._episode_ref_mtf_obs.copy()
