# %%
"""
LensAlignmentEnv 基础正确性测试脚本。

运行方式：
    python test.py          # 运行全部测试
    python test.py -k T01   # 只运行某一个测试
    python test.py --plots -k T13  # 运行人工检查型绘图测试

测试覆盖：
    T01  空间维度与数据类型（默认 2D 偏心对准）
    T02  reset 固定种子的可复现性
    T03  零动作时状态不变
    T04  完美对准时质量 ≈ 1.0 且归一化 MTF 在 [0,1]
    T05  往 Y 方向偏心后质量低于完美对准
    T06  行程限制：超限动作被截断，状态不超界
    T07  episode 正常终止（达到 max_episode_steps）
    T08  奖励函数：Δq + success_bonus 的数值一致性
    T09  补偿器有效：启用后质量高于禁用
    T10  Gymnasium check_env 通过
    T11  4D 扩展（偏心+倾斜）维度正确且 check_env
    T12  双片 4D 对准维度正确且 check_env
    T13  2D 偏心-质量伪彩色图（人工检查单峰性）
    T14  指定偏心点的离焦 MTF 曲线（人工检查离焦规律）
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env
import argparse
import matplotlib.pyplot as plt

from config import LensEnvConfig, LensGroupConfig
import env.lens_env as el


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
# %%


def _assert(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)


# ======================================================================
# T01 空间维度与数据类型（默认 2D 偏心对准）
# ======================================================================
def t01_spaces():
    env = el.LensAlignmentEnv()
    cfg = env.cfg
    # 默认：1 个 lens_group，active_dofs=["dx","dy"] → 2D
    expected_act = sum(len(lg.active_dofs) for lg in cfg.lens_groups)
    expected_obs = (
        cfg.obs_history_len * len(env._mgr._nominal_mtf_obs)
        + cfg.action_history_len * expected_act
    )

    _assert(env.action_space.shape == (expected_act,),
            f"action_space.shape={env.action_space.shape}, expected ({expected_act},)")
    _assert(env.observation_space.shape == (expected_obs,),
            f"obs_space.shape={env.observation_space.shape}, expected ({expected_obs},)")
    _assert(env.action_space.dtype == np.float32, "action_space dtype should be float32")
    _assert(env.observation_space.dtype == np.float32, "obs_space dtype should be float32")
    _assert(len(env.dof_names) == expected_act,
            f"dof_names length={len(env.dof_names)}, expected {expected_act}")
    _assert(env.dof_names == ["L2_dx", "L2_dy"],
            f"dof_names={env.dof_names}, expected ['L2_dx', 'L2_dy']")
    env.close()


# ======================================================================
# T02 reset 固定种子可复现性
# ======================================================================
def t02_reset_reproducibility():
    env = el.LensAlignmentEnv()
    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)
    _assert(np.allclose(obs1, obs2, atol=1e-6),
            "Same seed should produce identical obs")
    _assert(np.allclose(info1["state"], info2["state"], atol=1e-9),
            "Same seed should produce identical initial state")

    obs3, _ = env.reset(seed=999)
    _assert(not np.allclose(obs1, obs3, atol=1e-6),
            "Different seeds should produce different obs")
    env.close()


# ======================================================================
# T03 零动作不改变对准状态（禁用公差，禁用补偿器简化验证）
# ======================================================================
def t03_zero_action_no_state_change():
    cfg = LensEnvConfig(
        tol_radius_rel=0.0, tol_thickness_mm=0.0,
        tol_decenter_mm=0.0, tol_tilt_deg=0.0,
        use_compensator=False,
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    env.reset(seed=0)

    state_before = env._alignment_state.copy()
    zero_action = np.zeros(env.n_dof, dtype=np.float32)
    env.step(zero_action)
    state_after = env._alignment_state.copy()

    _assert(np.allclose(state_before, state_after, atol=1e-12),
            f"Zero action should not change state. Δstate={state_after - state_before}")
    env.close()


# ======================================================================
# T04 完美对准时质量 ≈ 1.0，MTF 归一化在 [0,1]（无公差，无补偿器）
# ======================================================================
def t04_perfect_alignment_quality():
    cfg = LensEnvConfig(
        tol_radius_rel=0.0, tol_thickness_mm=0.0,
        tol_decenter_mm=0.0, tol_tilt_deg=0.0,
        use_compensator=False,
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    env.reset(seed=0)

    # 强制设为零错位（完美对准）
    perfect_state = np.zeros(env.n_dof, dtype=np.float64)
    env._alignment_state = perfect_state.copy()
    env._mgr.apply_alignment_state(perfect_state)

    mtf_obs = env._mgr.get_normalized_mtf_obs()
    q = float(np.mean(mtf_obs))

    _assert(0.0 <= mtf_obs.min() and mtf_obs.max() <= 1.0 + 1e-6,
            f"Normalized MTF should be in [0,1], got [{mtf_obs.min():.4f}, {mtf_obs.max():.4f}]")
    _assert(q > 0.95, f"Perfect alignment quality should be > 0.95, got {q:.4f}")
    env.close()


# ======================================================================
# T05 Y 偏心灵敏度：偏心后质量低于完美对准（无公差，无补偿器）
# ======================================================================
def t05_sensitivity_to_decenter():
    cfg = LensEnvConfig(
        tol_radius_rel=0.0, tol_thickness_mm=0.0,
        tol_decenter_mm=0.0, tol_tilt_deg=0.0,
        use_compensator=False,
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    env.reset(seed=0)

    perfect = np.zeros(env.n_dof, dtype=np.float64)
    env._alignment_state = perfect.copy()
    env._mgr.apply_alignment_state(perfect)
    q_perfect = env._mgr.quality_metric()

    # dy = +0.3 mm（state 索引 1 = dy）
    state_pos = perfect.copy()
    state_pos[1] = 0.3
    env._alignment_state = state_pos.copy()
    env._mgr.apply_alignment_state(state_pos)
    q_pos = env._mgr.quality_metric()

    # dy = -0.3 mm
    state_neg = perfect.copy()
    state_neg[1] = -0.3
    env._alignment_state = state_neg.copy()
    env._mgr.apply_alignment_state(state_neg)
    q_neg = env._mgr.quality_metric()

    _assert(q_perfect > q_pos,
            f"Perfect ({q_perfect:.4f}) should be > dy=+0.3mm ({q_pos:.4f})")
    _assert(q_perfect > q_neg,
            f"Perfect ({q_perfect:.4f}) should be > dy=-0.3mm ({q_neg:.4f})")
    env.close()


# ======================================================================
# T06 行程限制：超限动作被截断
# ======================================================================
def t06_action_clipping():
    cfg = LensEnvConfig(
        tol_radius_rel=0.0, tol_thickness_mm=0.0,
        tol_decenter_mm=0.0, tol_tilt_deg=0.0,
        use_compensator=False,
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    env.reset(seed=0)

    huge_action = np.ones(env.n_dof, dtype=np.float32)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(huge_action)
        if terminated or truncated:
            env.reset(seed=0)

    state = env._alignment_state
    limits = env._action_limit

    _assert(np.all(state <= limits + 1e-9),
            f"State exceeded upper limit.\nState:  {state}\nLimits: {limits}")
    _assert(np.all(state >= -limits - 1e-9),
            f"State exceeded lower limit.\nState:  {state}\nLimits: {limits}")
    env.close()


# ======================================================================
# T07 episode 截断（超过 max_episode_steps）
# ======================================================================
def t07_truncation():
    cfg = LensEnvConfig(
        max_episode_steps=10,
        success_threshold=2.0,          # 永不触发 terminated
        use_compensator=False,
        tol_radius_rel=0.0, tol_thickness_mm=0.0,
        tol_decenter_mm=0.0, tol_tilt_deg=0.0,
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    env.reset(seed=0)

    n_steps = 0
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(np.zeros(env.n_dof, dtype=np.float32))
        n_steps += 1
        _assert(n_steps <= 20, "Episode should truncate within max_episode_steps")

    _assert(n_steps == 10, f"Episode should truncate at step 10, got {n_steps}")
    _assert(info["step"] == 10, f"info['step'] should be 10, got {info['step']}")
    env.close()


# ======================================================================
# T08 奖励数值一致性：r == Δq (+ success_bonus 当 terminated)
# ======================================================================
def t08_reward_consistency():
    cfg = LensEnvConfig(
        use_compensator=False,
        tol_radius_rel=0.0, tol_thickness_mm=0.0,
        tol_decenter_mm=0.0, tol_tilt_deg=0.0,
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    _, init_info = env.reset(seed=42)
    q_prev = init_info["quality_metric"]

    for _ in range(5):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        q_curr = info["quality_metric"]
        expected_reward = (q_curr - q_prev) + cfg.success_bonus * float(terminated)
        _assert(abs(reward - expected_reward) < 1e-6,
                f"reward={reward:.6f}, expected={expected_reward:.6f} "
                f"(Δq={q_curr-q_prev:.6f}, terminated={terminated})")
        q_prev = q_curr
        if terminated or truncated:
            break

    env.close()


# ======================================================================
# T09 补偿器接口：info 中包含 compensator_z 字段
#     （当前 optics_core 路径尚未启用显式焦面扫描，始终为 0.0）
# ======================================================================
def t09_compensator_interface():
    env = el.LensAlignmentEnv()
    _, info_reset = env.reset(seed=0)
    _assert("compensator_z" in info_reset,
            "reset info must contain 'compensator_z' key")
    _assert(isinstance(info_reset["compensator_z"], float),
            "compensator_z must be a float")

    action = np.zeros(env.n_dof, dtype=np.float32)
    _, _, _, _, info_step = env.step(action)
    _assert("compensator_z" in info_step,
            "step info must contain 'compensator_z' key")
    env.close()


# ======================================================================
# T10 Gymnasium check_env
# ======================================================================
def t10_gymnasium_check():
    env = el.LensAlignmentEnv()
    check_env(env, warn=True)
    env.close()


# ======================================================================
# T13 2D 偏心景观图（人工检查：理论上应接近单峰）
# ======================================================================
def t13_plot_2d_quality_landscape(n_grid: int = 41):
# %%
    n_grid = 21
    cfg = LensEnvConfig(
        tol_radius_rel=0.0,
        tol_thickness_mm=0.0,
        tol_decenter_mm=0.0,
        tol_tilt_deg=0.0,
        use_compensator=False,
        mtf_num_rays=128,
        mtf_grid_size=None,
        lens_groups=[
            LensGroupConfig(
                surf_front=3,
                surf_rear=4,
                z_source_surf=2,
                active_dofs=["dx", "dy"],
            )
        ],
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    env.reset(seed=0)
    dx_idx = env.dof_names.index("L1_dx")
    dy_idx = env.dof_names.index("L1_dy")
    x_lim = min(cfg.lens_groups[0].limit_dx_mm, 0.03)
    y_lim = min(cfg.lens_groups[0].limit_dy_mm, 0.03)
    xs = np.linspace(-x_lim, x_lim, n_grid)
    ys = np.linspace(-y_lim, y_lim, n_grid)
    quality_map = np.zeros((len(ys), len(xs)), dtype=np.float64)

    for iy, y in enumerate(ys):
        print(f'Computing row {iy+1}/{len(ys)} (dy={y:+.3f} mm)...')
        for ix, x in enumerate(xs):
            state = np.zeros(env.n_dof, dtype=np.float64)
            state[dx_idx] = x
            state[dy_idx] = y
            env._alignment_state = state.copy()
            env._mgr.apply_alignment_state(state)
            quality_map[iy, ix] = env._mgr.quality_metric()

    peak_flat_idx = int(np.argmax(quality_map))
    peak_iy, peak_ix = np.unravel_index(peak_flat_idx, quality_map.shape)
    peak_x = xs[peak_ix]
    peak_y = ys[peak_iy]
    peak_q = quality_map[peak_iy, peak_ix]

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(
        quality_map,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin="lower",
        aspect="equal",
        cmap="viridis",
        vmin=float(np.min(quality_map)),
        vmax=float(np.max(quality_map)),
    )
    plt.colorbar(im, ax=ax, label="quality metric")
    ax.plot(0.0, 0.0, "+", color="white", markersize=12, markeredgewidth=2.0, label="nominal center")
    ax.plot(peak_x, peak_y, "r*", markersize=14, label=f"peak ({peak_x:+.3f}, {peak_y:+.3f}) mm")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("dy [mm]")
    ax.set_title("T13: 2D Decenter Quality Landscape")
    ax.legend(loc="lower left")
    ax.grid(False)
    fig.tight_layout()

    print("\n[T13] 2D 偏心质量景观图")
    print(f"  dx range: [{xs[0]:+.3f}, {xs[-1]:+.3f}] mm")
    print(f"  dy range: [{ys[0]:+.3f}, {ys[-1]:+.3f}] mm")
    print(f"  peak: dx={peak_x:+.4f} mm, dy={peak_y:+.4f} mm, quality={peak_q:.4f}")
    print("  人工检查要点：整体是否接近单峰，峰值是否位于 (0,0) 附近，等值区是否基本同心/平滑。")

    plt.show()
    env.close()
# %%

# ======================================================================
# T11 4D 扩展（偏心+倾斜）
# ======================================================================
def t11_four_dof_extension():
    cfg = LensEnvConfig(
        lens_groups=[
            LensGroupConfig(
                surf_front=3, surf_rear=4, z_source_surf=2,
                active_dofs=["dx", "dy", "rx", "ry"],
            )
        ]
    )
    env = el.LensAlignmentEnv(cfg=cfg)
    _assert(env.n_dof == 4, f"4D n_dof should be 4, got {env.n_dof}")
    _assert(env.action_space.shape == (4,),
            f"action_space.shape should be (4,), got {env.action_space.shape}")
    _assert(env.dof_names == ["L1_dx", "L1_dy", "L1_rx", "L1_ry"],
            f"dof_names={env.dof_names}")

    obs, info = env.reset(seed=0)
    _assert(obs.shape == env.observation_space.shape, "obs shape mismatch")
    _assert("quality_metric" in info, "info must contain 'quality_metric'")
    check_env(env, warn=True)
    env.close()


# ======================================================================
# T12 双片 4D 对准（2 lens × 2D = 4D）
# ======================================================================
def t12_two_lens_extension():
    cfg = LensEnvConfig(
        lens_groups=[
            LensGroupConfig(surf_front=3, surf_rear=4, z_source_surf=2),
            LensGroupConfig(surf_front=5, surf_rear=6, z_source_surf=4),
        ]
    )
    env = el.LensAlignmentEnv(cfg=cfg)

    _assert(env.n_dof == 4, f"2-lens (2×2D) n_dof should be 4, got {env.n_dof}")
    _assert(env.action_space.shape == (4,),
            f"action_space.shape should be (4,), got {env.action_space.shape}")
    _assert(env.dof_names == ["L1_dx", "L1_dy", "L2_dx", "L2_dy"],
            f"dof_names={env.dof_names}")

    obs, info = env.reset(seed=0)
    _assert(obs.shape == env.observation_space.shape, "obs shape mismatch")
    check_env(env, warn=True)
    env.close()


# ======================================================================
# 入口
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LensAlignmentEnv 测试与人工检查脚本")
    parser.add_argument("-k", "--keyword", default="T13", help="仅运行名称中包含该关键字的测试，如 T03 / T13")
    parser.add_argument("--plots", action="store_true", default=True, help="包含人工检查型绘图测试（T13/T14）")
    args = parser.parse_args()

    print("\n=== LensAlignmentEnv 基础正确性测试 ===\n")

    tests = [
        ("T01  空间维度与数据类型（默认 2D）",       t01_spaces),
        ("T02  reset 固定种子可复现性",               t02_reset_reproducibility),
        ("T03  零动作状态不变",                        t03_zero_action_no_state_change),
        ("T04  完美对准质量 ≈ 1.0",                   t04_perfect_alignment_quality),
        ("T05  Y 偏心灵敏度",                          t05_sensitivity_to_decenter),
        ("T06  超限动作被截断",                        t06_action_clipping),
        ("T07  episode 正常截断",                      t07_truncation),
        ("T08  奖励数值一致性",                        t08_reward_consistency),
        ("T09  补偿器接口（compensator_z 字段）",         t09_compensator_interface),
        ("T10  Gymnasium check_env",                   t10_gymnasium_check),
        ("T11  4D 扩展（偏心+倾斜）",                 t11_four_dof_extension),
        ("T12  双片 4D 对准",                          t12_two_lens_extension),
    ]

    if args.plots:
        tests.extend([
            ("T13  2D 偏心-质量伪彩色图",              t13_plot_2d_quality_landscape),
            ("T14  指定偏心点的离焦 MTF 曲线",         t14_plot_defocus_mtf_curve),
        ])

    if args.keyword:
        key = args.keyword.lower()
        tests = [(name, fn) for name, fn in tests if key in name.lower() or key in fn.__name__.lower()]
        if not tests:
            raise SystemExit(f"未找到匹配关键字 '{args.keyword}' 的测试")

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [{PASS}] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [{FAIL}] {name}")
            print(f"          → {e}")
            failed += 1
        except Exception as e:
            print(f"  [{FAIL}] {name}")
            print(f"          → 异常: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n结果：{passed} 通过 / {failed} 失败 / {passed+failed} 总计\n")
