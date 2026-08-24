import numpy as np
from env.lens_env import LensAlignmentEnv


_LENS_STAGNATION_NOISE = 0.002
_LENS_STAGNATION_LIMIT = 4
_LENS_PROBE_SHRINK_FACTOR = 0.5
_LENS_PROBE_MIN_FRAC = 0.02
_LENS_MOVE_GROW_FACTOR = 1.6
_LENS_MOVE_SHRINK_FACTOR = 0.5
_LENS_MOVE_MIN_FRAC = 0.05
_LENS_MOVE_MAX_FRAC = 1.0
_LENS_MOVE_TRIAL_SCALES = (1.0, 0.5, 0.25, 0.125)


def _init_lens_rollout(env: LensAlignmentEnv, seed: int | None) -> tuple[float, np.ndarray, list[float], list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    _, info = env.reset(seed=seed)
    quality = info["quality_metric"]
    pos = info["state"].copy()
    qualities = [quality]
    states = [pos.copy()]
    actions_log: list[np.ndarray] = []
    mtf_obs_log = [info["mtf_obs"].copy()]
    return quality, pos, qualities, states, actions_log, mtf_obs_log, info


def _finalize_lens_rollout(
    qualities: list[float],
    states: list[np.ndarray],
    actions_log: list[np.ndarray],
    mtf_obs_log: list[np.ndarray],
    info: dict,
) -> dict:
    return {
        "qualities": qualities,
        "etas": qualities,
        "actions": actions_log,
        "states": states,
        "mtf_obs": mtf_obs_log,
        "steps": len(actions_log),
        "success": info["success"],
        "final_eta": qualities[-1],
    }

def _probe_lens_axis(
    env: LensAlignmentEnv,
    axis: int,
    probe_frac: float,
    current_q: float,
    current_pos: np.ndarray,
    pos_max: np.ndarray,
    step_scale: np.ndarray,
    qualities_buf: list[float],
    actions_buf: list[np.ndarray],
    states_buf: list[np.ndarray],
    mtf_obs_buf: list[np.ndarray],
) -> tuple[float | None, bool, bool, dict]:
    """边界感知的单轴梯度估计，适配 LensAlignmentEnv。"""
    action = np.zeros(env.n_dof, dtype=np.float32)
    phys = probe_frac * step_scale[axis]

    can_fwd = (current_pos[axis] + phys) <= pos_max[axis]
    can_bwd = (current_pos[axis] - phys) >= -pos_max[axis]

    if can_fwd and can_bwd:
        action[axis] = probe_frac
        _, _, t, tr, info = env.step(action)
        q_p = info["quality_metric"]
        qualities_buf.append(q_p)
        actions_buf.append(action.copy())
        states_buf.append(info["state"].copy())
        mtf_obs_buf.append(info["mtf_obs"].copy())
        if t or tr:
            return None, t, tr, info

        action[axis] = -2 * probe_frac
        _, _, t, tr, info = env.step(action)
        q_n = info["quality_metric"]
        qualities_buf.append(q_n)
        actions_buf.append(action.copy())
        states_buf.append(info["state"].copy())
        mtf_obs_buf.append(info["mtf_obs"].copy())
        if t or tr:
            return None, t, tr, info

        action[axis] = probe_frac
        _, _, t, tr, info = env.step(action)
        qualities_buf.append(info["quality_metric"])
        actions_buf.append(action.copy())
        states_buf.append(info["state"].copy())
        mtf_obs_buf.append(info["mtf_obs"].copy())
        return q_p - q_n, t, tr, info

    if can_fwd:
        action[axis] = probe_frac
        _, _, t, tr, info = env.step(action)
        q_p = info["quality_metric"]
        qualities_buf.append(q_p)
        actions_buf.append(action.copy())
        states_buf.append(info["state"].copy())
        mtf_obs_buf.append(info["mtf_obs"].copy())
        if t or tr:
            return None, t, tr, info

        action[axis] = -probe_frac
        _, _, t, tr, info = env.step(action)
        qualities_buf.append(info["quality_metric"])
        actions_buf.append(action.copy())
        states_buf.append(info["state"].copy())
        mtf_obs_buf.append(info["mtf_obs"].copy())
        return 2.0 * (q_p - current_q), t, tr, info

    if can_bwd:
        action[axis] = -probe_frac
        _, _, t, tr, info = env.step(action)
        q_n = info["quality_metric"]
        qualities_buf.append(q_n)
        actions_buf.append(action.copy())
        states_buf.append(info["state"].copy())
        mtf_obs_buf.append(info["mtf_obs"].copy())
        if t or tr:
            return None, t, tr, info

        action[axis] = probe_frac
        _, _, t, tr, info = env.step(action)
        qualities_buf.append(info["quality_metric"])
        actions_buf.append(action.copy())
        states_buf.append(info["state"].copy())
        mtf_obs_buf.append(info["mtf_obs"].copy())
        return 2.0 * (current_q - q_n), t, tr, info

    dummy_info = {
        "quality_metric": current_q,
        "state": current_pos.copy(),
        "success": current_q >= env.cfg.success_threshold,
        "step": -1,
        "mtf_obs": mtf_obs_buf[-1].copy() if mtf_obs_buf else np.zeros(env._n_mtf, dtype=np.float32),
    }
    return 0.0, False, False, dummy_info


def _step_lens_env(
    env: LensAlignmentEnv,
    action: np.ndarray,
    qualities_buf: list[float],
    actions_buf: list[np.ndarray],
    states_buf: list[np.ndarray],
    mtf_obs_buf: list[np.ndarray],
) -> tuple[bool, bool, dict]:
    _, _, terminated, truncated, info = env.step(action)
    qualities_buf.append(info["quality_metric"])
    actions_buf.append(action.copy())
    states_buf.append(info["state"].copy())
    mtf_obs_buf.append(info["mtf_obs"].copy())
    return terminated, truncated, info


def _try_lens_move(
    env: LensAlignmentEnv,
    direction: np.ndarray,
    move_frac: float,
    current_q: float,
    qualities_buf: list[float],
    actions_buf: list[np.ndarray],
    states_buf: list[np.ndarray],
    mtf_obs_buf: list[np.ndarray],
) -> tuple[bool, float, float, bool, bool, dict]:
    """沿梯度方向尝试多级步长，若劣化则回退。"""
    last_info = {
        "quality_metric": current_q,
        "state": states_buf[-1].copy(),
        "success": current_q >= env.cfg.success_threshold,
        "step": -1,
        "mtf_obs": mtf_obs_buf[-1].copy(),
    }

    for scale in _LENS_MOVE_TRIAL_SCALES:
        trial_frac = float(np.clip(move_frac * scale, _LENS_MOVE_MIN_FRAC, _LENS_MOVE_MAX_FRAC))
        action = np.clip(direction * trial_frac, -1.0, 1.0).astype(np.float32)
        terminated, truncated, info = _step_lens_env(
            env, action, qualities_buf, actions_buf, states_buf, mtf_obs_buf
        )
        new_q = info["quality_metric"]
        last_info = info
        if terminated or truncated:
            return True, new_q, trial_frac, terminated, truncated, info
        if new_q >= current_q:
            return True, new_q, trial_frac, False, False, info

        back_action = -action
        terminated, truncated, info = _step_lens_env(
            env, back_action, qualities_buf, actions_buf, states_buf, mtf_obs_buf
        )
        last_info = info
        if terminated or truncated:
            return False, info["quality_metric"], trial_frac, terminated, truncated, info

    return False, current_q, move_frac, False, False, last_info


def _lens_random_kick(
    env: LensAlignmentEnv,
    rng: np.random.Generator,
    dyn_probe: float,
    qualities: list[float],
    actions_log: list[np.ndarray],
    states: list[np.ndarray],
    mtf_obs_log: list[np.ndarray],
) -> tuple[float, np.ndarray, bool, bool, dict]:
    action = rng.uniform(-dyn_probe, dyn_probe, size=env.n_dof).astype(np.float32)
    terminated, truncated, info = _step_lens_env(
        env, action, qualities, actions_log, states, mtf_obs_log
    )
    return info["quality_metric"], info["state"].copy(), terminated, truncated, info


def _run_lens_aligner(
    env: LensAlignmentEnv,
    seed: int | None,
    probe_fraction: float,
    move_fraction: float,
    build_direction,
) -> dict:
    quality, pos, qualities, states, actions_log, mtf_obs_log, info = _init_lens_rollout(env, seed)
    terminated, truncated = False, False

    pos_max = np.array(env._action_limit, dtype=np.float64)
    step_scale = np.array(env._action_scale, dtype=np.float64)
    rng = np.random.default_rng(seed)

    stagnation_count = 0
    dyn_probe = probe_fraction
    dyn_move = move_fraction
    last_move_q = quality
    search_state: dict[str, int] = {"axis": 0}

    while not (terminated or truncated):
        current_q = quality
        direction, quality, pos, terminated, truncated, info = build_direction(
            env=env,
            dyn_probe=dyn_probe,
            quality=quality,
            pos=pos,
            pos_max=pos_max,
            step_scale=step_scale,
            qualities=qualities,
            actions_log=actions_log,
            states=states,
            mtf_obs_log=mtf_obs_log,
            search_state=search_state,
        )
        if terminated or truncated:
            break

        moved = False
        new_quality = quality
        if direction is not None:
            moved, new_quality, accepted_move, terminated, truncated, info = _try_lens_move(
                env,
                direction,
                dyn_move,
                current_q,
                qualities,
                actions_log,
                states,
                mtf_obs_log,
            )
            pos = info["state"].copy()
            quality = new_quality
            if terminated or truncated:
                break
            improvement = new_quality - current_q
            if moved and improvement >= 2.0 * _LENS_STAGNATION_NOISE:
                dyn_move = min(accepted_move * _LENS_MOVE_GROW_FACTOR, _LENS_MOVE_MAX_FRAC)
            elif moved:
                dyn_move = accepted_move
            else:
                dyn_move = max(dyn_move * _LENS_MOVE_SHRINK_FACTOR, _LENS_MOVE_MIN_FRAC)
        else:
            dyn_move = max(dyn_move * _LENS_MOVE_SHRINK_FACTOR, _LENS_MOVE_MIN_FRAC)

        if moved and (new_quality - last_move_q) >= _LENS_STAGNATION_NOISE:
            stagnation_count = 0
            dyn_probe = max(min(0.5 * dyn_move, probe_fraction), _LENS_PROBE_MIN_FRAC)
        else:
            stagnation_count += 1
            dyn_probe = max(dyn_probe * _LENS_PROBE_SHRINK_FACTOR, _LENS_PROBE_MIN_FRAC)
        last_move_q = quality

        if stagnation_count >= _LENS_STAGNATION_LIMIT and not (terminated or truncated):
            quality, pos, terminated, truncated, info = _lens_random_kick(
                env,
                rng,
                dyn_probe,
                qualities,
                actions_log,
                states,
                mtf_obs_log,
            )
            stagnation_count = 0
            dyn_probe = probe_fraction
            dyn_move = min(max(dyn_move, move_fraction), _LENS_MOVE_MAX_FRAC)
            last_move_q = quality
            if terminated or truncated:
                break

    return _finalize_lens_rollout(qualities, states, actions_log, mtf_obs_log, info)


def _build_simultaneous_direction(
    *,
    env: LensAlignmentEnv,
    dyn_probe: float,
    quality: float,
    pos: np.ndarray,
    pos_max: np.ndarray,
    step_scale: np.ndarray,
    qualities: list[float],
    actions_log: list[np.ndarray],
    states: list[np.ndarray],
    mtf_obs_log: list[np.ndarray],
    search_state: dict[str, int],
) -> tuple[np.ndarray | None, float, np.ndarray, bool, bool, dict]:
    grad = np.zeros(env.n_dof, dtype=np.float64)
    current_pos = pos
    current_quality = quality

    for axis in range(env.n_dof):
        q_diff, terminated, truncated, info = _probe_lens_axis(
            env,
            axis,
            dyn_probe,
            current_quality,
            current_pos,
            pos_max,
            step_scale,
            qualities,
            actions_log,
            states,
            mtf_obs_log,
        )
        current_pos = info["state"].copy()
        current_quality = qualities[-1]
        if terminated or truncated:
            return None, current_quality, current_pos, terminated, truncated, info
        if q_diff is not None:
            grad[axis] = q_diff

    g_inf = np.max(np.abs(grad))
    if g_inf <= 1e-10:
        return None, current_quality, current_pos, False, False, info

    direction = np.clip(grad / g_inf, -1.0, 1.0).astype(np.float32)
    return direction, current_quality, current_pos, False, False, info


def _build_axis_direction(
    *,
    env: LensAlignmentEnv,
    dyn_probe: float,
    quality: float,
    pos: np.ndarray,
    pos_max: np.ndarray,
    step_scale: np.ndarray,
    qualities: list[float],
    actions_log: list[np.ndarray],
    states: list[np.ndarray],
    mtf_obs_log: list[np.ndarray],
    search_state: dict[str, int],
) -> tuple[np.ndarray | None, float, np.ndarray, bool, bool, dict]:
    axis = search_state["axis"]
    q_diff, terminated, truncated, info = _probe_lens_axis(
        env,
        axis,
        dyn_probe,
        quality,
        pos,
        pos_max,
        step_scale,
        qualities,
        actions_log,
        states,
        mtf_obs_log,
    )
    search_state["axis"] = (axis + 1) % env.n_dof
    current_pos = info["state"].copy()
    current_quality = qualities[-1]
    if terminated or truncated:
        return None, current_quality, current_pos, terminated, truncated, info
    if q_diff is None or abs(q_diff) <= 1e-10:
        return None, current_quality, current_pos, False, False, info

    direction = np.zeros(env.n_dof, dtype=np.float32)
    direction[axis] = float(np.sign(q_diff))
    return direction, current_quality, current_pos, False, False, info

class LensHillClimbAligner:
    """用于 LensAlignmentEnv 的全轴同步梯度爬山法。"""

    def __init__(self, probe_fraction: float = 0.08, move_fraction: float = 0.5):
        self.probe_frac = probe_fraction
        self.move_frac = move_fraction

    def run(self, env: LensAlignmentEnv, seed: int | None = None) -> dict:
        return _run_lens_aligner(env, seed, self.probe_frac, self.move_frac, _build_simultaneous_direction)


class LensCoordinateSearchAligner:
    """用于 LensAlignmentEnv 的单轴交替搜索基线。"""

    def __init__(self, probe_fraction: float = 0.08, move_fraction: float = 0.5):
        self.probe_frac = probe_fraction
        self.move_frac = move_fraction

    def run(self, env: LensAlignmentEnv, seed: int | None = None) -> dict:
        return _run_lens_aligner(env, seed, self.probe_frac, self.move_frac, _build_axis_direction)
