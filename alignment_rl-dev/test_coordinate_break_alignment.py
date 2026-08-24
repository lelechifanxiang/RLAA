"""Regression checks for the optics_core Coordinate Break alignment mapping."""

from __future__ import annotations

import numpy as np
import torch

import optics_core as oc

from config import make_lens_rl_config
from env.batch_lens_env import BatchLensAlignmentVecEnv
from env.lens_env import LensAlignmentEnv


_FIRST_ORDER_FIELDS = (
    "effl",
    "working_f_number",
    "ttl",
    "image_plane_distance",
    "bfl",
    "valid",
    "entrance_pupil_z",
    "entrance_pupil_radius",
    "stop_radius",
    "exit_pupil_z",
    "exit_pupil_radius",
)


def _first_order_snapshot(system: oc.MultiOpticalSystem) -> dict[str, torch.Tensor]:
    data = system.first_order_data
    assert data is not None
    return {
        name: torch.as_tensor(getattr(data, name)).detach().cpu().clone()
        for name in _FIRST_ORDER_FIELDS
    }


def test_cb_pose_does_not_change_first_order_data_or_mtf() -> None:
    """CB x/y/tilt changes preserve first-order data exactly."""
    cfg = make_lens_rl_config(fast_mode=True)
    cfg.lens_groups[1].active_dofs = ["dx", "dy", "rx", "ry"]
    env = BatchLensAlignmentVecEnv(cfg, n_envs=4, seed=19)
    env.reset()
    system = env._core_system
    settings = oc.MTFSettings(
        pupil_sample_count=cfg.mtf_num_rays,
        image_sample_count=64,
        frequencies_lp_per_mm=tuple(cfg.mtf_frequencies),
        field_indices=(0,),
        wavelength_index=-1,
    )

    snapshots: list[dict[str, torch.Tensor]] = []
    full_mtf: list[torch.Tensor] = []
    frame_only_mtf: list[torch.Tensor] = []
    for state in (
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.25, -0.15, 0.20, -0.10]),
        np.array([-0.40, 0.30, -0.35, 0.25]),
    ):
        env._state_managers[0].apply_alignment_state(state)
        vector = env._vector_from_manager(env._state_managers[0])
        vectors = [list(vector) for _ in range(env.num_envs)]

        system.parameters.vectors = vectors
        system.prepare(
            recompute_materials=False,
            recompute_frame=True,
            recompute_first_order=True,
            recompute_clear_apertures=False,
        )
        snapshots.append(_first_order_snapshot(system))
        result = system.analysis.mtf(settings).run()
        full_mtf.append(torch.as_tensor(result.sagittal).detach().cpu().clone())

        system.parameters.vectors = vectors
        system.prepare(
            recompute_materials=False,
            recompute_frame=True,
            recompute_first_order=False,
            recompute_clear_apertures=False,
        )
        result = system.analysis.mtf(settings).run()
        frame_only_mtf.append(torch.as_tensor(result.sagittal).detach().cpu().clone())

    for name in _FIRST_ORDER_FIELDS:
        for snapshot in snapshots[1:]:
            torch.testing.assert_close(snapshot[name], snapshots[0][name], rtol=0.0, atol=0.0)
    for expected, actual in zip(full_mtf, frame_only_mtf):
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)
    env.close()


def test_single_environment_alignment_changes_rays_mtf_and_reward() -> None:
    cfg = make_lens_rl_config(fast_mode=True)
    env = LensAlignmentEnv(cfg)
    manager = env._mgr

    assert manager._core_cb_pairs == ((0, 3), (5, 9), (12, 16), (18, 21))
    assert not hasattr(manager, "lens")
    assert all(
        isinstance(manager._core_system.architecture.surfaces[index], oc.CoordinateBreak)
        for pair in manager._core_cb_pairs
        for index in pair
    )

    zero_state = np.zeros(env.n_dof, dtype=np.float64)
    manager.apply_alignment_state(zero_state)
    manager._ensure_core_prepared()
    nominal_mtf = manager._compute_mtf_obs()
    nominal_trace = manager._core_system.trace(
        sampler=oc.SquarePupilSampler(nx=3, ny=3),
        options=oc.TraceOptions(record_intersections=True),
    )

    moved_state = np.array([0.25, -0.15], dtype=np.float64)
    manager.apply_alignment_state(moved_state)
    manager._ensure_core_prepared()
    moved_mtf = manager._compute_mtf_obs()
    moved_trace = manager._core_system.trace(
        sampler=oc.SquarePupilSampler(nx=3, ny=3),
        options=oc.TraceOptions(record_intersections=True),
    )

    assert np.all(np.isfinite(nominal_mtf))
    assert np.all(nominal_mtf > 0.0)
    assert np.max(np.abs(moved_mtf - nominal_mtf)) > 1e-4
    assert torch.all(nominal_trace.valid)
    assert torch.all(moved_trace.valid)
    assert torch.max(torch.abs(moved_trace.rays.x - nominal_trace.rays.x)).item() > 1e-4
    assert torch.max(torch.abs(moved_trace.rays.y - nominal_trace.rays.y)).item() > 1e-4

    _, reset_info = env.reset(seed=123)
    _, reward, _, _, step_info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
    assert np.max(np.abs(step_info["raw_mtf_obs"] - reset_info["raw_mtf_obs"])) > 1e-4
    assert abs(reward) > 1e-6
    env.close()


def test_batched_alignment_changes_only_the_acted_design() -> None:
    cfg = make_lens_rl_config(fast_mode=True)
    env = BatchLensAlignmentVecEnv(cfg, n_envs=4, seed=123)
    env.reset()
    before = env._compute_mtf_batch()

    actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
    actions[0] = 1.0
    env.step_async(actions)
    _, rewards, _, infos = env.step_wait()
    after = np.stack([info["raw_mtf_obs"] for info in infos])

    differences = np.max(np.abs(after - before), axis=1)
    assert np.all(np.isfinite(after))
    assert np.all(np.max(after, axis=1) > 0.0)
    assert differences[0] > 1e-4
    np.testing.assert_array_equal(differences[1:], np.zeros(3))
    assert abs(float(rewards[0])) > 1e-6
    np.testing.assert_array_equal(rewards[1:], np.zeros(3, dtype=np.float32))
    env.close()
