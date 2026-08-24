"""Single-process batched vector environment for lens alignment training.

The logical environments keep independent episode state, while their optical
designs are represented by one ``MultiOpticalSystem``.  MTF therefore traces
all designs in one call and shares the CUDA context and prepared topology.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

import optics_core as oc

from config import LensEnvConfig
from env.lens_env import LensAlignmentEnv, _LensManager


class BatchLensAlignmentVecEnv(VecEnv):
    """SB3 ``VecEnv`` backed by one batched optics-core system."""

    def __init__(self, cfg: LensEnvConfig, n_envs: int = 4, seed: int = 0):
        if n_envs < 1:
            raise ValueError("n_envs must be positive")

        # Build one manager as the source of the optical topology. Logical
        # environments clone only alignment state and share one optics-core
        # system and CUDA context.
        prototype = LensAlignmentEnv(cfg=cfg)
        self.cfg = cfg
        self._prototype = prototype
        self._n_envs = int(n_envs)
        self._state_managers = [self._clone_state_manager(prototype._mgr) for _ in range(n_envs)]
        self._n_action = prototype._n_action
        self._n_mtf = prototype._n_mtf
        self._obs_dim = prototype.observation_space.shape[0]
        self._action_scale = prototype._action_scale.copy()
        self._action_limit = prototype._action_limit.copy()
        self._core_system = self._build_batch_system(prototype._mgr._core_system)
        self._parameter_indices = self._build_parameter_indices()
        self._core_tolerance_values: list[dict[str, float]] = [
            {} for _ in range(self._n_envs)
        ]

        self._alignment_state = np.zeros((n_envs, self._n_action), dtype=np.float64)
        self._prev_quality = np.zeros(n_envs, dtype=np.float64)
        self._step_count = np.zeros(n_envs, dtype=np.int32)
        self._mtf_obs_buffer = np.zeros(
            (n_envs, cfg.obs_history_len, self._n_mtf), dtype=np.float32
        )
        self._action_buffer = np.zeros(
            (n_envs, max(int(cfg.action_history_len), 0), self._n_action), dtype=np.float32
        )
        self._episode_ref_mtf_obs = np.zeros((n_envs, self._n_mtf), dtype=np.float32)
        self._rngs = [np.random.default_rng(seed + i) for i in range(n_envs)]
        self._seed_base = int(seed)
        self._pending_actions: np.ndarray | None = None
        self._closed = False
        self._mtf_error_reported = False

        super().__init__(n_envs, prototype.observation_space, prototype.action_space)
        self.metadata = dict(prototype.metadata)

    @staticmethod
    def _clone_state_manager(template: _LensManager) -> _LensManager:
        manager = object.__new__(_LensManager)
        manager.cfg = template.cfg
        manager._device = template._device
        manager._core_system = None
        manager._core_prepared = False
        manager._core_prepare_dirty = True
        manager._core_surface_state = {}
        manager._core_cb_pairs = tuple(template._core_cb_pairs)
        manager._target_group_indices = set(template._target_group_indices)
        manager._group_tolerance_states = copy.deepcopy(template._group_tolerance_states)
        manager._surface_group_indices = dict(template._surface_group_indices)
        manager._surface_tolerance_indices = set(template._surface_tolerance_indices)
        manager._group_alignment_states = copy.deepcopy(template._group_alignment_states)
        manager._nominal_mtf_obs = template._nominal_mtf_obs.copy()
        manager._episode_ref_mtf_obs = template._episode_ref_mtf_obs.copy()
        return manager

    def _build_batch_system(self, base: oc.MultiOpticalSystem) -> oc.MultiOpticalSystem:
        specs = []
        vector = []
        dynamic_frames = {
            surface_index
            for pair in self._prototype._mgr._core_cb_pairs
            for surface_index in pair
        }
        # The imported ZMX architecture starts at the first physical optical
        # surface (there is no explicit object surface); only the terminal
        # image surface is excluded by position.
        tolerance_excluded = (
            {len(base.architecture.surfaces) - 1}
            | set(self.cfg.tol_exclude_surfs)
            | dynamic_frames
        )
        for surface_index, surface in enumerate(base.architecture.surfaces):
            if surface_index in dynamic_frames:
                for attr in ("x", "y", "rx", "ry"):
                    value = float(getattr(surface.frame, attr))
                    specs.append(oc.ParameterSpec(
                        name=f"surface_{surface_index}_frame_{attr}",
                        path=f"surface[{surface_index}].frame.{attr}",
                        default=value,
                    ))
                    vector.append(value)
                continue

            if surface_index in tolerance_excluded:
                continue

            if surface_index in self._prototype._mgr._surface_tolerance_indices:
                for attr, kind in (("x", "surface_decenter"), ("y", "surface_decenter"), ("rx", "surface_tilt"), ("ry", "surface_tilt")):
                    value = float(getattr(surface.frame, attr))
                    specs.append(oc.ParameterSpec(
                        name=f"surface_{surface_index}_frame_{attr}",
                        path=f"surface[{surface_index}].frame.{attr}",
                        default=value,
                        metadata={"tolerance_kind": kind, "surface_index": surface_index},
                    ))
                    vector.append(value)

            geometry = surface.geometry
            radius = getattr(geometry, "radius", None)
            if radius is not None and np.isfinite(float(radius)) and float(radius) != 0.0:
                specs.append(oc.ParameterSpec(
                    name=f"surface_{surface_index}_geometry_radius",
                    path=f"surface[{surface_index}].geometry.radius",
                    default=float(radius),
                    metadata={"tolerance_kind": "radius", "surface_index": surface_index},
                ))
                vector.append(float(radius))

            thickness = getattr(surface.gap, "thickness", None)
            if thickness is not None and np.isfinite(float(thickness)):
                specs.append(oc.ParameterSpec(
                    name=f"surface_{surface_index}_gap_thickness",
                    path=f"surface[{surface_index}].gap.thickness",
                    default=float(thickness),
                    metadata={"tolerance_kind": "thickness", "surface_index": surface_index},
                ))
                vector.append(float(thickness))

        schema = oc.ParameterSchema(specs)
        parameters = oc.ParameterVectorBatch(
            schema=schema,
            vectors=[list(vector) for _ in range(self._n_envs)],
            grid_shape=(self._n_envs,),
        )
        # Reuse the already initialized container.  This keeps one optical
        # system and one prepared material/runtime cache in the process.
        base.parameters = parameters
        base.surfaces = base.architecture.surfaces.bind(owner=base, materials=base.materials)
        base._material_data = None
        base.frame_data = None
        base.first_order_data = None
        base.clear_aperture_data = None
        base._analysis_hub = None
        system = base
        system.prepare()
        return system

    def _sample_core_tolerances(self, index: int) -> dict[str, float]:
        """Draw one independent manufacturing-tolerance realization for a design."""
        rng = self._rngs[index]
        values: dict[str, float] = {}
        manager = self._state_managers[index]
        for group_values in manager._group_tolerance_states:
            group_values.update({dof: 0.0 for dof in manager.DOF_NAMES_PER_LENS})
        for group_index, group_values in enumerate(manager._group_tolerance_states):
            if group_index in manager._target_group_indices:
                continue
            group_values["dx"] = float(rng.normal(0.0, self.cfg.tol_lens_decenter_mm))
            group_values["dy"] = float(rng.normal(0.0, self.cfg.tol_lens_decenter_mm))
            group_values["rx"] = float(rng.normal(0.0, self.cfg.tol_lens_tilt_deg))
            group_values["ry"] = float(rng.normal(0.0, self.cfg.tol_lens_tilt_deg))
            values.update({
                f"lens_group_{group_index}_dx_mm": group_values["dx"],
                f"lens_group_{group_index}_dy_mm": group_values["dy"],
                f"lens_group_{group_index}_rx_deg": group_values["rx"],
                f"lens_group_{group_index}_ry_deg": group_values["ry"],
            })
        for spec in self._core_system.parameter_schema:
            kind = spec.metadata.get("tolerance_kind")
            if kind is None:
                continue
            base = float(spec.default_value())
            if kind == "radius":
                value = base + rng.normal(0.0, abs(base) * self.cfg.tol_radius_rel)
            elif kind == "thickness":
                value = base + rng.normal(0.0, self.cfg.tol_thickness_mm)
            elif kind == "surface_decenter":
                value = base + rng.normal(0.0, self.cfg.tol_decenter_mm)
            elif kind == "surface_tilt":
                value = base + rng.normal(0.0, self.cfg.tol_tilt_deg)
            else:
                raise ValueError(f"Unsupported core tolerance kind: {kind!r}")
            values[spec.name] = float(value)
        self._core_tolerance_values[index] = values
        return values

    def _build_parameter_indices(self) -> dict[tuple[int, str], int]:
        indices = {}
        for index, spec in enumerate(self._core_system.parameter_schema):
            indices[(int(spec.path.split("[")[1].split("]")[0]), spec.path.rsplit(".", 1)[-1])] = index
        return indices

    def _vector_from_manager(self, manager: _LensManager, index: int | None = None) -> list[float]:
        """Translate logical group poses to paired Coordinate Break values."""
        vector = list(self._core_system.parameter_schema.default_vector())
        if index is not None:
            for name, value in self._core_tolerance_values[index].items():
                try:
                    vector[self._core_system.parameter_schema.index_of(name)] = value
                except KeyError:
                    # Lens-level tolerance is applied through the paired CB
                    # frame below rather than as a standalone schema field.
                    pass
        for group_index, (entrance_index, return_index) in enumerate(manager._core_cb_pairs):
            values = manager._group_alignment_states[group_index]
            manufacturing = manager._group_tolerance_states[group_index]
            for attr, dof in (("x", "dx"), ("y", "dy"), ("rx", "rx"), ("ry", "ry")):
                value = float(values[dof] + manufacturing[dof])
                vector[self._parameter_indices[(entrance_index, attr)]] = value
                vector[self._parameter_indices[(return_index, attr)]] = -value
        return vector

    def _set_vectors(self, vectors: list[list[float]], *, full_recompute: bool = False) -> None:
        self._core_system.parameters.vectors = [list(vector) for vector in vectors]
        try:
            self._core_system.prepare(
                recompute_materials=False,
                recompute_frame=True,
                # The training action space currently contains only CB
                # decenter/tilt.  First-order data is invariant under these
                # rigid coordinate changes; recomputing it dominated the
                # step cost.  If axial dz is added to the batch schema, this
                # must be enabled again for that path.
                recompute_first_order=full_recompute,
                recompute_clear_apertures=full_recompute,
            )
        except TypeError as exc:
            if "recompute_materials" not in str(exc):
                raise
            self._core_system.prepare()

    def _compute_mtf_batch(self) -> np.ndarray:
        field_indices = [
            int(field_index)
            for field_index in self.cfg.mtf_field_indices
            if 0 <= int(field_index) < len(self._core_system.fields)
        ]
        if not field_indices:
            field_indices = [0]
        settings = oc.MTFSettings(
            pupil_sample_count=self.cfg.mtf_num_rays,
            image_sample_count=self.cfg.mtf_grid_size if self.cfg.mtf_grid_size else 64,
            frequencies_lp_per_mm=tuple(self.cfg.mtf_frequencies),
            field_indices=tuple(field_indices),
            wavelength_index=-1,
        )
        try:
            result = self._core_system.analysis.mtf(settings).run()
            sagittal = torch.as_tensor(result.sagittal, dtype=torch.float64).detach().cpu().numpy()
            tangential = torch.as_tensor(result.tangential, dtype=torch.float64).detach().cpu().numpy()
            if sagittal.ndim == 2:
                sagittal = sagittal[None, ...]
                tangential = tangential[None, ...]
            output = np.zeros((self._n_envs, self._n_mtf), dtype=np.float32)
            cursor = 0
            for field_index in range(sagittal.shape[1]):
                for orientation in (sagittal, tangential):
                    width = orientation.shape[2]
                    output[:, cursor:cursor + width] = np.clip(orientation[:, field_index], 0.0, 1.0)
                    cursor += width
            return output
        except Exception as exc:
            if not self._mtf_error_reported:
                print(f"Batched MTF calculation failed: {exc}")
                self._mtf_error_reported = True
            n = len(field_indices) * 2 * len(self.cfg.mtf_frequencies)
            return np.zeros((self._n_envs, n), dtype=np.float32)

    def _reset_indices(self, indices: list[int]) -> dict[int, dict[str, Any]]:
        if not indices:
            return {}
        # Reference MTF: all non-reset designs remain in the batch so that the
        # same kernel path is used and no per-design optics call is introduced.
        ref_vectors = [self._vector_from_manager(manager, index) for index, manager in enumerate(self._state_managers)]
        for index in indices:
            manager = self._state_managers[index]
            manager.reset_to_nominal()
            self._sample_core_tolerances(index)
            manager.apply_alignment_state(np.zeros(self._n_action, dtype=np.float64))
            ref_vectors[index] = self._vector_from_manager(manager, index)
        self._set_vectors(ref_vectors, full_recompute=True)
        references = self._compute_mtf_batch()
        self._episode_ref_mtf_obs[indices] = references[indices]

        initial_vectors = list(ref_vectors)
        for index in indices:
            state = self._sample_init_state(index)
            self._alignment_state[index] = state
            self._state_managers[index].apply_alignment_state(state)
            initial_vectors[index] = self._vector_from_manager(self._state_managers[index], index)
        self._set_vectors(initial_vectors)
        raw = self._compute_mtf_batch()
        ceiling = self.cfg.initial_quality_ceiling
        if ceiling is not None:
            # Only reset designs may be resampled: the other batched designs
            # retain their live episode state during an asynchronous reset.
            pending = list(indices)
            attempts = max(int(self.cfg.initial_quality_sampling_attempts), 1)
            for _ in range(attempts - 1):
                pending = [
                    index for index in pending
                    if float(np.mean(self._relative_log(raw[index], self._episode_ref_mtf_obs[index]))) >= ceiling
                ]
                if not pending:
                    break
                for index in pending:
                    state = self._sample_init_state(index)
                    self._alignment_state[index] = state
                    self._state_managers[index].apply_alignment_state(state)
                    initial_vectors[index] = self._vector_from_manager(self._state_managers[index], index)
                self._set_vectors(initial_vectors)
                raw = self._compute_mtf_batch()
            fallback = [
                index for index in pending
                if float(np.mean(self._relative_log(raw[index], self._episode_ref_mtf_obs[index]))) >= ceiling
            ]
            if fallback:
                # q=0 at the tolerance-specific zero pose is a valid start
                # for maps with no negative-quality region in the travel box.
                for index in fallback:
                    state = np.zeros(self._n_action, dtype=np.float64)
                    self._alignment_state[index] = state
                    self._state_managers[index].apply_alignment_state(state)
                    initial_vectors[index] = self._vector_from_manager(self._state_managers[index], index)
                self._set_vectors(initial_vectors)
                raw = self._compute_mtf_batch()
        observations = {}
        for index in indices:
            mtf = self._relative_log(raw[index], self._episode_ref_mtf_obs[index])
            self._prev_quality[index] = float(np.mean(mtf))
            self._step_count[index] = 0
            self._mtf_obs_buffer[index] = mtf
            self._action_buffer[index].fill(0.0)
            observations[index] = {
                "quality_metric": self._prev_quality[index],
                "state": self._alignment_state[index].copy(),
                "step": 0,
                "success": bool(self._prev_quality[index] >= self.cfg.success_threshold),
                "mtf_obs": mtf,
                "raw_mtf_obs": raw[index],
                "episode_ref_mtf_obs": self._episode_ref_mtf_obs[index].copy(),
                "nominal_mtf_obs": self._prototype._mgr._nominal_mtf_obs.copy(),
                "compensator_z": 0.0,
                "core_tolerances": dict(self._core_tolerance_values[index]),
            }
        return observations

    def _sample_init_state(self, index: int) -> np.ndarray:
        values = []
        for lg in self.cfg.lens_groups:
            for dof in lg.active_dofs:
                attr = {"dx": "init_dx_mm", "dy": "init_dy_mm", "dz": "init_dz_mm", "rx": "init_rx_deg", "ry": "init_ry_deg"}[dof]
                values.append(self._rngs[index].uniform(-float(getattr(lg, attr)), float(getattr(lg, attr))))
        return np.asarray(values, dtype=np.float64)

    def _relative_log(self, raw: np.ndarray, reference: np.ndarray) -> np.ndarray:
        eps = self.cfg.mtf_log_epsilon
        return np.clip(np.log((raw + eps) / (reference + eps)), -self.cfg.mtf_relative_clip, self.cfg.mtf_relative_clip).astype(np.float32)

    def reset(self) -> np.ndarray:
        seed_values = self._seeds
        for index, value in enumerate(seed_values):
            if value is not None:
                self._rngs[index] = np.random.default_rng(value)
        self._reset_seeds()
        self._reset_options()
        infos = self._reset_indices(list(range(self._n_envs)))
        self.reset_infos = [infos[index] for index in range(self._n_envs)]
        return self._mtf_obs_buffer.reshape(self._n_envs, -1).copy() if self.cfg.action_history_len == 0 else np.concatenate((self._mtf_obs_buffer.reshape(self._n_envs, -1), self._action_buffer.reshape(self._n_envs, -1)), axis=1).astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32).copy()

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_wait() called before step_async()")
        actions = np.clip(self._pending_actions, -1.0, 1.0)
        self._pending_actions = None
        vectors = []
        for index, manager in enumerate(self._state_managers):
            delta = actions[index].astype(np.float64) * self._action_scale
            state = np.clip(self._alignment_state[index] + delta, -self._action_limit, self._action_limit)
            self._alignment_state[index] = state
            manager.apply_alignment_state(state)
            vectors.append(self._vector_from_manager(manager, index))
        self._set_vectors(vectors)
        raw = self._compute_mtf_batch()
        observations = np.empty((self._n_envs, self._obs_dim), dtype=np.float32)
        rewards = np.zeros(self._n_envs, dtype=np.float32)
        dones = np.zeros(self._n_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        reset_indices = []
        for index in range(self._n_envs):
            mtf = self._relative_log(raw[index], self._episode_ref_mtf_obs[index])
            quality = float(np.mean(mtf))
            terminated = bool(quality >= self.cfg.success_threshold)
            self._step_count[index] += 1
            truncated = bool(self._step_count[index] >= self.cfg.max_episode_steps)
            done = terminated or truncated
            rewards[index] = (quality - self._prev_quality[index]) + self.cfg.success_bonus * float(terminated)
            self._prev_quality[index] = quality
            self._mtf_obs_buffer[index] = np.roll(self._mtf_obs_buffer[index], -1, axis=0)
            self._mtf_obs_buffer[index, -1] = mtf
            if self.cfg.action_history_len > 0:
                self._action_buffer[index] = np.roll(self._action_buffer[index], -1, axis=0)
                self._action_buffer[index, -1] = actions[index]
            infos.append({"quality_metric": quality, "state": self._alignment_state[index].copy(), "step": int(self._step_count[index]), "success": terminated, "mtf_obs": mtf, "raw_mtf_obs": raw[index], "episode_ref_mtf_obs": self._episode_ref_mtf_obs[index].copy(), "nominal_mtf_obs": self._prototype._mgr._nominal_mtf_obs.copy(), "compensator_z": 0.0, "core_tolerances": dict(self._core_tolerance_values[index]), "TimeLimit.truncated": truncated and not terminated})
            dones[index] = done
            reset_indices.append(index) if done else None
            observations[index] = self._get_obs(index)

        for index in reset_indices:
            infos[index]["terminal_observation"] = observations[index].copy()
        reset_infos = self._reset_indices(reset_indices)
        for index in reset_indices:
            infos[index].update({"reset_info": reset_infos[index]})
            observations[index] = self._get_obs(index)
        return observations, rewards, dones, infos

    def _get_obs(self, index: int) -> np.ndarray:
        parts = [self._mtf_obs_buffer[index].reshape(-1)]
        if self.cfg.action_history_len > 0:
            parts.append(self._action_buffer[index].reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def close(self) -> None:
        if not self._closed:
            self._prototype.close()
            self._closed = True

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return [getattr(self._prototype, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for index in self._get_indices(indices):
            if index == 0:
                setattr(self._prototype, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        method = getattr(self._prototype, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def _get_indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self._n_envs))
        if isinstance(indices, (int, np.integer)):
            return [int(indices)]
        return [int(index) for index in indices]
