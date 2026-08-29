"""Short runtime benchmark for the single-process batched lens VecEnv."""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlaa" / "Lib" / "site-packages"))
OPTICS_CORE_PATH = Path(
    os.environ.get("ALIGNMENT_RL_OPTICS_CORE_PATH", str(ROOT / "optics_core-dev"))
)
sys.path.insert(0, str(OPTICS_CORE_PATH))

import numpy as np
import torch

from config import make_lens_rl_config
from env.batch_lens_env import BatchLensAlignmentVecEnv


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    cfg = make_lens_rl_config(fast_mode=True)
    print(f"optics_core_path={OPTICS_CORE_PATH}")
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    env = BatchLensAlignmentVecEnv(cfg, n_envs=4, seed=42)
    sync()
    construct_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    initial_obs = env.reset()
    sync()
    reset_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"reset_obs_finite={bool(np.isfinite(initial_obs).all())}, "
        f"reset_obs_mean={float(initial_obs.mean()):.6f}, "
        f"reset_obs_std={float(initial_obs.std()):.6f}"
    )

    actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
    for _ in range(5):
        env.step(actions)
    sync()

    steps = 30
    started = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    sync()
    elapsed = time.perf_counter() - started
    env.close()

    print(f"construct_ms={construct_ms:.2f}")
    print(f"reset_ms={reset_ms:.2f}")
    print(f"step_calls={steps}")
    print(f"step_ms={elapsed * 1000.0 / steps:.2f}")
    print(f"vec_steps_per_sec={steps / elapsed:.3f}")
    print(f"logical_env_steps_per_sec={steps * 4 / elapsed:.3f}")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"peak_cuda_allocated_mb={peak_mb:.1f}")


if __name__ == "__main__":
    main()
