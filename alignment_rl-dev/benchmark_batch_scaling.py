"""Benchmark logical design-batch scaling without changing optics_core."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlaa" / "Lib" / "site-packages"))
sys.path.insert(0, os.environ.get("ALIGNMENT_RL_OPTICS_CORE_PATH", str(ROOT / "optics_core-dev")))

import numpy as np
import torch

from config import make_lens_rl_config
from env.batch_lens_env import BatchLensAlignmentVecEnv


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cfg = make_lens_rl_config(fast_mode=True)
    started = time.perf_counter()
    env = BatchLensAlignmentVecEnv(cfg, n_envs=args.n_envs, seed=42)
    sync()
    construct_ms = (time.perf_counter() - started) * 1000.0

    env.reset()
    actions = np.zeros((args.n_envs, env.action_space.shape[0]), dtype=np.float32)
    for _ in range(5):
        env.step(actions)
    sync()

    started = time.perf_counter()
    for _ in range(args.steps):
        env.step(actions)
    sync()
    elapsed = time.perf_counter() - started

    result = {
        "n_envs": args.n_envs,
        "construct_ms": construct_ms,
        "vec_steps_per_sec": args.steps / elapsed,
        "logical_env_steps_per_sec": args.steps * args.n_envs / elapsed,
        "step_ms": elapsed * 1000.0 / args.steps,
    }
    if torch.cuda.is_available():
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024**2
    print(" ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in result.items()))
    env.close()


if __name__ == "__main__":
    main()
