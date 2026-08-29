"""Background entry point for the 100k-step SAC run."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "rlaa" / "Lib" / "site-packages"))
sys.path.insert(0, str(ROOT / "optics_core-dev"))

from train import train


if __name__ == "__main__":
    train(algo="sac", total_timesteps=100_000, seed=42)
