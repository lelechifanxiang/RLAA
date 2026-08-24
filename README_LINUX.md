# AlignmentRL Linux Deployment Bundle

This bundle contains the current `alignment_rl-dev` implementation and the
matching `optics_core-dev` runtime. It is self-contained and uses only the
Double Gauss six-lens prescription:

`optics_core-dev/tests/zemax/zmx_files/Double Gauss 28 degree field with CB.ZMX`

The existing source trees were not modified.

## Ubuntu 22.04 / A10 setup

From the bundle root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r alignment_rl-dev/requirements-linux.txt
python -m pip install -e optics_core-dev
python -m pip install -e alignment_rl-dev --no-deps
export MPLBACKEND=Agg
```

Install the PyTorch wheel matching the host CUDA driver before the project
requirements, for example the CUDA 12.1 index supplied by the PyTorch release.

## Smoke checks

```bash
python alignment_rl-dev/quick_cuda_check.py
python alignment_rl-dev/test_coordinate_break_alignment.py
python alignment_rl-dev/test_lens_env.py
```

## Train, evaluate, and resume

```bash
python alignment_rl-dev/train.py --algo sac --timesteps 100000
python alignment_rl-dev/evaluate.py --model_path alignment_rl-dev/models/<model> --only_rl
```

Training creates `alignment_rl-dev/models/` and `alignment_rl-dev/logs/` on the
target machine. They are intentionally absent from this source bundle so that
historical replay buffers and process logs are not transferred.

The core path is resolved automatically to the bundled `optics_core-dev`; it
can be overridden with `ALIGNMENT_RL_OPTICS_CORE_PATH` when maintaining a
separate core checkout.

## Maintenance

Keep source and generated outputs separate. Back up only selected `.zip` model
checkpoints; replay buffers (`*_replay_buffer.pkl`) are large and should be
retained only when exact continuation of a run is required.
