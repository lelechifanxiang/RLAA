# 4DOF Alternating Alignment - Quick Start

## Quick Test (1 minute)
```bash
python test_4dof_alternating.py
```

## Start Training
```bash
# Alternating mode (recommended)
python train_4dof_alternating.py --mode alternating --timesteps 2000000

# Simultaneous mode (comparison)
python train_4dof_alternating.py --mode simultaneous --timesteps 2000000
```

## Analyze Results
```bash
python analyze_4dof_alternating.py \
    --model models/sac_4dof_alternating_TIMESTAMP_final.zip \
    --mode alternating \
    --n-episodes 10
```

## What's New
- 4 DOF: dx, dy (decenter) + rx, ry (tilt ±1°)
- Alternating motion: odd steps adjust decenter, even steps adjust tilt
- Dual mode comparison: alternating vs simultaneous

## Files
- `config_4dof.py` - Configuration
- `env/alternating_lens_env.py` - Environment
- `train_4dof_alternating.py` - Training script
- `analyze_4dof_alternating.py` - Analysis script
- `test_4dof_alternating.py` - Test script
- `README_4DOF_ALTERNATING.md` - Full documentation

## Training Time
~26 hours for 2M steps on RTX 5060 Ti (12 parallel environments)
