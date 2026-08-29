# 4DOF Alternating Alignment System - Linux Package

## Quick Start

```bash
# 1. Extract (if compressed)
tar -xzf alignment_rl-4dof-linux-YYYYMMDD.tar.gz
cd alignment_rl-4dof-linux

# 2. Deploy
chmod +x deploy_linux.sh
./deploy_linux.sh

# 3. Train
source venv/bin/activate
python3 train_4dof_alternating.py --mode alternating --timesteps 2000000
```

## What's New

- **4 Degrees of Freedom**: dx, dy (decenter) + rx, ry (tilt ±1°)
- **Alternating Motion**: Odd steps adjust decenter, even steps adjust tilt
- **Dual Mode**: Compare alternating vs simultaneous adjustment

## System Requirements

- Python 3.8+
- CUDA 11.8+
- NVIDIA GPU (8GB+ VRAM)
- Ubuntu 20.04+ / CentOS 7+

## Documentation

- `QUICKSTART_4DOF.md` - Quick start guide
- `README_4DOF_ALTERNATING.md` - Full documentation
- `LINUX_DEPLOYMENT_GUIDE.md` - Deployment details

## Training Time

~26 hours for 2M steps on RTX 5060 Ti (12 parallel envs)

## Files

- `config_4dof.py` - Configuration
- `env/alternating_lens_env.py` - Environment wrapper
- `train_4dof_alternating.py` - Training script
- `analyze_4dof_alternating.py` - Analysis tools
- `test_4dof_alternating.py` - System tests
- `deploy_linux.sh` - Deployment script

## Contact

For issues, check documentation or run `python3 test_4dof_alternating.py`

---

**Version**: v1.0  
**Date**: 2026-08-29
