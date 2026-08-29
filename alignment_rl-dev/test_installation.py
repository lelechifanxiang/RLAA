"""验证镜头对准RL环境安装"""
import sys
import torch
import gymnasium
import stable_baselines3
import optics_core

# 添加alignment_rl到路径
sys.path.insert(0, 'alignment_rl-dev')
from env.lens_env import LensAlignmentEnv

print("=" * 60)
print("镜头对准RL环境安装验证")
print("=" * 60)

print("\n[核心依赖]")
print(f"  PyTorch:           {torch.__version__}")
print(f"  Gymnasium:         {gymnasium.__version__}")
print(f"  Stable-Baselines3: {stable_baselines3.__version__}")

print("\n[自定义包]")
print(f"  optics_core:       已安装")
print(f"  alignment_rl:      已安装 (脚本模式)")

print("\n[环境测试]")
# 测试镜头对准环境
lens_env = LensAlignmentEnv()
obs, info = lens_env.reset()
print(f"  LensAlignmentEnv:  观测空间 {obs.shape}, 动作空间 {lens_env.action_space}")

print("\n[GPU支持]")
print(f"  CUDA可用:          {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU设备:           {torch.cuda.get_device_name(0)}")
else:
    print(f"  当前模式:          CPU (已安装CPU版PyTorch)")

print("\n" + "=" * 60)
print("所有测试通过! 环境配置完成")
print("=" * 60)
