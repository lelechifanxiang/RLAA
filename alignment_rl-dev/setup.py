from setuptools import setup, find_packages

setup(
    name="alignment_rl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "sb3-contrib>=2.0.0",
        "torch>=2.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "tensorboard>=2.10.0",
        "tqdm>=4.60.0",
        "imageio>=2.9.0",
    ],
    description="Reinforcement learning for optical alignment tasks",
    author="alignment_rl team",
)
