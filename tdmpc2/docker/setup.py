from setuptools import setup, find_packages

setup(
    name="tdmpc2",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "dm-control==1.0.16",
        "glfw==2.7.0",
        "gymnasium==0.29.1",
        "ffmpeg==1.4",
        "imageio==2.34.1",
        "imageio-ffmpeg==0.4.9",
        "h5py==3.11.0",
        "hydra-core==1.3.2",
        "hydra-submitit-launcher==1.2.0",
        "submitit==1.5.1",
        "omegaconf==2.3.0",
        "moviepy==1.0.3",
        "mujoco==3.1.2",
        "numpy==1.24.4",
        "tensordict-nightly==2024.11.14",
        "torchrl-nightly==2024.11.14",
        "kornia==0.7.2",
        "termcolor==2.4.0",
        "tqdm==4.66.4",
        "pandas==2.0.3",
        "wandb==0.17.4",
    ],
    extras_require={
        "gym": [
            "cython<3",
            "wheel==0.38.0",
            "setuptools==65.5.0",
            "mujoco==2.3.1",
            "mujoco-py==2.1.2.14",
            "gym==0.21.0",
        ],
        "maniskill2": [
            "mani-skill2==0.4.1",
        ],
        "metaworld": [
            "git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb#egg=metaworld",
        ],
        "myosuite": [
            "myosuite",
        ],
    },
)
