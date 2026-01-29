import gymnasium as gym
from wheeled_lab.tasks.common.utils import create_env_wrapper

from .agents import *

# Create wrapper function using the common utility
_make_mushr_elevation_env = create_env_wrapper(
    "wheeled_lab.tasks.elevation.mushr_elevation_env_cfg:MushrElevationRLEnvCfg"
)


gym.register(
    id="Template-WheeledLab-Mushr-Elevation-v0",
    entry_point="wheeled_lab.tasks.elevation.config:_make_mushr_elevation_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wheeled_lab.tasks.elevation.mushr_elevation_env_cfg:MushrElevationRLEnvCfg",
        "rsl_rl_cfg_entry_point": "wheeled_lab.tasks.elevation.config.agents.mushr.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
    },
)
