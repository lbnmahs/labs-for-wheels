import gymnasium as gym
from wheeled_lab.tasks.common.utils import create_env_wrapper

from .agents import *

# Create wrapper functions using the common utility
_make_mushr_drift_env = create_env_wrapper(
    "wheeled_lab.tasks.drifting.mushr_drift_env_cfg:MushrDriftRLEnvCfg"
)

_make_f1tenth_drift_env = create_env_wrapper(
    "wheeled_lab.tasks.drifting.f1tenth_drift_env_cfg:F1TenthDriftRLEnvCfg"
)


gym.register(
    id="Template-WheeledLab-Mushr-Drift-v0",
    entry_point="wheeled_lab.tasks.drifting.config:_make_mushr_drift_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wheeled_lab.tasks.drifting.mushr_drift_env_cfg:MushrDriftRLEnvCfg",
        "rsl_rl_cfg_entry_point": "wheeled_lab.tasks.drifting.config.agents.mushr.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
    },
)

gym.register(
    id="Template-WheeledLab-F1Tenth-Drift-v0",
    entry_point="wheeled_lab.tasks.drifting.config:_make_f1tenth_drift_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wheeled_lab.tasks.drifting.f1tenth_drift_env_cfg:F1TenthDriftRLEnvCfg",
        "rsl_rl_cfg_entry_point": "wheeled_lab.tasks.drifting.config.agents.f1tenth.rsl_rl_ppo_cfg:F1TenthPPORunnerCfg",
    },
)