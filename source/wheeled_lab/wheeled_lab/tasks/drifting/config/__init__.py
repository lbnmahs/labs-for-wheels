import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from .agents import *


def _make_mushr_drift_env(**kwargs):
    """Wrapper to create Mushr drift environment, filtering metadata kwargs."""
    # Filter out metadata kwargs that shouldn't be passed to the environment
    metadata_keys = {"env_cfg_entry_point", "rsl_rl_cfg_entry_point"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in metadata_keys}
    
    # If cfg is provided, use it; otherwise create from config class
    if "cfg" in filtered_kwargs:
        cfg = filtered_kwargs.pop("cfg")
    else:
        from wheeled_lab.tasks.drifting.mushr_drift_env_cfg import MushrDriftRLEnvCfg
        cfg = MushrDriftRLEnvCfg()
    
    return ManagerBasedRLEnv(cfg=cfg, **filtered_kwargs)


def _make_f1tenth_drift_env(**kwargs):
    """Wrapper to create F1Tenth drift environment, filtering metadata kwargs."""
    # Filter out metadata kwargs that shouldn't be passed to the environment
    metadata_keys = {"env_cfg_entry_point", "rsl_rl_cfg_entry_point"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in metadata_keys}
    
    # If cfg is provided, use it; otherwise create from config class
    if "cfg" in filtered_kwargs:
        cfg = filtered_kwargs.pop("cfg")
    else:
        from wheeled_lab.tasks.drifting.f1tenth_drift_env_cfg import F1TenthDriftRLEnvCfg
        cfg = F1TenthDriftRLEnvCfg()
    
    return ManagerBasedRLEnv(cfg=cfg, **filtered_kwargs)


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