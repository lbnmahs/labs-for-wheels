"""Gymnasium environment registration for waypoint following task."""
import gymnasium as gym
from importlib import import_module

from . import agents  # noqa: F401


def _make_mushr_waypoint_env(**kwargs):
    """Create waypoint following environment with custom environment class."""
    from wheeled_lab.tasks.waypoint.mushr_waypoint_env import MushrWaypointEnv
    
    # Filter out metadata kwargs
    metadata_keys = {"env_cfg_entry_point", "rsl_rl_cfg_entry_point", "skrl_amp_cfg_entry_point", 
                     "skrl_ippo_cfg_entry_point", "skrl_mappo_cfg_entry_point", "skrl_cfg_entry_point"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in metadata_keys}
    
    # If cfg is provided, use it; otherwise create from config class
    if "cfg" in filtered_kwargs:
        cfg = filtered_kwargs.pop("cfg")
    else:
        module_path, class_name = "wheeled_lab.tasks.waypoint.mushr_waypoint_env_cfg:MushrWaypointEnvCfg".rsplit(':', 1)
        module = import_module(module_path)
        cfg_class = getattr(module, class_name)
        cfg = cfg_class()
    
    return MushrWaypointEnv(cfg=cfg, **filtered_kwargs)

gym.register(
    id="Template-WheeledLab-Mushr-Waypoint-v0",
    entry_point="wheeled_lab.tasks.waypoint.config:_make_mushr_waypoint_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wheeled_lab.tasks.waypoint.mushr_waypoint_env_cfg:MushrWaypointEnvCfg",
        "rsl_rl_cfg_entry_point": "wheeled_lab.tasks.waypoint.config.agents.mushr.rsl_rl_ppo_cfg:MushrWaypointPPORunnerCfg",
        "skrl_amp_cfg_entry_point": "wheeled_lab.tasks.waypoint.config.agents.mushr:skrl_amp_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "wheeled_lab.tasks.waypoint.config.agents.mushr:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "wheeled_lab.tasks.waypoint.config.agents.mushr:skrl_mappo_cfg.yaml",
        "skrl_cfg_entry_point": "wheeled_lab.tasks.waypoint.config.agents.mushr:skrl_ppo_cfg.yaml",
    },
)
