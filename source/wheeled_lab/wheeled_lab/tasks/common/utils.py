"""Common utilities for task configuration."""
from typing import Type
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


def create_env_wrapper(cfg_class_path: str):
    """Create an environment wrapper function for a given config class.
    
    This function creates a wrapper that filters out metadata kwargs and properly
    instantiates the environment. It's used as the entry_point for gymnasium registrations.
    
    Args:
        cfg_class_path: Full import path to config class (e.g., 
            "wheeled_lab.tasks.drifting.mushr_drift_env_cfg:MushrDriftRLEnvCfg")
    
    Returns:
        A function that can be used as a gymnasium entry_point.
    """
    def _make_env(**kwargs):
        """Wrapper to create environment, filtering metadata kwargs."""
        from importlib import import_module
        
        # Filter out metadata kwargs that shouldn't be passed to the environment
        metadata_keys = {"env_cfg_entry_point", "rsl_rl_cfg_entry_point"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in metadata_keys}
        
        # If cfg is provided, use it; otherwise create from config class
        if "cfg" in filtered_kwargs:
            cfg = filtered_kwargs.pop("cfg")
        else:
            module_path, class_name = cfg_class_path.rsplit(':', 1)
            module = import_module(module_path)
            cfg_class = getattr(module, class_name)
            cfg = cfg_class()
        
        return ManagerBasedRLEnv(cfg=cfg, **filtered_kwargs)
    
    return _make_env
