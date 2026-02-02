"""Common utilities for task configuration."""
from isaaclab.envs import ManagerBasedRLEnv


# Task name mapping: short name -> base task name (without -v0 suffix)
TASK_NAME_MAP = {
    "waypoint": "Template-WheeledLab-Mushr-Waypoint",
    "drift": "Template-WheeledLab-Mushr-Drift",
    "visual": "Template-WheeledLab-Mushr-Visual",
    "elevation": "Template-WheeledLab-Mushr-Elevation",
}


def resolve_task_name(task_name: str, is_play: bool = False) -> str:
    """Resolve short task name to full gymnasium task name.

    Args:
        task_name: Short task name (e.g., "waypoint") or full task name
            (e.g., "Template-WheeledLab-Mushr-Waypoint-v0")
        is_play: If True, resolves to Play variant; if False, resolves to training variant.
            If Play variant doesn't exist, falls back to training variant.

    Returns:
        Full gymnasium task name with appropriate suffix.
    """
    # If already a full name, return as-is (but ensure correct Play suffix)
    if task_name.startswith("Template-"):
        # If it's a play script and task doesn't have -Play, try to add it
        if is_play and "-Play" not in task_name:
            # Remove -v0 suffix, add -Play-v0
            if task_name.endswith("-v0"):
                play_name = task_name[:-3] + "-Play-v0"
            else:
                play_name = task_name + "-Play-v0"

            # Check if Play environment exists, fall back to training if not
            try:
                import gymnasium as gym
                try:
                    gym.spec(play_name)
                    return play_name
                except gym.error.NameNotFound:
                    # Play environment doesn't exist, use training environment
                    return task_name
            except (ImportError, AttributeError):
                # If gymnasium not available or registry not populated, return Play name
                # (will fail later with a clearer error message)
                return play_name
        # If it's a train script and task has -Play, remove it
        elif not is_play and "-Play" in task_name:
            # Remove -Play-v0 suffix, add -v0
            if task_name.endswith("-Play-v0"):
                return task_name[:-8] + "-v0"
            return task_name.replace("-Play", "")
        return task_name

    # Resolve short name to base task name
    base_name = TASK_NAME_MAP.get(task_name.lower())
    if base_name is None:
        # If not found in map, assume it's already a full name or raise error
        raise ValueError(
            f"Unknown task name: '{task_name}'. "
            f"Available short names: {list(TASK_NAME_MAP.keys())}"
        )

    # Add appropriate suffix
    if is_play:
        play_name = f"{base_name}-Play-v0"
        # Check if Play environment exists, fall back to training if not
        try:
            import gymnasium as gym
            try:
                gym.spec(play_name)
                return play_name
            except gym.error.NameNotFound:
                # Play environment doesn't exist, use training environment
                return f"{base_name}-v0"
        except (ImportError, AttributeError):
            # If gymnasium not available or registry not populated, return Play name
            # (will fail later with a clearer error message)
            return play_name
    else:
        return f"{base_name}-v0"


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
