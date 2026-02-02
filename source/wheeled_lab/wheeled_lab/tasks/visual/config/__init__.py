import gymnasium as gym
from wheeled_lab.tasks.common.utils import create_env_wrapper

from .agents import *

# Create wrapper function using the common utility
_make_mushr_visual_env = create_env_wrapper(
    "wheeled_lab.tasks.visual.mushr_visual_env_cfg:MushrVisualRLEnvCfg"
)


gym.register(
    id="Template-WheeledLab-Mushr-Visual-v0",
    entry_point="wheeled_lab.tasks.visual.config:_make_mushr_visual_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wheeled_lab.tasks.visual.mushr_visual_env_cfg:MushrVisualRLEnvCfg",
        "rsl_rl_cfg_entry_point": "wheeled_lab.tasks.visual.config.agents.mushr.rsl_rl_ppo_cfg:MushrPPORunnerCfg",
        "skrl_amp_cfg_entry_point": "wheeled_lab.tasks.visual.config.agents.mushr:skrl_amp_cfg.yaml",
        "skrl_ippo_cfg_entry_point": "wheeled_lab.tasks.visual.config.agents.mushr:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": "wheeled_lab.tasks.visual.config.agents.mushr:skrl_mappo_cfg.yaml",
        "skrl_cfg_entry_point": "wheeled_lab.tasks.visual.config.agents.mushr:skrl_ppo_cfg.yaml",
    },
)
