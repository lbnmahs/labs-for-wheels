from isaaclab.utils import configclass
from wheeled_lab.tasks.common.agents import BasePPORunnerCfg

@configclass
class MushrPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner configuration for Mushr drift task."""
    max_iterations = 150
    experiment_name = "ppo_mushr"
    # All other settings inherited from BasePPORunnerCfg
