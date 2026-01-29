from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from wheeled_lab.tasks.common.agents import BasePPORunnerCfg

@configclass
class F1TenthPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner configuration for F1Tenth drift task."""
    max_iterations = 1500
    experiment_name = "ppo_f1tenth"
    # All other settings inherited from BasePPORunnerCfg