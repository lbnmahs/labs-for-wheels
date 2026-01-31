"""RSL-RL PPO configuration for MuSHR waypoint following task."""
from isaaclab.utils import configclass
from wheeled_lab.tasks.common.agents import BasePPORunnerCfg


@configclass
class MushrWaypointPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner configuration for MuSHR waypoint following task."""

    max_iterations = 200
    experiment_name = "ppo_mushr_waypoint"
    # All other settings inherited from BasePPORunnerCfg
