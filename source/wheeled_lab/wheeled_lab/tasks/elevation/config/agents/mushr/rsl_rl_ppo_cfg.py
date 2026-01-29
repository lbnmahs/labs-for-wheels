from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from wheeled_lab.tasks.common.agents import BasePPORunnerCfg

@configclass
class MushrPPORunnerCfg(BasePPORunnerCfg):
    """PPO runner configuration for Mushr elevation task."""
    max_iterations = 4000
    experiment_name = "ppo_mushr_elevation"
    # Override activation to use relu instead of elu
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="relu",
    )
