# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from wheeled_lab.assets.articulations.mushr import MUSHR_SUS_CFG, MUSHR_SUS_2WD_CFG
from .waypoints import WAYPOINT_CFG
from .markers import ROBOT_MARKER_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class MushrWaypointEnvCfg(DirectRLEnvCfg):
    """Configuration for MuSHR vehicle waypoint-following environment."""

    # Drive mode: "rwd" (2WD rear) or "4wd"
    drive_mode: str = "rwd"
    
    # Environment timing: decimation=4 means actions applied every 4 sim steps
    decimation = 4
    episode_length_s = 20.0  # Maximum episode duration [s]
    
    # Action space: [throttle, steering]
    action_space = 2
    # Observation space: [position_error, cos(heading_error), sin(heading_error),
    #                     lin_vel_x, lin_vel_y, ang_vel_z, throttle_state, steering_state]
    observation_space = 8
    state_space = 0  # No privileged state information
    
    # Simulation settings: 60 Hz physics, render every 4 steps (15 Hz)
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    
    # Robot and visualization configurations
    robot_cfg: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG  # Waypoint marker visualization
    robot_marker_cfg = ROBOT_MARKER_CFG  # Robot heading marker visualization

    # Joint names for action mapping
    throttle_dof_name = [
        "back_left_wheel_throttle",
        "back_right_wheel_throttle",
    ]
    steering_dof_name = [
        "front_left_wheel_steer",
        "front_right_wheel_steer",
    ]

    # Action scaling parameters
    throttle_scale = 10  # Scale factor for throttle action
    throttle_max = 50  # Maximum wheel velocity [rad/s]
    steering_scale = 0.1  # Scale factor for steering action
    steering_max = 0.75  # Maximum steering angle [rad]

    # Waypoint parameters
    num_waypoints = 10
    position_tolerance = 0.15  # Distance threshold to reach waypoint [m]
    course_length_coefficient = 2.5  # Course length scaling
    course_width_coefficient = 2.0  # Course width scaling

    # Reward weights
    goal_reached_bonus = 10.0  # Bonus for reaching a waypoint
    position_progress_weight = 1.0  # Weight for position progress
    heading_coefficient = 0.25  # Heading error exponential decay
    heading_progress_weight = 0.05  # Weight for heading alignment reward
    forward_velocity_weight = 0.2  # Weight for forward velocity reward

    # Scene configuration
    env_spacing = 32.0  # Distance between parallel environments [m]
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=env_spacing, replicate_physics=True
    )
    
    def __post_init__(self):
        """Post initialization for drive mode selection."""
        super().__post_init__()

        # Preserve prim path pattern
        prim_path = self.robot_cfg.prim_path

        if self.drive_mode.lower() == "4wd":
            # Use 4WD articulation config and all wheel throttles
            self.robot_cfg = MUSHR_SUS_CFG.replace(prim_path=prim_path)
            self.throttle_dof_name = [
                "back_left_wheel_throttle",
                "back_right_wheel_throttle",
                "front_left_wheel_throttle",
                "front_right_wheel_throttle",
            ]
        else:
            # Default: 2WD rear
            self.robot_cfg = MUSHR_SUS_2WD_CFG.replace(prim_path=prim_path)
            self.throttle_dof_name = [
                "back_left_wheel_throttle",
                "back_right_wheel_throttle",
            ]