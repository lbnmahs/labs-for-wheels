# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_from_angle_axis

from .mushr_waypoint_env_cfg import MushrWaypointEnvCfg
from .waypoints import WAYPOINT_CFG
from .markers import ROBOT_MARKER_CFG


class MushrWaypointEnv(DirectRLEnv):
    """
    Waypoint-following environment for the MuSHR vehicle.
    
    Task: Drive the car to reach a sequence of waypoints while maintaining
    appropriate heading towards each target. Reward is based on progress
    toward waypoints and alignment with target heading.
    """
    cfg: MushrWaypointEnvCfg

    def __init__(self, cfg: MushrWaypointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Find joint indices for throttle (rear wheels) and steering (front wheels)
        # Note: mushr is created in _setup_scene(), so we access it after super().__init__()
        self._throttle_dof_idx, _ = self.mushr.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.mushr.find_joints(self.cfg.steering_dof_name)
        self._throttle_state = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        self._steering_state = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        # Task state tracking
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._num_goals = self.cfg.num_waypoints
        
        # Waypoint positions (x, y) and visualization markers (x, y, z)
        self._target_positions = torch.zeros(
            (self.num_envs, self._num_goals, 2),
            device=self.device,
            dtype=torch.float32,
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self._num_goals, 3),
            device=self.device,
            dtype=torch.float32,
        )
        
        # Environment parameters
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = self.cfg.course_length_coefficient
        self.course_width_coefficient = self.cfg.course_width_coefficient
        self.position_tolerance = self.cfg.position_tolerance
        
        # Reward weights
        self.goal_reached_bonus = self.cfg.goal_reached_bonus
        self.position_progress_weight = self.cfg.position_progress_weight
        self.heading_coefficient = self.cfg.heading_coefficient
        self.heading_progress_weight = self.cfg.heading_progress_weight
        self.forward_velocity_weight = self.cfg.forward_velocity_weight
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )

        # Marker visualization state
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.marker_locations = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
        # Offset 0.5m above robot
        self.marker_offset[:, -1] = 0.5
        self.forward_marker_orientations = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.target_marker_orientations = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.target_heading_w = torch.zeros(
            (self.num_envs), device=self.device
        )

    def _setup_scene(self):
        """Setup the simulation scene: ground plane, robot, and visualizations."""
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # Setup rest of the scene
        self.mushr = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(WAYPOINT_CFG)
        self.robot_markers = VisualizationMarkers(ROBOT_MARKER_CFG)
        self.object_state = []
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["mushr"] = self.mushr

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _visualize_markers(self):
        """
        Visualize robot heading markers (cyan=current direction, red=target direction).
        
        Shows two arrows above each robot: one for current heading and one pointing
        toward the target waypoint. Updated every physics step.
        """
        # Get robot position and set marker locations
        self.marker_locations = self.mushr.data.root_pos_w
        loc = self.marker_locations + self.marker_offset

        # Get robot's current heading
        heading = self.mushr.data.heading_w

        # Forward marker orientation (robot's current heading)
        forward_yaws = heading.unsqueeze(dim=1)
        self.forward_marker_orientations = quat_from_angle_axis(
            forward_yaws, self.up_dir
        ).squeeze()

        # Target marker orientation (heading towards waypoint)
        if hasattr(self, "target_heading_w"):
            target_yaws = self.target_heading_w.unsqueeze(dim=1)
            self.target_marker_orientations = quat_from_angle_axis(
                target_yaws, self.up_dir
            ).squeeze()
        else:
            # Fallback if target heading not yet computed
            self.target_marker_orientations = (
                self.forward_marker_orientations.clone()
            )

        # Stack positions and orientations for both marker types
        all_locations = torch.vstack((loc, loc))
        all_orientations = torch.vstack(
            (self.forward_marker_orientations, self.target_marker_orientations)
        )

        # Create marker indices: 0 for forward, 1 for target
        all_envs = torch.arange(self.cfg.scene.num_envs, device=self.device)
        marker_indices = torch.hstack(
            (torch.zeros_like(all_envs), torch.ones_like(all_envs))
        )

        # Visualize markers
        self.robot_markers.visualize(
            all_locations, all_orientations, marker_indices=marker_indices
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Process raw actions into vehicle controls before physics step.
        
        Actions: [throttle (-1 to 1), steering (-1 to 1)]
        - Throttle: scaled to velocity targets for rear 2 wheels
        - Steering: scaled to position targets for front 2 wheels
        """
        # Action scaling parameters
        throttle_scale = self.cfg.throttle_scale  # Scale factor for throttle action
        throttle_max = self.cfg.throttle_max  # Maximum wheel velocity [rad/s]
        steering_scale = self.cfg.steering_scale  # Scale factor for steering action
        steering_max = self.cfg.steering_max  # Maximum steering angle [rad]

        # Apply throttle to rear 2 wheels (same value per wheel)
        self._throttle_action = actions[:, 0].repeat_interleave(2).reshape((-1, 2)) * throttle_scale
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max)
        self._throttle_state = self._throttle_action
        
        # Apply steering to front 2 wheels (same value per wheel)
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action
        
        # Update marker visualization
        self._visualize_markers()

    def _apply_action(self) -> None:
        """Apply processed actions to robot joints."""
        # Throttle: velocity control for wheel rotation
        self.mushr.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        # Steering: position control for front wheel angle
        self.mushr.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        """
        Compute observations: position error, heading error, velocities, and action states.
        
        Returns 8D observation: [position_error, cos(heading_error), sin(heading_error),
                                 lin_vel_x, lin_vel_y, ang_vel_z, throttle, steering]
        """
        # Compute position error to current target waypoint
        current_target_positions = self._target_positions[self.mushr._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.mushr.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        # Compute heading error: difference between current and target heading
        heading = self.mushr.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.mushr._ALL_INDICES, self._target_index, 1] - self.mushr.data.root_link_pos_w[:, 1],
            self._target_positions[self.mushr._ALL_INDICES, self._target_index, 0] - self.mushr.data.root_link_pos_w[:, 0],
        )
        # Wrap heading error to [-pi, pi]
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        
        # Store target heading for visualization
        self.target_heading_w = target_heading_w

        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self.mushr.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.mushr.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.mushr.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute reward based on progress to waypoint and heading alignment.
        
        Reward components:
        1. Position progress: distance reduction to target
        2. Heading alignment: exponential of heading error
        3. Forward velocity: velocity toward target
        4. Goal bonus: fixed bonus when waypoint reached
        """
        # Progress reward: positive when getting closer to target
        position_progress_rew = self._previous_position_error - self._position_error
        # Heading alignment reward: higher when pointing toward target
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        
        # Forward velocity reward: velocity toward target
        direction_to_target = self._position_error_vector
        distance_to_target = torch.norm(direction_to_target, dim=-1, keepdim=True)
        direction_to_target_normalized = direction_to_target / (distance_to_target + 1e-6)
        robot_vel_xy = self.mushr.data.root_lin_vel_w[:, :2]
        forward_velocity_rew = torch.sum(robot_vel_xy * direction_to_target_normalized, dim=-1)
        forward_velocity_rew = torch.clamp(forward_velocity_rew, min=0.0)
        
        # Check if waypoint reached and advance to next target
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            forward_velocity_rew * self.forward_velocity_weight +
            goal_reached * self.goal_reached_bonus
        )

        # Update waypoint visualization
        marker_indices = []
        for env_idx in range(self.num_envs):
            current_idx = self._target_index[env_idx].item()
            for wp_idx in range(self._num_goals):
                if wp_idx == current_idx:
                    marker_indices.append(0)  # current (red)
                else:
                    marker_indices.append(1)  # future (green)
        self.waypoints.visualize(
            translations=self._markers_pos.view(-1, 3),
            marker_indices=marker_indices
        )

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Reset specified environments: randomize robot pose and generate new waypoint course.
        
        Robot is placed at random position and orientation. Waypoints are generated
        along a curved path in front of the starting position.
        """
        if env_ids is None:
            env_ids = self.mushr._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        
        # Get default robot state
        default_state = self.mushr.data.default_root_state[env_ids]
        mushr_pose = default_state[:, :7]  # Position + quaternion
        mushr_velocities = default_state[:, 7:]  # Linear + angular velocity
        joint_positions = self.mushr.data.default_joint_pos[env_ids]
        joint_velocities = self.mushr.data.default_joint_vel[env_ids]

        # Randomize robot position: offset from env origin with lateral variation
        mushr_pose[:, :3] += self.scene.env_origins[env_ids]
        mushr_pose[:, 0] -= self.env_spacing / 2  # Start behind origin
        mushr_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        # Randomize robot orientation: ±30 degrees from forward
        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        mushr_pose[:, 3] = torch.cos(angles * 0.5)  # Quaternion w
        mushr_pose[:, 6] = torch.sin(angles * 0.5)  # Quaternion z

        self.mushr.write_root_pose_to_sim(mushr_pose, env_ids)
        self.mushr.write_root_velocity_to_sim(mushr_velocities, env_ids)
        self.mushr.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        # Generate waypoint course: curved path in front of robot
        spacing = 2 / self._num_goals
        # X positions: curved path from behind to ahead
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        # Y positions: random variation to create curves
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) * self.course_width_coefficient
        # Offset waypoints to environment origin
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        self._markers_pos[env_ids, :, 2] = 0.5  # Set Z-height for visualization
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[self.mushr._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.mushr.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.mushr.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._target_positions[:, 0, 1] - self.mushr.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.mushr.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(
            torch.sin(target_heading_w - heading),
            torch.cos(target_heading_w - heading),
        )
        self._previous_heading_error = self._heading_error.clone()

        # Initialize target heading for visualization
        self.target_heading_w[env_ids] = target_heading_w[env_ids]

        # Visualize markers after reset
        self._visualize_markers()
