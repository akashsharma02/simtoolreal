# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""SimToolReal Isaac Lab environment for dexterous tool manipulation.

This is the Isaac Lab (DirectRLEnv) port of the original Isaac Gym environment
at isaacgymenvs/tasks/simtoolreal/env.py.

Migration status (Phase 1):
  [x] Scene setup (robot, table, object, goal visualization)
  [x] State tensor access (joint states, rigid body states, root states)
  [x] Action application (position targets with moving average)
  [x] Observation computation (policy + critic for asymmetric)
  [x] Reward computation (lifting, keypoint, fingertip delta, action penalties)
  [x] Reset logic (randomized object/robot poses, goal sampling)
  [x] Random force/torque perturbations on object
  [ ] Domain randomization (EventCfg wired but disabled by default)
  [ ] Observation/action delay queues
  [ ] Procedural object generation (handle_head_primitives)
  [ ] Camera sensor / video capture
  [ ] Good-reset-boundary state buffer
  [ ] Tyler curriculum
  [ ] Keyboard interactive controls
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
    sample_uniform,
    saturate,
)

if TYPE_CHECKING:
    from .env_cfg import SimToolRealEnvCfg


class SimToolRealEnv(DirectRLEnv):
    """SimToolReal environment ported to Isaac Lab DirectRLEnv.

    This environment trains a policy for dexterous tool manipulation using
    an IIWA arm + Sharpa hand. The task is to manipulate a tool object to
    match a sequence of goal poses (position + orientation).

    Observations are structured for asymmetric actor-critic training:
    - "policy": observations available to the actor (no privileged info)
    - "critic": full state observations for the critic (includes velocities, etc.)
    """

    cfg: SimToolRealEnvCfg

    def __init__(self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs):
        # Compute observation/state sizes from obs_list/state_list before super().__init__
        obs_type_sizes = self._get_obs_type_sizes(cfg)
        cfg.observation_space = sum(obs_type_sizes[k] for k in cfg.obs_list)
        cfg.state_space = sum(obs_type_sizes[k] for k in cfg.state_list)

        super().__init__(cfg, render_mode, **kwargs)

        # Robot DOF counts
        self.num_arm_dofs = cfg.num_arm_dofs
        self.num_hand_dofs = cfg.num_hand_dofs
        self.num_hand_arm_dofs = self.num_arm_dofs + self.num_hand_dofs
        self.num_fingertips = cfg.num_fingertips
        self.num_keypoints = cfg.num_keypoints

        # Find joint and body indices
        self._setup_joint_indices()
        self._setup_body_indices()

        # Joint limits
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower_limits = joint_pos_limits[0, :self.num_hand_arm_dofs, 0]
        self.dof_upper_limits = joint_pos_limits[0, :self.num_hand_arm_dofs, 1]

        # Default joint positions from robot init state
        self.hand_arm_default_dof_pos = torch.zeros(
            self.num_hand_arm_dofs, dtype=torch.float, device=self.device
        )
        # Arm defaults
        arm_defaults = torch.tensor(
            [-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308],
            device=self.device,
        )
        self.hand_arm_default_dof_pos[:7] = arm_defaults

        # Action targets (with moving average smoothing)
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_arm_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_hand_arm_dofs), dtype=torch.float, device=self.device
        )

        # Object keypoint offsets (scaled by object_base_size * keypoint_scale)
        kp_offsets = torch.tensor(cfg.keypoint_offsets, dtype=torch.float, device=self.device)
        # Shape: (num_keypoints, 3), scale by half object_base_size * keypoint_scale
        self.keypoint_offsets_base = kp_offsets * (cfg.object_base_size / 2.0 * cfg.keypoint_scale)
        # Expand to (num_envs, num_keypoints, 3)
        self.object_keypoint_offsets = self.keypoint_offsets_base.unsqueeze(0).expand(
            self.num_envs, -1, -1
        ).clone()

        # Fixed-size keypoint offsets for reward
        fixed = torch.tensor(cfg.fixed_size, dtype=torch.float, device=self.device)
        fixed_kp = kp_offsets * (fixed / 2.0 * cfg.keypoint_scale).unsqueeze(0)
        self.object_keypoint_offsets_fixed_size = fixed_kp.unsqueeze(0).expand(
            self.num_envs, -1, -1
        ).clone()

        # Palm offset
        self.palm_offset = torch.tensor(cfg.palm_offset, dtype=torch.float, device=self.device)

        # Fingertip offsets
        self.fingertip_offsets_np = torch.tensor(
            cfg.fingertip_offsets, dtype=torch.float, device=self.device
        ).unsqueeze(0).expand(self.num_envs, -1, -1).clone()

        # Goal state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        self.goal_states = torch.zeros(
            (self.num_envs, 13), dtype=torch.float, device=self.device
        )

        # Object initial state (set during reset)
        self.object_init_state = torch.zeros(
            (self.num_envs, 13), dtype=torch.float, device=self.device
        )

        # Keypoint position buffers
        self.obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )
        self.goal_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )
        self.obj_keypoint_pos_fixed_size = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )
        self.goal_keypoint_pos_fixed_size = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )

        # Tracking buffers
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.closest_keypoint_max_dist_fixed_size = -torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.closest_fingertip_dist = -torch.ones(
            (self.num_envs, self.num_fingertips), dtype=torch.float, device=self.device
        )

        # Object scale noise (for observation augmentation)
        self.object_scale_noise_multiplier = torch.ones(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )

        # Random force/torque buffers
        self.rb_forces = torch.zeros(
            (self.num_envs, 1, 3), dtype=torch.float, device=self.device
        )
        self.rb_torques = torch.zeros(
            (self.num_envs, 1, 3), dtype=torch.float, device=self.device
        )

        # Success tolerance (with curriculum)
        self.success_tolerance = cfg.success_tolerance

        # Target volume for goal sampling
        mins = torch.tensor(cfg.target_volume_mins, dtype=torch.float, device=self.device)
        maxs = torch.tensor(cfg.target_volume_maxs, dtype=torch.float, device=self.device)
        self.target_volume_min = mins
        self.target_volume_max = maxs

        # Unit tensors for quaternion operations
        self.x_unit = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).expand(
            self.num_envs, -1
        )
        self.y_unit = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).expand(
            self.num_envs, -1
        )
        self.z_unit = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).expand(
            self.num_envs, -1
        )

        # Reward tracking
        self.reward_keys = [
            "fingertip_delta_rew", "lifting_rew", "lift_bonus_rew",
            "keypoint_rew", "bonus_rew",
            "kuka_actions_penalty", "hand_actions_penalty",
            "object_lin_vel_penalty", "object_ang_vel_penalty",
            "total_reward",
        ]
        self.rewards_episode = {
            k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for k in self.reward_keys
        }

        # Store previous fingertip distances for delta reward
        self.prev_fingertip_distances = None

    # ------------------------------------------------------------------
    # Helpers for obs size computation (called before super().__init__)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_obs_type_sizes(cfg: SimToolRealEnvCfg) -> dict[str, int]:
        n_arm = cfg.num_arm_dofs
        n_hand = cfg.num_hand_dofs
        n_dofs = n_arm + n_hand
        n_ft = cfg.num_fingertips
        n_kp = cfg.num_keypoints
        return {
            "joint_pos": n_dofs,
            "joint_vel": n_dofs,
            "prev_action_targets": n_dofs,
            "palm_pos": 3,
            "palm_rot": 4,
            "palm_vel": 6,
            "object_rot": 4,
            "object_vel": 6,
            "fingertip_pos_rel_palm": 3 * n_ft,
            "keypoints_rel_palm": 3 * n_kp,
            "keypoints_rel_goal": 3 * n_kp,
            "object_scales": 3,
            "closest_keypoint_max_dist": 1,
            "closest_fingertip_dist": n_ft,
            "lifted_object": 1,
            "progress": 1,
            "successes": 1,
            "reward": 1,
        }

    # ------------------------------------------------------------------
    # Joint / body index setup
    # ------------------------------------------------------------------
    def _setup_joint_indices(self):
        """Find arm and hand joint indices in the articulation."""
        # Arm joints
        arm_joint_names = [f"iiwa14_joint_{i}" for i in range(1, 8)]
        self.arm_joint_ids = []
        for name in arm_joint_names:
            idx = self.robot.joint_names.index(name)
            self.arm_joint_ids.append(idx)

        # All hand+arm joint ids (first num_hand_arm_dofs joints)
        self.hand_arm_joint_ids = list(range(self.num_hand_arm_dofs))

    def _setup_body_indices(self):
        """Find fingertip and palm body indices."""
        self.fingertip_body_ids = []
        for name in self.cfg.fingertip_body_names:
            idx = self.robot.body_names.index(name)
            self.fingertip_body_ids.append(idx)
        self.fingertip_body_ids.sort()

        self.palm_body_id = self.robot.body_names.index(self.cfg.palm_body_name)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        """Create simulation scene with robot, table, object, and ground."""
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)
        # Table
        self.table = Articulation(self.cfg.table_cfg)
        # Manipulated object
        self.object = RigidObject(self.cfg.object_cfg)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register with scene
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["table"] = self.table
        self.scene.rigid_objects["object"] = self.object

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process raw actions before physics step.

        Actions are in [-1, 1] and mapped to joint position targets with
        exponential moving average smoothing.
        """
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply position targets to the robot joints.

        Called `decimation` times per RL step. Implements the same control
        scheme as the original: hand joints use absolute position targets
        scaled from [-1,1] to joint limits; arm joints use relative
        velocity-like targets.
        """
        actions = self.actions
        dof_speed_scale = self.cfg.dof_speed_scale
        hand_ma = self.cfg.hand_moving_average
        arm_ma = self.cfg.arm_moving_average
        dt = self.cfg.sim.dt

        # Split actions into arm and hand
        arm_actions = actions[:, :self.num_arm_dofs]
        hand_actions = actions[:, self.num_arm_dofs:self.num_hand_arm_dofs]

        # Hand: scale from [-1,1] to joint limits
        hand_lower = self.dof_lower_limits[self.num_arm_dofs:]
        hand_upper = self.dof_upper_limits[self.num_arm_dofs:]
        hand_targets = 0.5 * (hand_actions + 1.0) * (hand_upper - hand_lower) + hand_lower

        # Moving average for hand
        hand_targets = (
            hand_ma * hand_targets
            + (1.0 - hand_ma) * self.prev_targets[:, self.num_arm_dofs:]
        )

        # Arm: relative control (delta from previous target)
        arm_delta = arm_actions * dof_speed_scale * dt
        arm_targets = self.prev_targets[:, :self.num_arm_dofs] + arm_delta
        arm_targets = (
            arm_ma * arm_targets
            + (1.0 - arm_ma) * self.prev_targets[:, :self.num_arm_dofs]
        )

        # Combine and clamp
        self.cur_targets[:, :self.num_arm_dofs] = arm_targets
        self.cur_targets[:, self.num_arm_dofs:] = hand_targets
        self.cur_targets = torch.clamp(
            self.cur_targets,
            self.dof_lower_limits.unsqueeze(0),
            self.dof_upper_limits.unsqueeze(0),
        )

        self.prev_targets[:] = self.cur_targets

        # Apply to robot
        self.robot.set_joint_position_target(
            self.cur_targets, joint_ids=self.hand_arm_joint_ids
        )

        # Apply random forces/torques to object
        self._apply_random_perturbations()

    def _apply_random_perturbations(self):
        """Apply random forces and torques to the object for robustness training."""
        if self.cfg.force_scale == 0.0 and self.cfg.torque_scale == 0.0:
            return

        # Sample whether to apply force/torque this step
        force_prob = self._sample_log_uniform(
            self.cfg.force_prob_range[0], self.cfg.force_prob_range[1]
        )
        apply_force = torch.rand(self.num_envs, device=self.device) < force_prob

        if self.cfg.force_only_when_lifted:
            apply_force = apply_force & self.lifted_object

        # Sample random force direction and apply
        if apply_force.any():
            force_dir = torch.randn(self.num_envs, 3, device=self.device)
            force_dir = force_dir / (force_dir.norm(dim=-1, keepdim=True) + 1e-8)
            force_mag = self.cfg.force_scale
            new_forces = force_dir * force_mag
            self.rb_forces[:, 0, :] = torch.where(
                apply_force.unsqueeze(-1), new_forces, self.rb_forces[:, 0, :] * self.cfg.force_decay
            )

        # Similar for torques
        torque_prob = self._sample_log_uniform(
            self.cfg.torque_prob_range[0], self.cfg.torque_prob_range[1]
        )
        apply_torque = torch.rand(self.num_envs, device=self.device) < torque_prob
        if self.cfg.torque_only_when_lifted:
            apply_torque = apply_torque & self.lifted_object

        if apply_torque.any():
            torque_dir = torch.randn(self.num_envs, 3, device=self.device)
            torque_dir = torque_dir / (torque_dir.norm(dim=-1, keepdim=True) + 1e-8)
            torque_mag = self.cfg.torque_scale
            new_torques = torque_dir * torque_mag
            self.rb_torques[:, 0, :] = torch.where(
                apply_torque.unsqueeze(-1), new_torques, self.rb_torques[:, 0, :] * self.cfg.torque_decay
            )

        # Apply forces and torques to object via wrench
        forces_and_torques = torch.cat([self.rb_forces, self.rb_torques], dim=-1)  # (N, 1, 6)
        self.object.set_external_force_and_torque(
            forces=self.rb_forces.squeeze(1),
            torques=self.rb_torques.squeeze(1),
        )

    def _sample_log_uniform(self, low: float, high: float) -> torch.Tensor:
        """Sample a scalar log-uniformly per environment."""
        log_low = math.log(low)
        log_high = math.log(high)
        return torch.exp(
            torch.rand(self.num_envs, device=self.device) * (log_high - log_low) + log_low
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        """Compute policy and critic observations."""
        self._compute_intermediate_values()

        obs_dict = self._build_obs_dict()

        # Build policy observation (concatenated from obs_list)
        policy_obs_parts = [obs_dict[k] for k in self.cfg.obs_list]
        policy_obs = torch.cat(policy_obs_parts, dim=-1)

        # Clamp observations
        policy_obs = torch.clamp(
            policy_obs, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations
        )

        result = {"policy": policy_obs}

        # Build critic observation if state_space > 0 (asymmetric)
        if self.cfg.state_space > 0:
            state_obs_parts = [obs_dict[k] for k in self.cfg.state_list]
            state_obs = torch.cat(state_obs_parts, dim=-1)
            state_obs = torch.clamp(
                state_obs, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations
            )
            result["critic"] = state_obs

        return result

    def _build_obs_dict(self) -> dict[str, torch.Tensor]:
        """Build observation dictionary with all observation components."""
        dof_pos = self.robot.data.joint_pos[:, :self.num_hand_arm_dofs]
        dof_vel = self.robot.data.joint_vel[:, :self.num_hand_arm_dofs]

        # Unscale joint positions to [-1, 1]
        unscaled_dof_pos = (
            (2.0 * dof_pos - self.dof_upper_limits - self.dof_lower_limits)
            / (self.dof_upper_limits - self.dof_lower_limits + 1e-8)
        )

        # Unscale prev_action_targets
        unscaled_prev_targets = (
            (2.0 * self.prev_targets - self.dof_upper_limits - self.dof_lower_limits)
            / (self.dof_upper_limits - self.dof_lower_limits + 1e-8)
        )

        # Object scales (as observation)
        # In training, these come from the procedural object generation
        # For now, use a default representing the object_base_size
        object_scales = torch.ones(
            self.num_envs, 3, device=self.device
        ) * self.cfg.object_base_size / self.cfg.object_base_size  # normalized to 1

        obs = {
            "joint_pos": unscaled_dof_pos,
            "joint_vel": dof_vel,
            "prev_action_targets": unscaled_prev_targets,
            "palm_pos": self.palm_center_pos,
            "palm_rot": self.palm_rot,
            "palm_vel": self.palm_vel,
            "object_rot": self.object_rot,
            "object_vel": torch.cat([self.object_linvel, self.object_angvel], dim=-1),
            "fingertip_pos_rel_palm": self.fingertip_pos_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_palm": self.keypoints_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_goal": self.keypoints_rel_goal.reshape(self.num_envs, -1),
            "object_scales": object_scales * self.object_scale_noise_multiplier,
            "closest_keypoint_max_dist": self.closest_keypoint_max_dist.unsqueeze(-1),
            "closest_fingertip_dist": self.closest_fingertip_dist,
            "lifted_object": self.lifted_object.float().unsqueeze(-1),
            "progress": (self.episode_length_buf / self.max_episode_length).unsqueeze(-1),
            "successes": self.successes.unsqueeze(-1),
            "reward": self.reward_buf.unsqueeze(-1) if hasattr(self, 'reward_buf') else torch.zeros(self.num_envs, 1, device=self.device),
        }

        return obs

    def _compute_intermediate_values(self):
        """Compute derived quantities from simulation state."""
        # Object state (from rigid object)
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # Goal state
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # Palm state
        palm_pos_w = self.robot.data.body_pos_w[:, self.palm_body_id]
        self.palm_rot = self.robot.data.body_quat_w[:, self.palm_body_id]
        palm_offset_expanded = self.palm_offset.unsqueeze(0).expand(self.num_envs, -1)
        self.palm_center_pos = palm_pos_w - self.scene.env_origins + quat_rotate(
            self.palm_rot, palm_offset_expanded
        )
        self.palm_vel = self.robot.data.body_vel_w[:, self.palm_body_id]  # (N, 6)

        # Fingertip positions
        ft_pos_w = self.robot.data.body_pos_w[:, self.fingertip_body_ids]  # (N, num_ft, 3)
        ft_rot_w = self.robot.data.body_quat_w[:, self.fingertip_body_ids]  # (N, num_ft, 4)
        ft_pos = ft_pos_w - self.scene.env_origins.unsqueeze(1)

        # Apply fingertip offsets
        ft_pos_offset = torch.zeros_like(ft_pos)
        for i in range(self.num_fingertips):
            ft_pos_offset[:, i] = ft_pos[:, i] + quat_rotate(
                ft_rot_w[:, i], self.fingertip_offsets_np[:, i]
            )

        # Fingertip positions relative to palm
        palm_center_expanded = self.palm_center_pos.unsqueeze(1).expand_as(ft_pos_offset)
        self.fingertip_pos_rel_palm = ft_pos_offset - palm_center_expanded

        # Fingertip distances to object (for delta reward)
        obj_pos_expanded = self.object_pos.unsqueeze(1).expand_as(ft_pos_offset)
        fingertip_pos_rel_object = ft_pos_offset - obj_pos_expanded
        self.curr_fingertip_distances = torch.norm(fingertip_pos_rel_object, dim=-1)

        # Track closest fingertip distances
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0,
            self.curr_fingertip_distances,
            self.closest_fingertip_dist,
        )

        # Compute keypoints
        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + quat_rotate(
                self.object_rot,
                self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier,
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot,
                self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier,
            )
            self.obj_keypoint_pos_fixed_size[:, i] = self.object_pos + quat_rotate(
                self.object_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )
            self.goal_keypoint_pos_fixed_size[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_keypoint_offsets_fixed_size[:, i]
            )

        # Keypoint distances
        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        self.keypoints_rel_goal_fixed_size = (
            self.obj_keypoint_pos_fixed_size - self.goal_keypoint_pos_fixed_size
        )

        palm_for_kp = self.palm_center_pos.unsqueeze(1).expand_as(self.obj_keypoint_pos)
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_for_kp

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        self.keypoint_distances_l2_fixed_size = torch.norm(
            self.keypoints_rel_goal_fixed_size, dim=-1
        )
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values
        self.keypoints_max_dist_fixed_size = self.keypoint_distances_l2_fixed_size.max(
            dim=-1
        ).values

        # Track closest keypoint distance in episode
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0,
            self.keypoints_max_dist,
            self.closest_keypoint_max_dist,
        )
        self.closest_keypoint_max_dist_fixed_size = torch.where(
            self.closest_keypoint_max_dist_fixed_size < 0.0,
            self.keypoints_max_dist_fixed_size,
            self.closest_keypoint_max_dist_fixed_size,
        )

        # Check if object is lifted
        table_z = 0.38 + 0.25  # table height + object offset
        self.lifted_object = self.object_pos[:, 2] > (table_z + self.cfg.lifting_bonus_threshold)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward for all environments."""
        return self._compute_reward()

    def _compute_reward(self) -> torch.Tensor:
        """Full reward computation matching original SimToolReal."""
        cfg = self.cfg

        # -- Lifting reward --
        table_z = 0.38 + 0.25  # table height + object offset
        lift_height = self.object_pos[:, 2] - table_z
        lifting_rew = torch.clamp(lift_height, min=0.0)
        lift_bonus = (lift_height > cfg.lifting_bonus_threshold).float() * cfg.lifting_bonus

        # -- Fingertip delta reward --
        if self.prev_fingertip_distances is not None:
            fingertip_delta = self.prev_fingertip_distances - self.curr_fingertip_distances
            fingertip_delta_rew = fingertip_delta.sum(dim=-1)
        else:
            fingertip_delta_rew = torch.zeros(self.num_envs, device=self.device)
        self.prev_fingertip_distances = self.curr_fingertip_distances.clone()

        # -- Keypoint reward --
        if cfg.fixed_size_keypoint_reward:
            kp_dist = self.keypoints_max_dist_fixed_size
        else:
            kp_dist = self.keypoints_max_dist
        keypoint_rew = -kp_dist

        # -- Success detection --
        keypoint_success_tolerance = self.success_tolerance * cfg.keypoint_scale
        if cfg.fixed_size_keypoint_reward:
            near_goal = self.keypoints_max_dist_fixed_size <= keypoint_success_tolerance
        else:
            near_goal = self.keypoints_max_dist <= keypoint_success_tolerance

        if cfg.force_consecutive_near_goal_steps:
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= cfg.success_steps
        self.successes += is_success

        # -- Action penalties --
        arm_actions = self.actions[:, :self.num_arm_dofs]
        hand_actions = self.actions[:, self.num_arm_dofs:self.num_hand_arm_dofs]
        kuka_actions_penalty = -cfg.kuka_actions_penalty_scale * torch.sum(arm_actions ** 2, dim=-1)
        hand_actions_penalty = -cfg.hand_actions_penalty_scale * torch.sum(hand_actions ** 2, dim=-1)

        # -- Object velocity penalties --
        object_lin_vel_penalty = -cfg.object_lin_vel_penalty_scale * torch.sum(
            self.object_linvel ** 2, dim=-1
        )
        object_ang_vel_penalty = -cfg.object_ang_vel_penalty_scale * torch.sum(
            self.object_angvel ** 2, dim=-1
        )

        # -- Bonus for being near goal --
        if cfg.force_consecutive_near_goal_steps:
            bonus_rew = is_success.float() * cfg.reach_goal_bonus
        else:
            bonus_rew = near_goal.float() * (cfg.reach_goal_bonus / cfg.success_steps)

        # -- Combine all rewards --
        reward = (
            cfg.distance_delta_rew_scale * fingertip_delta_rew
            + cfg.lifting_rew_scale * lifting_rew
            + lift_bonus
            + cfg.keypoint_rew_scale * keypoint_rew
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )

        # Track episode rewards
        self.rewards_episode["fingertip_delta_rew"] += cfg.distance_delta_rew_scale * fingertip_delta_rew
        self.rewards_episode["lifting_rew"] += cfg.lifting_rew_scale * lifting_rew
        self.rewards_episode["lift_bonus_rew"] += lift_bonus
        self.rewards_episode["keypoint_rew"] += cfg.keypoint_rew_scale * keypoint_rew
        self.rewards_episode["bonus_rew"] += bonus_rew
        self.rewards_episode["kuka_actions_penalty"] += kuka_actions_penalty
        self.rewards_episode["hand_actions_penalty"] += hand_actions_penalty
        self.rewards_episode["object_lin_vel_penalty"] += object_lin_vel_penalty
        self.rewards_episode["object_ang_vel_penalty"] += object_ang_vel_penalty
        self.rewards_episode["total_reward"] += reward

        # Reset goals for successful environments
        goal_env_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target(goal_env_ids, is_first_goal=False)

        # Log extras
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["successes"] = self.prev_episode_successes.mean()
        self.extras["log"]["closest_keypoint_max_dist"] = self.closest_keypoint_max_dist.mean()

        return reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which environments should be reset."""
        # Fall detection: object too far from initial position
        fall_dist = torch.norm(
            self.object_pos - self.object_init_state[:, :3], dim=-1
        )
        terminated = fall_dist >= self.cfg.fall_distance

        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Max consecutive successes reached
        if self.cfg.max_consecutive_successes > 0:
            max_success = self.successes >= self.cfg.max_consecutive_successes
            time_out = time_out | max_success

        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # Save previous episode successes
        self.prev_episode_successes[env_ids] = self.successes[env_ids]

        # Reset tracking buffers
        self.successes[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.lifted_object[env_ids] = False
        self.closest_keypoint_max_dist[env_ids] = -1
        self.closest_keypoint_max_dist_fixed_size[env_ids] = -1
        self.closest_fingertip_dist[env_ids] = -1
        self.rb_forces[env_ids] = 0
        self.rb_torques[env_ids] = 0
        for k in self.rewards_episode:
            self.rewards_episode[k][env_ids] = 0

        # Reset robot to default + noise
        default_dof_pos = self.hand_arm_default_dof_pos.unsqueeze(0).expand(num_resets, -1).clone()

        # Add noise to arm joints
        arm_noise = sample_uniform(
            -self.cfg.reset_dof_pos_noise_arm,
            self.cfg.reset_dof_pos_noise_arm,
            (num_resets, self.num_arm_dofs),
            device=self.device,
        )
        default_dof_pos[:, :self.num_arm_dofs] += arm_noise

        # Add noise to hand joints
        hand_noise = sample_uniform(
            -self.cfg.reset_dof_pos_noise_fingers,
            self.cfg.reset_dof_pos_noise_fingers,
            (num_resets, self.num_hand_dofs),
            device=self.device,
        )
        default_dof_pos[:, self.num_arm_dofs:] += hand_noise

        # Clamp to limits
        default_dof_pos = torch.clamp(
            default_dof_pos,
            self.dof_lower_limits.unsqueeze(0),
            self.dof_upper_limits.unsqueeze(0),
        )

        dof_vel = torch.zeros_like(default_dof_pos)

        # Write robot state
        # Need to write full joint state (including any non-arm-hand joints)
        full_dof_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_dof_pos[:, :self.num_hand_arm_dofs] = default_dof_pos
        full_dof_vel = self.robot.data.default_joint_vel[env_ids].clone()
        full_dof_vel[:, :self.num_hand_arm_dofs] = dof_vel

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(full_dof_pos, full_dof_vel, env_ids=env_ids)

        # Set position targets
        self.prev_targets[env_ids, :] = default_dof_pos
        self.cur_targets[env_ids, :] = default_dof_pos
        self.robot.set_joint_position_target(default_dof_pos, env_ids=env_ids)

        # Reset object pose
        self._reset_object(env_ids)

        # Reset goal
        self._reset_target(env_ids, is_first_goal=True)

        # Reset table
        table_root = self.table.data.default_root_state[env_ids].clone()
        table_root[:, :3] += self.scene.env_origins[env_ids]
        self.table.write_root_pose_to_sim(table_root[:, :7], env_ids)
        self.table.write_root_velocity_to_sim(table_root[:, 7:], env_ids)

    def _reset_object(self, env_ids):
        """Reset object to a randomized pose on the table."""
        num_resets = len(env_ids)

        object_default = self.object.data.default_root_state[env_ids].clone()

        # Add position noise
        pos_noise = torch.zeros(num_resets, 3, device=self.device)
        pos_noise[:, 0] = sample_uniform(
            -self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_x,
            (num_resets,), device=self.device,
        )
        pos_noise[:, 1] = sample_uniform(
            -self.cfg.reset_position_noise_y, self.cfg.reset_position_noise_y,
            (num_resets,), device=self.device,
        )
        pos_noise[:, 2] = sample_uniform(
            -self.cfg.reset_position_noise_z, self.cfg.reset_position_noise_z,
            (num_resets,), device=self.device,
        )

        object_default[:, 0:3] += pos_noise + self.scene.env_origins[env_ids]

        # Randomize rotation if enabled
        if self.cfg.randomize_object_rotation:
            rand_angles = sample_uniform(
                -1.0, 1.0, (num_resets, 2), device=self.device
            )
            new_rot = _randomize_rotation(
                rand_angles[:, 0], rand_angles[:, 1],
                self.x_unit[env_ids], self.y_unit[env_ids],
            )
            object_default[:, 3:7] = new_rot

        # Zero velocities
        object_default[:, 7:] = 0.0

        self.object.write_root_pose_to_sim(object_default[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default[:, 7:], env_ids)

        # Store initial state for fall detection
        self.object_init_state[env_ids, :3] = object_default[:, :3] - self.scene.env_origins[env_ids]
        self.object_init_state[env_ids, 3:7] = object_default[:, 3:7]

        # Sample object scale noise multiplier
        scale_range = self.cfg.object_scale_noise_multiplier_range
        self.object_scale_noise_multiplier[env_ids] = sample_uniform(
            scale_range[0], scale_range[1],
            (len(env_ids), 3), device=self.device,
        )

    def _reset_target(self, env_ids, is_first_goal: bool = True):
        """Sample a new goal pose for the given environments."""
        num_resets = len(env_ids)
        cfg = self.cfg

        if is_first_goal or cfg.goal_sampling_type == "absolute":
            # Sample absolute goal in target volume
            rand_pos = sample_uniform(0.0, 1.0, (num_resets, 3), device=self.device)
            target_pos = self.target_volume_min + rand_pos * (
                self.target_volume_max - self.target_volume_min
            )
            self.goal_states[env_ids, 0:3] = target_pos

            # Random orientation
            rand_angles = sample_uniform(-1.0, 1.0, (num_resets, 2), device=self.device)
            new_rot = _randomize_rotation(
                rand_angles[:, 0], rand_angles[:, 1],
                self.x_unit[env_ids], self.y_unit[env_ids],
            )
            self.goal_states[env_ids, 3:7] = new_rot
        elif cfg.goal_sampling_type == "delta":
            # Delta from current goal
            delta_pos = sample_uniform(
                -cfg.delta_goal_distance, cfg.delta_goal_distance,
                (num_resets, 3), device=self.device,
            )
            new_pos = self.goal_states[env_ids, 0:3] + delta_pos
            new_pos = torch.clamp(new_pos, self.target_volume_min, self.target_volume_max)
            self.goal_states[env_ids, 0:3] = new_pos

            # Delta rotation
            delta_rad = math.radians(cfg.delta_rotation_degrees)
            rand_axis = torch.randn(num_resets, 3, device=self.device)
            rand_axis = rand_axis / (rand_axis.norm(dim=-1, keepdim=True) + 1e-8)
            rand_angle = sample_uniform(-delta_rad, delta_rad, (num_resets,), device=self.device)
            delta_quat = quat_from_angle_axis(rand_angle, rand_axis)
            self.goal_states[env_ids, 3:7] = quat_mul(
                delta_quat, self.goal_states[env_ids, 3:7]
            )

        # Zero velocities in goal state
        self.goal_states[env_ids, 7:13] = 0.0

        # Reset near-goal tracking for these envs
        self.near_goal_steps[env_ids] = 0


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------

@torch.jit.script
def _randomize_rotation(
    rand0: torch.Tensor,
    rand1: torch.Tensor,
    x_unit: torch.Tensor,
    y_unit: torch.Tensor,
) -> torch.Tensor:
    """Generate random quaternions from two uniform random values in [-1, 1]."""
    return quat_mul(
        quat_from_angle_axis(rand0 * math.pi, x_unit),
        quat_from_angle_axis(rand1 * math.pi, y_unit),
    )
