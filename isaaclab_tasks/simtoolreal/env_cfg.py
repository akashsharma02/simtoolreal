# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Configuration for the SimToolReal Isaac Lab environment."""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg, UrdfFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

# Path to assets directory (relative to this file)
_ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../assets"))


def _robot_urdf_path() -> str:
    return os.path.join(
        _ASSETS_DIR, "urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )


def _table_urdf_path() -> str:
    return os.path.join(_ASSETS_DIR, "urdf/table_narrow.urdf")


# ---------------------------------------------------------------------------
# Robot ArticulationCfg
# ---------------------------------------------------------------------------

IIWA_SHARPA_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_robot_urdf_path(),
        fix_base=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="position"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # IIWA arm default pose (upright with hand above table)
            "iiwa14_joint_1": -1.571,
            "iiwa14_joint_2": 1.571,
            "iiwa14_joint_3": 0.0,
            "iiwa14_joint_4": 1.376,
            "iiwa14_joint_5": 0.0,
            "iiwa14_joint_6": 1.485,
            "iiwa14_joint_7": 1.308,  # 60 deg offset for sharpa mount
            # All hand joints start at 0
            ".*": 0.0,
        },
    ),
)

# ---------------------------------------------------------------------------
# Table ArticulationCfg (fixed base, no DOFs)
# ---------------------------------------------------------------------------

TABLE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_table_urdf_path(),
        fix_base=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)


# ---------------------------------------------------------------------------
# Domain Randomization EventCfg
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    """Configuration for domain randomization events."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 100,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 100,
        },
    )
    object_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.3]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


# ---------------------------------------------------------------------------
# Main Environment Config
# ---------------------------------------------------------------------------

@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    """Configuration for the SimToolReal DirectRL environment.

    This mirrors the original Isaac Gym config in isaacgymenvs/cfg/task/SimToolReal.yaml
    for the IIWA arm + Sharpa hand dexterous tool manipulation task.
    """

    # ---- env ----
    decimation = 1  # controlFrequencyInv=1, so 60 Hz control at 60 Hz sim
    episode_length_s = 10.0  # episodeLength=600 steps at dt=1/60
    action_space = 29  # 7 arm DOFs + 22 hand DOFs
    observation_space = 140  # will be computed from obs_list; placeholder
    state_space = 0  # will be computed from state_list for asymmetric; placeholder

    # ---- simulation ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        substeps=2,
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            solver_type=1,  # TGS
            max_depenetration_velocity=1000.0,
            gpu_max_rigid_contact_count=8 * 1024 * 1024,
        ),
        gravity=(0.0, 0.0, -9.81),
    )

    # ---- scene ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=1.2,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # ---- assets ----
    robot_cfg: ArticulationCfg = IIWA_SHARPA_CFG
    table_cfg: ArticulationCfg = TABLE_CFG
    # Object cfg will be set dynamically based on object_name (URDF path varies per object)
    # For now, provide a placeholder that users must override
    object_cfg: RigidObjectCfg = MISSING

    # ---- robot structure ----
    num_arm_dofs: int = 7
    num_hand_dofs: int = 22
    num_fingertips: int = 5

    # Fingertip body names in the URDF
    fingertip_body_names: list = [
        "left_index_DP",
        "left_middle_DP",
        "left_ring_DP",
        "left_thumb_DP",
        "left_pinky_DP",
    ]
    # Palm body name for computing palm center position
    palm_body_name: str = "left_palm_link"
    palm_offset: tuple = (-0.0, -0.02, 0.16)

    fingertip_offsets: list = [
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
    ]

    # ---- control ----
    dof_speed_scale: float = 1.5
    hand_moving_average: float = 0.1
    arm_moving_average: float = 0.1

    # ---- object / keypoint ----
    object_name: str = "handle_head_primitives"
    object_base_size: float = 0.04
    keypoint_scale: float = 1.5
    num_keypoints: int = 4
    keypoint_offsets: list = [
        [1, 1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]

    # ---- reward scales ----
    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15
    keypoint_rew_scale: float = 200.0
    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0
    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003
    fall_distance: float = 0.24
    fall_penalty: float = 0.0
    object_lin_vel_penalty_scale: float = 0.0
    object_ang_vel_penalty_scale: float = 0.0

    # ---- success ----
    success_tolerance: float = 0.075
    target_success_tolerance: float = 0.01
    tolerance_curriculum_increment: float = 0.9
    tolerance_curriculum_interval: int = 3000
    max_consecutive_successes: int = 50
    success_steps: int = 10
    force_consecutive_near_goal_steps: bool = False

    # ---- reset noise ----
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    randomize_object_rotation: bool = True
    reset_dof_pos_noise_fingers: float = 0.1
    reset_dof_pos_noise_arm: float = 0.1
    reset_dof_vel_noise: float = 0.5

    # ---- random forces/torques on object ----
    force_scale: float = 20.0
    force_prob_range: tuple = (0.001, 0.1)
    force_decay: float = 0.0
    force_only_when_lifted: bool = True
    torque_scale: float = 2.0
    torque_prob_range: tuple = (0.001, 0.1)
    torque_decay: float = 0.0
    torque_only_when_lifted: bool = True

    # ---- goal sampling ----
    goal_sampling_type: str = "delta"  # "absolute", "delta", or "coin_flip"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple = (0.35, 0.2, 0.95)

    # ---- observation delays / noise ----
    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    object_scale_noise_multiplier_range: tuple = (1.0, 1.0)
    joint_velocity_obs_noise_std: float = 0.1
    clamp_abs_observations: float = 10.0

    # ---- observation lists (for asymmetric actor-critic) ----
    # Critic (privileged) observations
    state_list: list = [
        "joint_pos", "joint_vel", "prev_action_targets",
        "palm_pos", "palm_rot", "palm_vel",
        "object_rot", "object_vel",
        "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal",
        "object_scales",
        "closest_keypoint_max_dist", "closest_fingertip_dist",
        "lifted_object", "progress", "successes", "reward",
    ]
    # Actor (policy) observations
    obs_list: list = [
        "joint_pos", "joint_vel", "prev_action_targets",
        "palm_pos", "palm_rot",
        "object_rot",
        "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal",
        "object_scales",
    ]

    # ---- asset friction ----
    robot_friction: float = 0.5
    fingertip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5

    # ---- fixed-size keypoint reward ----
    fixed_size_keypoint_reward: bool = True
    fixed_size: tuple = (0.141, 0.03025, 0.0271)

    # ---- procedural objects ----
    handle_head_types: list = [
        "hammer", "screwdriver", "marker", "spatula", "eraser", "brush",
    ]

    # ---- domain randomization (disabled by default, matching original) ----
    # Set events to enable randomization
    # events: EventCfg = EventCfg()
