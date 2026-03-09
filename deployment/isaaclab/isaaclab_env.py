#!/usr/bin/env python3
# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Isaac Lab environment factory for deployment and evaluation.

This replaces deployment/isaac/isaac_env.py, providing environment
creation using Isaac Lab's DirectRLEnv instead of Isaac Gym's VecTask.

The key difference is that Isaac Lab uses @configclass dataclasses
instead of Hydra YAML, so config overrides are applied as attribute
assignments on SimToolRealEnvCfg.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from isaaclab_tasks.simtoolreal.env import SimToolRealEnv
from isaaclab_tasks.simtoolreal.env_cfg import SimToolRealEnvCfg


def create_env(
    env_cfg: SimToolRealEnvCfg | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = False,
    overrides: dict[str, Any] | None = None,
) -> SimToolRealEnv:
    """Create an Isaac Lab SimToolReal environment for deployment/eval.

    Args:
        env_cfg: Environment config. If None, uses default SimToolRealEnvCfg.
        num_envs: Number of parallel environments (typically 1 for eval).
        device: Torch device string.
        headless: Whether to run without rendering.
        overrides: Dict of attribute overrides to apply to env_cfg.
            Keys are attribute names on SimToolRealEnvCfg (e.g. "reset_position_noise_x").

    Returns:
        Instantiated SimToolRealEnv.
    """
    if env_cfg is None:
        env_cfg = SimToolRealEnvCfg()

    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if not hasattr(env_cfg, key):
                raise ValueError(
                    f"SimToolRealEnvCfg has no attribute '{key}'. "
                    f"Available attributes: {[a for a in dir(env_cfg) if not a.startswith('_')]}"
                )
            print(f"  Override: {key} = {getattr(env_cfg, key)} -> {value}")
            setattr(env_cfg, key, value)

    env = SimToolRealEnv(cfg=env_cfg, render_mode=None if headless else "human")
    return env


def create_eval_env(
    object_name: str = "claw_hammer",
    trajectory_goals: list[list[float]] | None = None,
    start_pose: list[float] | None = None,
    device: str = "cuda:0",
    headless: bool = False,
) -> SimToolRealEnv:
    """Create an environment configured for evaluation (no noise, no randomization).

    This mirrors the overrides applied in the original
    deployment/isaac/isaac_env_no_ros_simple.py.

    Args:
        object_name: Name of the DexToolBench object.
        trajectory_goals: List of 7-element goal poses [x,y,z,qx,qy,qz,qw].
        start_pose: Initial object pose [x,y,z,qx,qy,qz,qw].
        device: Torch device.
        headless: Run without rendering.

    Returns:
        SimToolRealEnv configured for deterministic evaluation.
    """
    overrides = {
        # Zero all randomization noise
        "reset_position_noise_x": 0.0,
        "reset_position_noise_y": 0.0,
        "reset_position_noise_z": 0.0,
        "randomize_object_rotation": False,
        "reset_dof_pos_noise_fingers": 0.0,
        "reset_dof_pos_noise_arm": 0.0,
        "reset_dof_vel_noise": 0.0,
        # Object
        "object_name": object_name,
        # Disable obs/action noise
        "obs_noise_std": 0.0,
        "action_noise_std": 0.0,
        # Disable random forces
        "force_scale": 0.0,
        "torque_scale": 0.0,
        # Success criteria (stricter for eval)
        "success_steps": 1,
        "fixed_size_keypoint_reward": True,
    }

    return create_env(
        num_envs=1,
        device=device,
        headless=headless,
        overrides=overrides,
    )
