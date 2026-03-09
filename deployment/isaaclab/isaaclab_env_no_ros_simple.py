#!/usr/bin/env python3
# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Standalone sim2sim evaluation using Isaac Lab (no ROS).

This is the Isaac Lab equivalent of deployment/isaac/isaac_env_no_ros_simple.py.
It creates a single-env Isaac Lab environment, loads a policy checkpoint,
and runs the control loop with visualization.

Usage:
    python deployment/isaaclab/isaaclab_env_no_ros_simple.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth \
        --object-category hammer --object-name claw_hammer --task-name swing_down
"""

from __future__ import annotations

"""Launch Isaac Sim first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SimToolReal sim2sim evaluation (Isaac Lab).")
parser.add_argument("--config-path", type=str, required=True, help="Path to policy config YAML.")
parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to policy checkpoint.")
parser.add_argument("--object-category", type=str, default="hammer", help="Object category.")
parser.add_argument("--object-name", type=str, default="claw_hammer", help="Object name.")
parser.add_argument("--task-name", type=str, default="swing_down", help="Task / trajectory name.")
parser.add_argument("--control-hz", type=float, default=60.0, help="Control loop frequency.")
parser.add_argument("--goal-z-offset", type=float, default=0.1, help="Z offset for goal poses.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after sim launch."""

import json
import time
from pathlib import Path

import torch
from termcolor import colored

from deployment.isaaclab.isaaclab_env import create_eval_env
from deployment.rl_player import RlPlayer

N_OBS = 140
N_ACT = 29


def warn(msg: str):
    print(colored(msg, "yellow"))


def main():
    config_path = Path(args_cli.config_path)
    checkpoint_path = Path(args_cli.checkpoint_path)
    assert config_path.exists(), f"Config not found: {config_path}"
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    control_dt = 1.0 / args_cli.control_hz

    # Load trajectory
    repo_root = Path(__file__).resolve().parents[2]
    trajectory_path = (
        repo_root / "dextoolbench" / "trajectories"
        / args_cli.object_category / args_cli.object_name
        / f"{args_cli.task_name}.json"
    )
    assert trajectory_path.exists(), f"Trajectory not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    # Raise goal z to avoid table collision
    traj_data["goals"] = [
        [x, y, z + args_cli.goal_z_offset, qx, qy, qz, qw]
        for (x, y, z, qx, qy, qz, qw) in traj_data["goals"]
    ]

    DEVICE = "cuda:0"

    # Create eval environment
    env = create_eval_env(
        object_name=args_cli.object_name,
        trajectory_goals=traj_data["goals"],
        start_pose=traj_data.get("start_pose"),
        device=DEVICE,
        headless=args_cli.headless,
    )

    # Create policy player
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=DEVICE,
        num_envs=1,
    )

    # Reset and run
    obs, _ = env.reset()
    observation = obs["policy"]

    print(f"Running sim2sim evaluation at {args_cli.control_hz} Hz")
    print(f"Object: {args_cli.object_category}/{args_cli.object_name}")
    print(f"Task: {args_cli.task_name}")
    print(f"Trajectory goals: {len(traj_data['goals'])}")

    step = 0
    while simulation_app.is_running():
        start_time = time.time()

        action = policy.get_normalized_action(observation, deterministic_actions=True)
        obs, reward, terminated, truncated, info = env.step(action)
        observation = obs["policy"]

        step += 1

        # Print progress
        if step % 60 == 0:
            successes = env.successes[0].item() if hasattr(env, 'successes') else 0
            print(
                f"Step {step:5d} | "
                f"Reward: {reward[0].item():8.3f} | "
                f"Successes: {successes:.0f}"
            )

        elapsed = time.time() - start_time
        sleep_time = control_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif step % 100 == 0:
            warn(
                f"Control loop too slow! "
                f"Desired: {args_cli.control_hz:.0f} Hz, "
                f"Actual: {1.0 / elapsed:.0f} Hz"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
