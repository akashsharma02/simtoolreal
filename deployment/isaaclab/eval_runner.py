#!/usr/bin/env python3
# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""DexToolBench evaluation runner using Isaac Lab.

This provides the same evaluation functionality as dextoolbench/eval.py
but uses Isaac Lab instead of Isaac Gym.

Usage:
    python deployment/isaaclab/eval_runner.py \
        --config-path pretrained_policy/config.yaml \
        --checkpoint-path pretrained_policy/model.pth \
        --object-category hammer --object-name claw_hammer --task-name swing_down
"""

from __future__ import annotations

"""Launch Isaac Sim first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DexToolBench evaluation with Isaac Lab.")
parser.add_argument("--config-path", type=str, required=True, help="Policy config YAML.")
parser.add_argument("--checkpoint-path", type=str, required=True, help="Policy checkpoint.")
parser.add_argument("--object-category", type=str, required=True, help="Object category.")
parser.add_argument("--object-name", type=str, required=True, help="Object name.")
parser.add_argument("--task-name", type=str, required=True, help="Task / trajectory name.")
parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode.")
parser.add_argument("--goal-z-offset", type=float, default=0.1, help="Z offset for goal poses.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import json
from pathlib import Path

import torch

from deployment.isaaclab.isaaclab_env import create_eval_env
from deployment.rl_player import RlPlayer

N_OBS = 140
N_ACT = 29


def run_evaluation():
    config_path = Path(args_cli.config_path)
    checkpoint_path = Path(args_cli.checkpoint_path)

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

    traj_data["goals"] = [
        [x, y, z + args_cli.goal_z_offset, qx, qy, qz, qw]
        for (x, y, z, qx, qy, qz, qw) in traj_data["goals"]
    ]

    DEVICE = "cuda:0"

    env = create_eval_env(
        object_name=args_cli.object_name,
        trajectory_goals=traj_data["goals"],
        start_pose=traj_data.get("start_pose"),
        device=DEVICE,
        headless=args_cli.headless,
    )

    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=DEVICE,
        num_envs=1,
    )

    # Run episodes
    results = []
    for ep in range(args_cli.num_episodes):
        obs, _ = env.reset()
        observation = obs["policy"]
        episode_reward = 0.0
        episode_successes = 0

        for step in range(args_cli.max_steps):
            action = policy.get_normalized_action(observation, deterministic_actions=True)
            obs, reward, terminated, truncated, info = env.step(action)
            observation = obs["policy"]
            episode_reward += reward[0].item()

            if terminated[0] or truncated[0]:
                break

        episode_successes = int(env.successes[0].item()) if hasattr(env, 'successes') else 0
        max_successes = env.cfg.max_consecutive_successes

        results.append({
            "episode": ep,
            "reward": episode_reward,
            "successes": episode_successes,
            "max_successes": max_successes,
            "success_rate": episode_successes / max(max_successes, 1),
            "steps": step + 1,
        })

        print(
            f"Episode {ep + 1}/{args_cli.num_episodes}: "
            f"reward={episode_reward:.1f}, "
            f"successes={episode_successes}/{max_successes}, "
            f"steps={step + 1}"
        )

    # Summary
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_success_rate = sum(r["success_rate"] for r in results) / len(results)
    print()
    print("=" * 60)
    print(f"Evaluation: {args_cli.object_category}/{args_cli.object_name} - {args_cli.task_name}")
    print(f"  Episodes: {args_cli.num_episodes}")
    print(f"  Avg Reward: {avg_reward:.1f}")
    print(f"  Avg Success Rate: {avg_success_rate:.2%}")
    print("=" * 60)

    env.close()
    return results


if __name__ == "__main__":
    run_evaluation()
    simulation_app.close()
