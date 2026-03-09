#!/usr/bin/env python3
# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Inference/play script for SimToolReal using Isaac Lab + rl_games.

Usage:
    python isaaclab_tasks/simtoolreal/play.py --task Isaac-SimToolReal-LSTM-Direct-v0 \
        --checkpoint path/to/model.pth --num_envs 32

    # Record video:
    python isaaclab_tasks/simtoolreal/play.py --task Isaac-SimToolReal-LSTM-Direct-v0 \
        --checkpoint path/to/model.pth --video --video_length 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained SimToolReal RL-Games policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (steps).")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-LSTM-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks.simtoolreal  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Play with a trained RL-Games agent."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Load checkpoint
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")

    # Read agent config values
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "rl_games", "play_videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for rl_games
    obs_groups = {"obs": ["policy"], "states": ["critic"]}
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups=obs_groups)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # Create runner and play
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": False, "play": True, "checkpoint": args_cli.checkpoint})

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
