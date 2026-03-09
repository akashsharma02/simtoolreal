#!/usr/bin/env python3
# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Training script for SimToolReal using Isaac Lab + rl_games.

This follows Isaac Lab's standard rl_games training pattern from
scripts/reinforcement_learning/rl_games/train.py, adapted for
SimToolReal with the local rl_games fork (SAPG support).

Usage:
    # MLP asymmetric (default PPO):
    python isaaclab_tasks/simtoolreal/train.py --task Isaac-SimToolReal-Direct-v0 --num_envs 8192

    # LSTM asymmetric (paper default):
    python isaaclab_tasks/simtoolreal/train.py --task Isaac-SimToolReal-LSTM-Direct-v0 --num_envs 8192

    # Resume from checkpoint:
    python isaaclab_tasks/simtoolreal/train.py --task Isaac-SimToolReal-LSTM-Direct-v0 --checkpoint path/to/model.pth

    # With WandB tracking:
    python isaaclab_tasks/simtoolreal/train.py --task Isaac-SimToolReal-LSTM-Direct-v0 --track --wandb-entity myentity
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# -- CLI arguments --
parser = argparse.ArgumentParser(description="Train SimToolReal with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-LSTM-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default="simtoolreal-isaaclab", help="WandB project name.")
parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity (team).")
parser.add_argument("--wandb-name", type=str, default=None, help="WandB run name.")
parser.add_argument("--track", action="store_true", default=False, help="Track with Weights and Biases.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import math
import os
import random
import time
from datetime import datetime

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# Register our task environments
import isaaclab_tasks.simtoolreal  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Train SimToolReal with RL-Games agent."""
    # Override configurations with CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    # Max iterations
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )

    # Checkpoint
    if args_cli.checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args_cli.checkpoint
        print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")

    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # Logging directory
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    print(f"[INFO] Logging experiment in directory: {os.path.join(log_root_path, log_dir)}")

    # Dump configs for reproducibility
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    # Read agent config values
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # Create Isaac Lab environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # Wrap for rl_games
    # obs_groups maps Isaac Lab observation groups to rl_games' expected keys:
    #   "obs" -> policy observations (actor)
    #   "states" -> privileged observations (critic, for asymmetric AC)
    obs_groups = {"obs": ["policy"], "states": ["critic"]}
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups=obs_groups)

    # Register environment with rl_games
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )

    # Set number of actors
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # Create rl_games runner
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    # WandB tracking
    if args_cli.track:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb_project = args_cli.wandb_project_name
        experiment_name = args_cli.wandb_name or log_dir
        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    # Train
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": args_cli.checkpoint})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
