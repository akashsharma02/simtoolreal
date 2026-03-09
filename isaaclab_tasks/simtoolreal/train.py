#!/usr/bin/env python3
# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Training script for SimToolReal using Isaac Lab + rl_games.

Usage:
    # From repo root, with Isaac Lab environment activated:
    python -m isaaclab_tasks.simtoolreal.train --num_envs 4096

    # With rl_games via Isaac Lab's wrapper:
    ./isaaclab.sh -p isaaclab_tasks/simtoolreal/train.py --num_envs 4096
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import tyro


@dataclass
class TrainArgs:
    """Training configuration for SimToolReal on Isaac Lab."""

    num_envs: int = 8192
    """Number of parallel environments."""

    seed: int = 42
    """Random seed."""

    checkpoint: Optional[str] = None
    """Path to checkpoint for resuming training."""

    max_iterations: int = 50000
    """Maximum number of training iterations."""

    # RL hyperparameters (matching original SAPG config)
    learning_rate: float = 1e-4
    mini_epochs: int = 2
    minibatch_size: int = 98304
    horizon_length: int = 16

    # WandB
    wandb_activate: bool = True
    wandb_entity: str = "tylerlum"
    wandb_project: str = "simtoolreal-isaaclab"
    wandb_group: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    experiment_name: str = "simtoolreal"


def main():
    args = tyro.cli(TrainArgs)

    # Defer Isaac Lab imports to after argument parsing (Isaac Sim startup is slow)
    from isaaclab.app import AppLauncher

    # Create a minimal app launcher config
    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app

    # Now import Isaac Lab modules
    from isaaclab.envs import DirectRLEnvCfg

    from isaaclab_tasks.simtoolreal.env import SimToolRealEnv
    from isaaclab_tasks.simtoolreal.env_cfg import SimToolRealEnvCfg

    print("=" * 60)
    print("SimToolReal Isaac Lab Training")
    print(f"  Environments: {args.num_envs}")
    print(f"  Seed: {args.seed}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # TODO: Wire up rl_games training loop with the local SAPG fork
    # For now, this script validates that the environment can be instantiated.

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"

    print(f"Observation space: {cfg.observation_space}")
    print(f"State space: {cfg.state_space}")
    print(f"Action space: {cfg.action_space}")
    print()
    print("NOTE: Full rl_games training integration is not yet wired up.")
    print("This script currently validates environment instantiation.")
    print("The rl_games SAPG training loop needs to be connected next (Phase 2).")

    simulation_app.close()


if __name__ == "__main__":
    main()
