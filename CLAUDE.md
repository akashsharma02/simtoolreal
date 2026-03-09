# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimToolReal is a framework for zero-shot dexterous tool manipulation using an object-centric RL policy. It trains policies in Isaac Gym simulation and deploys them to real robots (IIWA arm + Sharpa hand) via ROS1. The project includes the DexToolBench benchmark with 6 tool categories (hammer, marker, eraser, brush, spatula, screwdriver), each with 2 object instances and 2 tasks.

## Environment Setup

Requires Python 3.8 (Isaac Gym constraint) and uv for package management:

```bash
uv venv --python 3.8
source .venv/bin/activate
uv pip install -e .
# Isaac Gym must be installed separately (Preview 4)
# The local rl_games fork must also be installed: cd rl_games && uv pip install -e .
```

Real-world deployment additionally requires ROS1 Noetic (global or RoboStack).

## Key Commands

All commands run from repo root.

**Training:**
```bash
python isaacgymenvs/launch_training.py --custom_experiment_name my_experiment
# With finetuning: --checkpoint pretrained_policy/model.pth
# Reduce GPU memory: --num_envs 12288 (must be divisible by num_blocks=6)
```

**Interactive evaluation:**
```bash
python dextoolbench/eval_interactive.py --config-path pretrained_policy/config.yaml --checkpoint-path pretrained_policy/model.pth
```

**Formatting (ruff check + format):**
```bash
./format_pys.sh                    # Format baselines, deployment, dextoolbench
./format_py.sh <path>              # Format a specific directory
./format_urdfs.sh                  # Format URDF files with xmllint
```

Note: `isaacgymenvs/` and `rl_games/` are intentionally excluded from formatting (forked upstream code).

## Architecture

### Training Pipeline
- `isaacgymenvs/train.py` — Hydra-based entry point; `launch_training.py` is a tyro wrapper that builds the hydra command
- `isaacgymenvs/tasks/simtoolreal/env.py` — Main Isaac Gym environment (the core simulation env)
- `isaacgymenvs/cfg/` — Hydra configs: `task/` (env variants: MLP/LSTM, symmetric/asymmetric), `train/` (PPO configs)
- `rl_games/` — Local fork of rl_games with PPO and SAPG algorithms; installed as a separate package
- Training uses SAPG (Sample-Adaptive Policy Gradient) with LSTM asymmetric architecture by default (`SimToolRealLSTMAsymmetric`)
- Training output goes to `train_dir/` and logs to WandB

### Deployment (ROS1 Node Architecture)
Deployment uses a multi-node ROS architecture communicating via topics:
- `deployment/rl_policy_node.py` — Subscribes to joint states and object poses, publishes joint commands
- `deployment/goal_pose_node.py` — Manages goal pose sequences, advances when object nears current goal
- `deployment/visualization_node.py` — Read-only Viser 3D web visualizer subscribing to all topics
- `deployment/rl_player.py` / `rl_player_utils.py` — Policy inference utilities used by the policy node
- `deployment/isaac/` — Sim2Sim: Isaac Gym simulation node replacing real hardware + perception
- `deployment/fake/` — No-physics testing: fake robot (interpolates joints) + fake perception (fixed pose)

### DexToolBench
- `dextoolbench/objects.py` — Object model definitions and metadata
- `dextoolbench/eval.py` / `run_all_evals.py` — Numerical policy evaluation
- `dextoolbench/eval_interactive.py` — Web-based interactive evaluation (Viser on port 8080)
- Task trajectories stored as JSON pose sequences in `dextoolbench/trajectories/<category>/<object>/<task>.json`
- Object URDFs in `assets/urdf/dextoolbench/<category>/<object>/`

### Important Import Order
Isaac Gym requires `isaacgym` to be imported **before** `torch`. Files in `isaacgymenvs/` use `# isort: off` to enforce this.

### Isaac Lab Migration (in progress, branch `akash.dev`)
- `isaaclab_tasks/simtoolreal/env.py` — Isaac Lab `DirectRLEnv` port of the Isaac Gym environment
- `isaaclab_tasks/simtoolreal/env_cfg.py` — Dataclass-based config replacing Hydra YAML
- `isaaclab_tasks/simtoolreal/train.py` — Training entry point (skeleton, needs rl_games wiring)
- See `docs/isaaclab_migration.md` for the full migration plan and API mapping
- Isaac Lab requires Python 3.11 and Isaac Sim 5.1 (separate venv from the Isaac Gym Python 3.8 env)

## Key Conventions
- Configuration uses Hydra (isaacgymenvs) and tyro (CLI tools in deployment/dextoolbench)
- `num_envs` must always be divisible by `num_blocks` (default 6) for SAPG training
- External dependency: [FoundationPose fork](https://github.com/kushal2000/FoundationPose) for perception (SAM + pose tracking), installed in a separate environment
- Isaac Lab env uses `@configclass` dataclasses instead of Hydra YAML for configuration
