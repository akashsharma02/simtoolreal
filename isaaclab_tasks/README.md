# SimToolReal — Isaac Lab

This directory contains the Isaac Lab (v2.3.2) port of the SimToolReal training environment, migrated from Isaac Gym Preview 4. It provides the same simulation environment, RL training pipeline, domain randomization, and deployment scripts using Isaac Lab's `DirectRLEnv` API.

For the full project overview, DexToolBench benchmark, and real-world deployment, see the [main README](../README.md).

## Requirements

- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- PyTorch 2.7+
- CUDA 12.8+

## Installation

```bash
# Create Python 3.11 environment
python3.11 -m venv env_isaaclab
source env_isaaclab/bin/activate

# Install Isaac Sim
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install PyTorch
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Clone and install Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
cd -

# Install this project (from repo root)
pip install -e .

# Install the local rl_games fork (SAPG support)
cd rl_games && pip install -e . && cd -
```

> **Note:** This is a separate environment from the Isaac Gym Python 3.8 setup. The original `isaacgymenvs/` code continues to work in its own environment.

## Project Structure

```
isaaclab_tasks/
  └── simtoolreal/
      ├── __init__.py          # Gymnasium env registration
      ├── env.py               # DirectRLEnv environment (core simulation)
      ├── env_cfg.py           # @configclass config (replaces Hydra YAML)
      ├── train.py             # Training entry point
      ├── play.py              # Inference / playback
      └── agents/
          ├── rl_games_ppo_cfg.yaml        # MLP asymmetric PPO config
          └── rl_games_ppo_lstm_cfg.yaml   # LSTM asymmetric PPO config (paper default)

deployment/
  └── isaaclab/
      ├── isaaclab_env.py                  # Environment factory (create_env / create_eval_env)
      ├── isaaclab_env_no_ros_simple.py    # Standalone sim2sim evaluation
      ├── isaaclab_env_node.py             # ROS1 sim2sim node
      └── eval_runner.py                   # DexToolBench numerical evaluation
```

## Registered Environments

| Task ID | Architecture | Description |
|---------|-------------|-------------|
| `Isaac-SimToolReal-Direct-v0` | MLP asymmetric PPO | Feedforward actor-critic |
| `Isaac-SimToolReal-LSTM-Direct-v0` | LSTM asymmetric PPO | Recurrent actor-critic (paper default) |

Both use asymmetric actor-critic: the actor receives partial observations (joint positions, object pose, goal pose, keypoints) while the critic additionally receives privileged information (velocities, contact forces, episode progress).

## Training

All commands are run from the repository root.

```bash
# LSTM asymmetric (paper default):
python isaaclab_tasks/simtoolreal/train.py \
    --task Isaac-SimToolReal-LSTM-Direct-v0 \
    --num_envs 8192 \
    --headless

# MLP asymmetric:
python isaaclab_tasks/simtoolreal/train.py \
    --task Isaac-SimToolReal-Direct-v0 \
    --num_envs 8192 \
    --headless

# Resume from checkpoint:
python isaaclab_tasks/simtoolreal/train.py \
    --task Isaac-SimToolReal-LSTM-Direct-v0 \
    --checkpoint path/to/model.pth

# Inference / playback:
python isaaclab_tasks/simtoolreal/play.py \
    --task Isaac-SimToolReal-LSTM-Direct-v0 \
    --checkpoint path/to/model.pth \
    --num_envs 32
```

### SAPG

The agent YAML configs include SAPG (Sample-Adaptive Policy Gradient) parameters that work with the local `rl_games` fork. To enable SAPG, set `expl_type: 'mixed_expl'` or `'mixed_expl_disjoint'` in the agent config. `num_envs` must be divisible by `num_blocks` (default 6).

## Domain Randomization

Domain randomization is configured via `EventCfg` in `env_cfg.py` and is disabled by default (matching the original `randomize: False`). The following randomizations are ported:

| Parameter | Isaac Lab EventTerm | Mode |
|-----------|-------------------|------|
| Robot DOF stiffness/damping | `mdp.randomize_actuator_gains` | reset, every 720 steps |
| Robot DOF friction/armature | `mdp.randomize_joint_parameters` | reset, every 720 steps |
| Robot rigid body material | `mdp.randomize_rigid_body_material` | reset, every 720 steps |
| Robot rigid body mass | `mdp.randomize_rigid_body_mass` | startup |
| Object rigid body mass | `mdp.randomize_rigid_body_mass` | startup |
| Object physics material | `mdp.randomize_rigid_body_material` | reset, every 720 steps |
| Gravity perturbation | `mdp.randomize_physics_scene_gravity` | interval, 12s |
| Observation noise | Gaussian, applied per-step in `_get_observations` | constant schedule |
| Action noise | Gaussian, applied per-step in `_pre_physics_step` | linear schedule |

To enable randomization, uncomment `events = EventCfg()` in `env_cfg.py` or set it programmatically:

```python
from isaaclab_tasks.simtoolreal.env_cfg import SimToolRealEnvCfg, EventCfg

cfg = SimToolRealEnvCfg()
cfg.events = EventCfg()
```

## Deployment and Evaluation

### Standalone Sim2Sim (no ROS)

Runs a policy in Isaac Lab with visualization:

```bash
python deployment/isaaclab/isaaclab_env_no_ros_simple.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --object-category hammer --object-name claw_hammer --task-name swing_down
```

### DexToolBench Numerical Evaluation

Runs multiple episodes and reports success rates:

```bash
python deployment/isaaclab/eval_runner.py \
    --config-path pretrained_policy/config.yaml \
    --checkpoint-path pretrained_policy/model.pth \
    --object-category hammer --object-name claw_hammer --task-name swing_down \
    --num-episodes 10
```

### ROS1 Sim2Sim Node

Drop-in replacement for `deployment/isaac/isaac_env_node.py`. Subscribes to joint commands and publishes joint states + object poses:

```bash
python deployment/isaaclab/isaaclab_env_node.py --object-name claw_hammer
```

This integrates with the existing ROS1 deployment nodes (`rl_policy_node.py`, `goal_pose_node.py`, `visualization_node.py`) without any changes to those files.

## Key Differences from Isaac Gym Version

| | Isaac Gym (`isaacgymenvs/`) | Isaac Lab (`isaaclab_tasks/`) |
|---|---|---|
| Python | 3.8 | 3.11 |
| Simulator | Isaac Gym Preview 4 | Isaac Sim 5.1 + Isaac Lab v2.3.2 |
| Base class | `VecTask` | `DirectRLEnv` |
| Configuration | Hydra YAML | `@configclass` dataclasses |
| Tensor access | `gym.acquire_*_tensor()` + `gymtorch.wrap_tensor()` | `articulation.data.joint_pos`, `.body_pos_w`, etc. |
| State reset | `gym.set_actor_root_state_tensor_indexed()` | `object.write_root_pose_to_sim()` |
| Domain randomization | `dr_utils.py` | `EventCfg` / `EventTerm` + `mdp.*` functions |
| Env registration | Hydra task configs | `gymnasium.register()` |

## What's Not Yet Ported

The following features from the Isaac Gym version are not yet migrated:

- Observation/action delay queues (sim-to-real transfer)
- Procedural object generation (`handle_head_primitives`)
- Camera sensor / video capture (WandB logging)
- Good-reset-boundary (state buffer for curriculum learning)
- Tyler curriculum (observation dropout)
- Keyboard interactive controls
- Blue robot PD target visualization
- Table force sensor
- Object state delay/noise

See [docs/isaaclab_migration.md](../docs/isaaclab_migration.md) for the full migration plan and API mapping reference.
