# Isaac Gym → Isaac Lab Migration Guide

## Overview

This document tracks the migration of SimToolReal from Isaac Gym Preview 4 to Isaac Lab v2.3.2.

**Key API differences:**
- Isaac Gym: procedural API (`gym.create_sim()`, `gym.load_asset()`, `gymtorch.wrap_tensor()`)
- Isaac Lab: declarative configs + `DirectRLEnv` base class with `Articulation` / `RigidObject` wrappers

**Python version change:** 3.8 → 3.11

## File Mapping

| Isaac Gym (original) | Isaac Lab (new) | Status |
|---|---|---|
| `isaacgymenvs/tasks/simtoolreal/env.py` | `isaaclab_tasks/simtoolreal/env.py` | Phase 1 complete |
| `isaacgymenvs/tasks/base/vec_task.py` | Isaac Lab `DirectRLEnv` (built-in) | Replaced |
| `isaacgymenvs/cfg/task/SimToolReal.yaml` | `isaaclab_tasks/simtoolreal/env_cfg.py` | Phase 1 complete |
| `isaacgymenvs/cfg/train/*.yaml` | TBD (Phase 2) | Not started |
| `isaacgymenvs/train.py` | `isaaclab_tasks/simtoolreal/train.py` | Skeleton only |
| `isaacgymenvs/utils/rlgames_utils.py` | Isaac Lab built-in rl_games wrapper | Phase 2 |
| `isaacgymenvs/utils/dr_utils.py` | Isaac Lab `EventCfg` / `mdp` randomization | Phase 3 |
| `isaacgymenvs/utils/observation_action_utils_sharpa.py` | Inlined into `env.py` | Phase 1 complete |
| `deployment/isaac/isaac_env*.py` | TBD (Phase 4) | Not started |

## Phase 1: Environment Port (Current)

### What's ported
- **Scene setup**: Robot (IIWA + Sharpa), table, object, ground plane
- **State access**: Joint positions/velocities, rigid body poses via `Articulation.data` / `RigidObject.data`
- **Action application**: Position targets with EMA smoothing (arm relative, hand absolute)
- **Observations**: Full obs dict with policy/critic split for asymmetric actor-critic
- **Rewards**: Lifting, keypoint distance, fingertip delta, action penalties, success bonus
- **Resets**: Randomized robot/object poses, goal sampling (absolute + delta)
- **Random perturbations**: Forces and torques on object

### What's NOT yet ported (deferred to later phases)
- **Observation/action delay queues** — the original adds latency for sim-to-real transfer
- **Procedural object generation** — `handle_head_primitives` generates random tool URDFs
- **Camera sensor / video capture** — used for WandB logging
- **Good-reset-boundary** — state buffer for curriculum learning
- **Tyler curriculum** — observation dropout curriculum
- **Keyboard interactive controls** — viewer keyboard callbacks
- **Blue robot visualization** — PD target visualization
- **Table force sensor** — optional force sensor on table
- **Object state delay/noise** — delayed noisy object observations

### Key API Translations

| Isaac Gym | Isaac Lab |
|---|---|
| `gym.create_sim()` | `SimulationCfg` in env config |
| `gym.load_asset(sim, dir, file, opts)` | `UrdfFileCfg(asset_path=...)` in `ArticulationCfg.spawn` |
| `gym.create_env()` / `create_actor()` | `scene.clone_environments()` + prim_path patterns |
| `gym.acquire_*_tensor()` + `gymtorch.wrap_tensor()` | `articulation.data.joint_pos`, `.body_pos_w`, etc. |
| `gym.refresh_*_tensor()` | Automatic (Isaac Lab refreshes each step) |
| `gym.set_dof_position_target_tensor()` | `articulation.set_joint_position_target()` |
| `gym.set_actor_root_state_tensor_indexed()` | `object.write_root_pose_to_sim()` + `write_root_velocity_to_sim()` |
| `gym.apply_rigid_body_force_tensors()` | `object.set_external_force_and_torque()` |
| `gym.set_dof_state_tensor_indexed()` | `articulation.write_joint_state_to_sim()` |
| `gymapi.AssetOptions()` | `RigidBodyPropertiesCfg`, `ArticulationRootPropertiesCfg` |
| `gymapi.SimParams` | `SimulationCfg` + `PhysxCfg` |
| `gymapi.Transform` / `Vec3` / `Quat` | Plain torch tensors |
| Domain randomization `dr_utils.py` | `EventCfg` with `EventTerm` + `mdp.*` functions |

## Phase 2: rl_games Training Integration

Connect the Isaac Lab environment to the local rl_games fork (with SAPG):
- Use Isaac Lab's built-in rl_games wrapper or adapt `rlgames_utils.py`
- Preserve `ComplexObsRLGPUEnv` for asymmetric actor-critic
- Port YAML training configs to work with Isaac Lab's runner

## Phase 3: Domain Randomization

Port the randomization from `dr_utils.py` to Isaac Lab's `EventCfg`:
- Robot DOF properties (stiffness, damping, effort, friction, armature)
- Robot rigid body mass
- Object mass and friction
- Gravity perturbation
- Observation and action noise

## Phase 4: Deployment & Evaluation

- Update `deployment/isaac/` for sim2sim with Isaac Lab
- Update `dextoolbench/eval.py` and `eval_interactive.py`

## Installation (Isaac Lab)

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

# Install this project
pip install -e .
```
