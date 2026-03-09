# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""SimToolReal Isaac Lab environment for dexterous tool manipulation."""

import gymnasium as gym

from . import agents

##
# Register Gymnasium environments.
##

# MLP asymmetric actor-critic (default PPO)
gym.register(
    id="Isaac-SimToolReal-Direct-v0",
    entry_point="isaaclab_tasks.simtoolreal.env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.simtoolreal.env_cfg:SimToolRealEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# LSTM asymmetric actor-critic (paper default)
gym.register(
    id="Isaac-SimToolReal-LSTM-Direct-v0",
    entry_point="isaaclab_tasks.simtoolreal.env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.simtoolreal.env_cfg:SimToolRealEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lstm_cfg.yaml",
    },
)
