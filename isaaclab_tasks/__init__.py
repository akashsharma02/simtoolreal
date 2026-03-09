# Copyright (c) 2024, SimToolReal Project
# SPDX-License-Identifier: MIT

"""Isaac Lab tasks for SimToolReal.

Importing this module registers all SimToolReal gymnasium environments:
  - Isaac-SimToolReal-Direct-v0 (MLP asymmetric PPO)
  - Isaac-SimToolReal-LSTM-Direct-v0 (LSTM asymmetric PPO, paper default)
"""

from . import simtoolreal  # noqa: F401 — triggers gym.register() calls
