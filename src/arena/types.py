from typing import TypeAlias

import numpy as np

Position: TypeAlias = np.ndarray  # Shape: (2,), dtype: int32, [row, col]

Action: TypeAlias = int  # 0=up, 1=right, 2=down, 3=left
Actions: TypeAlias = np.ndarray  # Shape: (num_players,), dtype: int32

Observation: TypeAlias = np.ndarray  # Shape: (height, width, channels), dtype: int32
Observations: TypeAlias = np.ndarray  # Shape: (num_players, height, width, channels)

Reward: TypeAlias = float
Rewards: TypeAlias = np.ndarray  # Shape: (num_players,), dtype: float32

Terminations: TypeAlias = np.ndarray  # Shape: (num_players,), dtype: bool

Grid: TypeAlias = np.ndarray  # Shape: (height, width), dtype: int32
