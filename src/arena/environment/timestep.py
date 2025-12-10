from dataclasses import dataclass

import numpy as np

from arena.types import Actions, Observations, Rewards, Terminations


@dataclass
class Metadata:
    cumulative_returns: np.ndarray


@dataclass
class Timestep:
    step: int
    action: Actions
    reward: Rewards
    observation: Observations
    terminations: Terminations
    metadata: Metadata
