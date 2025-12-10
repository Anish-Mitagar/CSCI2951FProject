from dataclasses import dataclass

import numpy as np


@dataclass
class Settings:
    height: int
    width: int
    max_steps: int
    survival_decay: float
    vision_radius: int


@dataclass
class Trajectory:
    episode: int
    representations: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[np.ndarray]
    cumulative_returns: list[np.ndarray]
    alive_status: list[np.ndarray]
    positions: list[np.ndarray]
    strengths: list[np.ndarray]
    identifiers: list[np.ndarray]
    agent_names: list[str]
    config: Settings
