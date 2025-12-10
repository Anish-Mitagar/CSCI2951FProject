import numpy as np

from arena.constants import Direction
from arena.types import Action, Observation, Reward

from .policy import Policy


class Zombie(Policy):
    ACTION_SEQUENCE = [
        Direction.RIGHT.value,
        Direction.DOWN.value,
        Direction.LEFT.value,
        Direction.UP.value,
    ]

    def __init__(self, identifier: int):
        super().__init__(identifier)
        self.counter = 0

    def get_action(self, observation: Observation) -> Action:
        action = self.ACTION_SEQUENCE[self.counter % len(self.ACTION_SEQUENCE)]
        self.counter += 1
        return int(action)

    def update(self, observation: Observation, action: Action, reward: Reward) -> None:
        pass

    def reset(self) -> None:
        self.counter = 0
