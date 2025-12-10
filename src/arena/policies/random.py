import numpy as np

from arena.types import Action, Observation, Reward

from .policy import Policy


class Random(Policy):

    def get_action(self, observation: Observation) -> Action:
        return np.random.randint(0, 4)

    def update(self, observation: Observation, action: Action, reward: Reward) -> None:
        pass

    def reset(self) -> None:
        pass
