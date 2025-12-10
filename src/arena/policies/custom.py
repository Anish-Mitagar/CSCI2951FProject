from arena.constants import Channel
from arena.types import Action, Observation, Reward

from .policy import Policy


class Custom(Policy):

    def get_action(self, observation: Observation) -> Action:
        """YOUR CODE HERE"""
        return 0

    def update(self, observation: Observation, action: Action, reward: Reward) -> None:
        """YOUR CODE HERE"""
        pass

    def reset(self) -> None:
        """YOUR CODE HERE"""
        pass
