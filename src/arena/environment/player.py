import numpy as np

from arena.policies.policy import Policy
from arena.types import Action, Observation, Position, Reward


class Player:
    def __init__(
        self,
        identifier: int,
        position: Position,
        strength: int,
        policy_type: type[Policy],
        alive: bool = True,
    ):
        self.identifier = identifier
        self.strength = strength
        self.policy_type = policy_type
        self.alive = alive

        self.position = np.array(position, dtype=np.int32)
        self.policy = policy_type(self.identifier)

    def get_action(self, observation: Observation) -> Action:
        if not self.alive:
            return 0  # Dead agents stay still

        return self.policy.get_action(observation)

    def update_policy(
        self,
        observation: Observation,
        action: Action,
        reward: Reward,
    ) -> None:
        self.policy.update(observation, action, reward)
