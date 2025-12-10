from abc import ABC, abstractmethod

from arena.types import Action, Observation, Reward


class Policy(ABC):

    def __init__(self, identifier: int):
        self.identifier = identifier

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get_action(self, observation: Observation) -> Action:
        raise NotImplementedError()

    @abstractmethod
    def update(
        self,
        observation: Observation,
        action: Action,
        reward: Reward,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()
