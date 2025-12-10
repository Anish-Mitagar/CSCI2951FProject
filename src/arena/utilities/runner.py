from typing import Type

import numpy as np

from arena.entities import Settings, Trajectory
from arena.environment import Arena
from arena.policies import Policy
from arena.utilities import logging


class Runner:
    def __init__(
        self,
        policy_types: list[Type[Policy]],
        num_episodes: int = 10,
        arena_height: int = 20,
        arena_width: int = 20,
        min_steps: int = 250,
        max_steps: int = 500,
        survival_decay: float = 0.99,
        vision_radius: int = 2,
        seed: int = 42,
    ):
        self.logger = logging.get_logger(__name__)
        self.policy_types = policy_types
        self.num_episodes = num_episodes
        self.arena_height = arena_height
        self.arena_width = arena_width
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.survival_decay = survival_decay
        self.vision_radius = vision_radius
        self.seed = seed

        self.arena = Arena(
            height=self.arena_height,
            width=self.arena_width,
            policy_types=self.policy_types,
            min_steps=min_steps,
            max_steps=max_steps,
            survival_decay=self.survival_decay,
            vision_radius=self.vision_radius,
        )

        self.logger.debug(f"Runner initialized with {len(policy_types)} agents")

    def run(self) -> list[Trajectory]:
        trajectories = []

        for episode in range(self.num_episodes):
            trajectory = self._collect_trajectory(episode)
            trajectories.append(trajectory)

            final_survivors = np.where(trajectory.alive_status[-1])[0]
            self.logger.info(
                f"  Episode {episode} complete. Survivors: {len(final_survivors)}"
            )
            self.logger.debug(f"  Final rewards: {trajectory.cumulative_returns[-1]}")

        return trajectories

    def _collect_trajectory(self, episode: int) -> Trajectory:
        timestep = self.arena.reset(seed=self.seed + episode)

        self.logger.debug(
            f"Arena reset. Total steps for episode: {self.arena.total_steps}"
        )

        representations = []
        actions = []
        rewards = []
        cumulative_returns = []
        alive_status = []
        positions = []
        strengths = []
        identifiers = []
        agent_names = [player.policy.name for player in self.arena.players]

        while not self.arena.is_done():
            timestep = self.arena.step()

            self.logger.debug(
                f"  Step {self.arena.current_step}: Actions={timestep.action}, "
                f"Rewards={timestep.reward}"
            )

            representations.append(np.array(self.arena.grid))
            actions.append(np.array(timestep.action))
            rewards.append(np.array(timestep.reward))
            cumulative_returns.append(np.array(timestep.metadata.cumulative_returns))
            alive_status.append(
                np.array([player.alive for player in self.arena.players])
            )
            positions.append(
                np.array([player.position for player in self.arena.players])
            )
            strengths.append(
                np.array([player.strength for player in self.arena.players])
            )
            identifiers.append(
                np.array([player.identifier for player in self.arena.players])
            )

        self.logger.debug(
            f"Episode {episode} ended after {self.arena.current_step} steps"
        )

        return Trajectory(
            episode=episode,
            representations=representations,
            actions=actions,
            rewards=rewards,
            cumulative_returns=cumulative_returns,
            alive_status=alive_status,
            positions=positions,
            strengths=strengths,
            identifiers=identifiers,
            agent_names=agent_names,
            config=Settings(
                height=self.arena.height,
                width=self.arena.width,
                max_steps=self.arena.total_steps,
                survival_decay=self.survival_decay,
                vision_radius=self.vision_radius,
            ),
        )
