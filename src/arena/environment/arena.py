import numpy as np

from arena.constants import NUM_ACTIONS, PLAYER_ID_OFFSET, AgentState, Cell
from arena.environment.timestep import Metadata, Timestep
from arena.policies.policy import Policy
from arena.types import Actions, Observations, Rewards, Terminations
from arena.utilities import logging

from .mechanics import combat, movement
from .mechanics import observations as observations_module
from .mechanics import rewards as reward_module
from .player import Player
from .utilities import create_walled_grid, random_positions


class Arena:
    def __init__(
        self,
        height: int,
        width: int,
        policy_types: list[type[Policy]],
        min_steps: int = 250,
        max_steps: int = 500,
        survival_decay: float = 0.99,
        vision_radius: int = 2,
    ):
        # Validate player count
        num_players = len(policy_types)
        if num_players <= 0:
            raise ValueError(f"num_players must be > 0, got {num_players}")

        # Validate step counts
        if min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")

        if max_steps < min_steps:
            raise ValueError(
                f"max_steps ({max_steps}) must be >= min_steps ({min_steps})"
            )

        # Validate survival_decay
        if not (0.0 < survival_decay <= 1.0):
            raise ValueError(
                f"survival_decay must be in (0.0, 1.0], got {survival_decay}"
            )

        # Validate vision_radius
        if vision_radius < 0:
            raise ValueError(f"vision_radius must be >= 0, got {vision_radius}")

        self.logger = logging.get_logger(__name__)
        self.height = height
        self.width = width
        self.num_players = num_players
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.survival_decay = survival_decay
        self.vision_radius = vision_radius
        self.policy_types = policy_types

        self.grid = create_walled_grid(height, width)

        self.players: list[Player] = []
        self.current_step = 0
        self.survival_time = np.zeros(self.num_players, dtype=np.int32)
        self.cumulative_returns = np.zeros(self.num_players, dtype=np.float32)
        self.in_combat = np.zeros(self.num_players, dtype=bool)
        self.combat_report = {}

        self.pre_combat_positions = np.zeros((self.num_players, 2), dtype=np.int32)
        self.pre_combat_alive = np.zeros(self.num_players, dtype=bool)

        self.logger.debug(
            f"Arena initialized: {height}x{width}, {self.num_players} players"
        )

    def reset(self, seed: int | None = None) -> Timestep:
        self._set_random_seed(seed)
        self._randomize_step_limit()
        self._initialize_players()
        self._reset_state()

        # Generate observations
        observations = self._generate_observations()

        # Build and return initial timestep
        return self._build_initial_timestep(observations)

    def _set_random_seed(self, seed: int | None) -> None:
        if seed is not None:
            np.random.seed(seed)

    def _randomize_step_limit(self) -> None:
        self.total_steps = np.random.randint(self.min_steps, self.max_steps + 1)

    def _initialize_players(self) -> None:
        occupied_grid = self.grid.copy()
        strength_assignments = np.random.permutation(self.num_players)

        if not self.players:
            self._spawn_new_players(occupied_grid, strength_assignments)
        else:
            self._respawn_existing_players(occupied_grid)

    def _spawn_new_players(
        self, occupied_grid: np.ndarray, strength_assignments: np.ndarray
    ) -> None:
        self.players = []
        for index in range(self.num_players):
            position = random_positions(occupied_grid)
            identifier = index
            strength = int(strength_assignments[index]) + 1

            player = Player(
                identifier=identifier + PLAYER_ID_OFFSET,
                position=position,
                strength=strength,
                policy_type=self.policy_types[index],
                alive=True,
            )
            self.players.append(player)
            occupied_grid[tuple(position)] = AgentState.NO_LOSER.value

            self.logger.debug(
                f"Spawned {player.policy.name} (ID: {player.identifier}, "
                f"Strength: {player.strength}) at {position}"
            )

    def _respawn_existing_players(self, occupied_grid: np.ndarray) -> None:
        for player in self.players:
            player.alive = True
            player.policy.reset()
            position = random_positions(occupied_grid)
            player.position = position
            occupied_grid[tuple(position)] = AgentState.NO_LOSER.value

    def _reset_state(self) -> None:
        self.current_step = 0
        self.survival_time = np.zeros(self.num_players, dtype=np.int32)
        self.cumulative_returns = np.zeros(self.num_players, dtype=np.float32)
        self.in_combat = np.zeros(self.num_players, dtype=bool)
        self.combat_report = {}
        self.pre_combat_positions = np.array(
            [player.position.copy() for player in self.players], dtype=np.int32
        )
        self.pre_combat_alive = np.array(
            [player.alive for player in self.players], dtype=bool
        )

    def _build_initial_timestep(self, observations: Observations) -> Timestep:
        metadata = Metadata(
            cumulative_returns=self.cumulative_returns.copy(),
        )

        self.logger.debug(f"Reset complete. Episode will last {self.total_steps} steps")

        return Timestep(
            step=0,
            observation=observations,
            action=np.full(
                self.num_players,
                AgentState.INVALID_ACTION.value,
                dtype=np.int32,
            ),
            reward=np.zeros(self.num_players, dtype=np.float32),
            terminations=np.zeros(self.num_players, dtype=bool),
            metadata=metadata,
        )

    def step(self) -> Timestep:
        if not self.players:
            raise RuntimeError("Arena not initialized. Call reset() before step()")

        # 1. Observe → Act → Execute → Regenerate Obs → Reward → Update
        observations = self._generate_observations()
        actions = self._collect_actions(observations)
        combat_state = self._execute_turn(actions)
        observations = self._regenerate_observations(combat_state)
        rewards = self._calculate_rewards(combat_state)
        self._update_players(observations, actions, rewards, combat_state)

        # 2. Build and return timestep
        self.current_step += 1
        return self._build_timestep(observations, actions, rewards)

    def _generate_observations(self) -> Observations:
        return observations_module.compute_observation(
            self.grid,
            self.players,
            self.in_combat,
            self.pre_combat_positions,
            self.pre_combat_alive,
            self.vision_radius,
            self.num_players,
            self.height,
            self.width,
        )
    
    def _collect_actions(self, observations: Observations) -> Actions:
        actions = []
        for idx, player in enumerate(self.players):
            action = player.get_action(observations[idx])
            if not (0 <= action < NUM_ACTIONS):
                self.logger.warning(
                    f"Player {player.identifier} returned invalid action {action}, "
                    f"clamping to 0"
                )
                action = 0
            actions.append(action)

        return np.array(actions, dtype=np.int32)

    def _execute_turn(self, actions: Actions) -> dict:
        # Store alive status BEFORE combat
        start_alive = np.array([p.alive for p in self.players])

        # Snapshot state before movement
        pre_positions = np.array([p.position.copy() for p in self.players])
        pre_alive = np.array([p.alive for p in self.players])

        # Move players
        new_positions = movement.apply_movement_actions(
            pre_positions,
            actions,
            self.grid,
            self.height,
            self.width,
        )

        # Update positions
        for idx, player in enumerate(self.players):
            player.position = new_positions[idx]

        # Resolve combat
        in_combat, combat_rewards, winners, losers = combat.check_battles(
            pre_positions,
            new_positions,
            np.array([p.strength for p in self.players]),
            np.array([p.alive for p in self.players]),
            self.num_players,
        )

        # Apply deaths
        for loser_idx in losers:
            if loser_idx != AgentState.NO_LOSER.value:
                self.players[loser_idx].alive = False

        # Update state for next observation
        self.pre_combat_positions = pre_positions
        self.pre_combat_alive = pre_alive
        self.in_combat = in_combat

        self._log_combat(winners, losers)

        return {
            "combat_rewards": combat_rewards,
            "winners": winners,
            "losers": losers,
            "start_alive": start_alive,
        }

    def _regenerate_observations(self, combat_state: dict) -> Observations:
        start_alive = combat_state["start_alive"]

        observations = observations_module.compute_observation(
            self.grid,
            self.players,
            self.in_combat,
            self.pre_combat_positions,
            self.pre_combat_alive,
            self.vision_radius,
            self.num_players,
            self.height,
            self.width,
            start_alive=start_alive,
        )

        # Zero out only agents who were already dead before this step
        for idx in range(self.num_players):
            if not start_alive[idx]:
                observations[idx] = np.zeros_like(observations[idx])

        return observations

    def _calculate_rewards(self, combat_state: dict) -> Rewards:
        # Update survival time
        self.survival_time += np.array([p.alive for p in self.players], dtype=np.int32)

        # Calculate survival rewards
        survival_rewards = reward_module.calculate_survival_rewards(
            # self.survival_time,
            np.array([p.strength for p in self.players]),
            np.array([p.alive for p in self.players]),
            self.num_players,
            # self.survival_decay,
            self.total_steps,
        )

        return combat_state["combat_rewards"] + survival_rewards

    def _update_players(
        self,
        observations: Observations,
        actions: Actions,
        rewards: Rewards,
        combat_state: dict,
    ) -> None:
        self.cumulative_returns += rewards

        for idx, player in enumerate(self.players):
            player.update_policy(
                observations[idx],
                int(actions[idx]),
                float(rewards[idx]),
            )

    def _build_timestep(
        self,
        observations: Observations,
        actions: Actions,
        rewards: Rewards,
    ) -> Timestep:
        terminations = self._compute_terminations()

        return Timestep(
            step=self.current_step,
            observation=observations,
            action=actions,
            reward=rewards,
            terminations=terminations,
            metadata=Metadata(cumulative_returns=self.cumulative_returns.copy()),
        )

    def _compute_terminations(self) -> Terminations:
        num_alive = sum(1 for p in self.players if p.alive)
        is_final_step = self.current_step >= self.total_steps

        return np.array(
            [not p.alive or num_alive <= 1 or is_final_step for p in self.players],
            dtype=bool,
        )

    def _log_combat(self, winners: np.ndarray, losers: np.ndarray) -> None:
        num_battles = len(winners[winners != AgentState.NO_LOSER.value])
        if num_battles == 0:
            return

        self.logger.debug(f"Battles occurred: {num_battles}")
        for winner_idx in np.where(winners != AgentState.NO_LOSER.value)[0]:
            loser_idx = losers[winner_idx]
            self.logger.debug(
                f"  {self.players[winner_idx].policy.name} (S:{self.players[winner_idx].strength}) "
                f"defeated {self.players[loser_idx].policy.name} (S:{self.players[loser_idx].strength})"
            )

    def get_alive_count(self) -> int:
        if not self.players:
            raise RuntimeError("Arena not initialized. Call reset() first.")

        return sum(1 for player in self.players if player.alive)

    def is_done(self) -> bool:
        if not self.players:
            raise RuntimeError("Arena not initialized. Call reset() first.")

        num_alive = self.get_alive_count()
        is_final_step = self.current_step >= self.total_steps

        return num_alive <= 1 or is_final_step
