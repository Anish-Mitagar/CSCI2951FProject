import numpy as np

from arena.constants import Channel, Direction
from arena.types import Action, Observation, Reward

from .policy import Policy


class Aggressive(Policy):
    def get_action(self, observation: Observation) -> Action:
        map_layer = observation[:, :, Channel.MAP.value]
        height, width = map_layer.shape

        # Find self position
        is_self = map_layer == self.identifier
        self_flat_idx = np.argmax(is_self.flatten())
        self_row = self_flat_idx // width
        self_col = self_flat_idx % width

        # Find enemies
        is_enemy = (map_layer > 0) & (map_layer != self.identifier)

        if not np.any(is_enemy):
            # No enemies visible, move randomly
            return np.random.randint(0, 4)

        # Compute distance to all cells
        row_grid, col_grid = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        distance_field = np.sqrt(
            (row_grid - self_row) ** 2 + (col_grid - self_col) ** 2
        )

        # Find closest enemy
        enemy_distances = np.where(is_enemy, distance_field, np.inf)
        closest_flat_idx = np.argmin(enemy_distances.flatten())
        closest_row = closest_flat_idx // width
        closest_col = closest_flat_idx % width

        # Compute direction to closest enemy
        row_diff = closest_row - self_row
        col_diff = closest_col - self_col
        abs_row = np.abs(row_diff)
        abs_col = np.abs(col_diff)

        # Choose action based on direction
        if abs_row > abs_col:
            return Direction.UP.value if row_diff < 0 else Direction.DOWN.value
        else:
            return Direction.LEFT.value if col_diff < 0 else Direction.RIGHT.value

    def update(self, observation: Observation, action: Action, reward: Reward) -> None:
        pass

    def reset(self) -> None:
        pass
