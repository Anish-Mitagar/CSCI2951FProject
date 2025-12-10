import numpy as np

from arena.constants import Cell, Channel, Direction
from arena.types import Action, Observation, Reward

from .policy import Policy


class Scanner(Policy):
    VERTICAL_DIRECTIONS = [Direction.UP.value, Direction.DOWN.value]
    HORIZONTAL_DIRECTIONS = [Direction.LEFT.value, Direction.RIGHT.value]

    def __init__(self, identifier: int):
        super().__init__(identifier)
        # Start with random vertical direction (UP or DOWN)
        self.primary_direction = np.random.choice(self.VERTICAL_DIRECTIONS)
        self.phase = "primary"  # "primary" or "redirect"

    def get_action(self, observation: Observation) -> Action:
        map_layer = observation[:, :, Channel.MAP.value]
        height, width = map_layer.shape

        # Find self position
        is_self = map_layer == self.identifier
        self_flat_idx = np.argmax(is_self.flatten())
        self_row = self_flat_idx // width
        self_col = self_flat_idx % width

        # Get direction vector from enum
        primary_dir = Direction(self.primary_direction)
        direction_vector = primary_dir.vector

        if self.phase == "primary":
            # Check if wall directly ahead in primary direction
            next_pos = np.array([self_row, self_col]) + direction_vector

            is_wall_ahead = (
                next_pos[0] < 0
                or next_pos[0] >= height
                or next_pos[1] < 0
                or next_pos[1] >= width
                or map_layer[tuple(next_pos)] == Cell.WALL.value
            )

            if is_wall_ahead:
                # Transition to REDIRECT phase
                self.phase = "redirect"
                # Randomly choose perpendicular direction
                perpendicular_direction = np.random.choice(self.HORIZONTAL_DIRECTIONS)
                return perpendicular_direction
            else:
                # Continue moving in primary direction
                return self.primary_direction

        else:  
            # phase == "redirect"
            # Just executed perpendicular move, now pick new primary direction
            self.phase = "primary"
            self.primary_direction = np.random.choice(self.VERTICAL_DIRECTIONS)
            return self.primary_direction

    def update(self, observation: Observation, action: Action, reward: Reward) -> None:
        pass

    def reset(self) -> None:
        self.primary_direction = np.random.choice(self.VERTICAL_DIRECTIONS)
        self.phase = "primary"
