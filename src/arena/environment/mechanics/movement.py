import numpy as np

from arena.constants import Cell, Direction


def apply_movement_actions(
    positions: np.ndarray,
    actions: np.ndarray,
    grid: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    new_positions = positions.copy()

    for player_index in range(len(positions)):
        action = int(actions[player_index])

        # Get direction from enum
        if 0 <= action < 4:
            direction = Direction(action)
            movement_vector = direction.vector
        else:
            # Invalid action, don't move
            continue

        # Calculate new position
        new_row = positions[player_index][0] + movement_vector[0]
        new_col = positions[player_index][1] + movement_vector[1]

        # Check bounds
        if not (0 <= new_row < height and 0 <= new_col < width):
            continue

        # Check for wall
        if grid[new_row, new_col] == Cell.WALL.value:
            continue

        # Valid move
        new_positions[player_index] = np.array([new_row, new_col], dtype=np.int32)

    return new_positions
