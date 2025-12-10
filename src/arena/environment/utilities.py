import numpy as np


def create_walled_grid(height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.int32)

    grid[0, :] = -1
    grid[-1, :] = -1
    grid[:, 0] = -1
    grid[:, -1] = -1

    return grid


def random_positions(grid: np.ndarray) -> np.ndarray:
    empty_cells = np.argwhere(grid == 0)

    if len(empty_cells) == 0:
        raise ValueError("No empty cells in grid")

    position = empty_cells[np.random.choice(len(empty_cells))]

    return position.astype(np.int32)
