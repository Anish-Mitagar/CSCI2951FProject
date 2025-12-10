import numpy as np

from arena.environment.player import Player
from arena.types import Observations


def compute_observation(
    grid: np.ndarray,
    players: list[Player],
    in_combat: np.ndarray,
    pre_combat_positions: np.ndarray,
    pre_combat_alive: np.ndarray,
    vision_radius: int,
    num_players: int,
    height: int,
    width: int,
    start_alive: np.ndarray = None,
) -> Observations:
    observations = np.zeros((num_players, height, width, 3), dtype=np.int32)

    # Use current alive if not specified (for initial generation)
    alive = np.array([p.alive for p in players], dtype=bool)
    mask_alive = start_alive if start_alive is not None else alive

    positions = np.array([p.position for p in players], dtype=np.int32)
    identifiers = np.array([p.identifier for p in players], dtype=np.int32)

    # Track which agents just died this step
    just_died = np.zeros(num_players, dtype=bool)
    if start_alive is not None:
        just_died = start_alive & ~alive

    for index in range(num_players):
        position = positions[index]

        # Channel 0: Map layer (walls + visible enemies)
        map_layer = _build_map_layer(
            grid,
            position,
            vision_radius,
            height,
            width,
            positions,
            mask_alive,
            identifiers,
            num_players,
        )

        # Zero out map for agents that just died
        if just_died[index]:
            map_layer = np.zeros_like(map_layer)

        # Channel 1: Pre-combat positions
        pre_combat_layer = _build_player_layer(
            pre_combat_positions,
            pre_combat_alive,
            height,
            width,
            identifiers,
            in_combat,
        )

        # Channel 2: Post-combat positions
        post_combat_layer = _build_player_layer(
            positions,
            # pre_combat_alive,
            alive,
            height,
            width,
            identifiers,
            in_combat,
        )

        observation = np.stack(
            [map_layer, pre_combat_layer, post_combat_layer], axis=-1
        )

        if not mask_alive[index]:
            observation = np.zeros_like(observation)

        observations[index] = observation

    return observations


def _build_map_layer(
    grid: np.ndarray,
    player_position: np.ndarray,
    vision_radius: int,
    height: int,
    width: int,
    all_positions: np.ndarray,
    all_alive: np.ndarray,
    identifiers: np.ndarray,
    num_players: int,
) -> np.ndarray:
    player_row, player_col = player_position

    row_grid, col_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    manhattan_distance = np.abs(row_grid - player_row) + np.abs(col_grid - player_col)
    in_vision = manhattan_distance <= vision_radius

    map_layer = np.where(in_vision, grid, 0)

    for other_player_idx in range(num_players):
        other_position = all_positions[other_player_idx]
        if not all_alive[other_player_idx]:
            continue

        distance = np.abs(other_position[0] - player_row) + np.abs(
            other_position[1] - player_col
        )
        if (
            distance <= vision_radius
            and 0 <= other_position[0] < height
            and 0 <= other_position[1] < width
        ):
            map_layer[tuple(other_position)] = identifiers[other_player_idx]

    return map_layer


def _build_player_layer(
    positions: np.ndarray,
    alive: np.ndarray,
    height: int,
    width: int,
    identifiers: np.ndarray,
    in_combat: np.ndarray = None,
) -> np.ndarray:
    layer = np.zeros((height, width), dtype=np.int32)

    for index in range(len(positions)):
        if not alive[index]:
            continue

        if in_combat is not None and not in_combat[index]:
            continue

        row, col = positions[index]

        if 0 <= row < height and 0 <= col < width:
            layer[row, col] = max(layer[row, col], identifiers[index])

    return layer
