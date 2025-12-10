import numpy as np

from arena.constants import AgentState


def check_battles(
    old_positions: np.ndarray,
    new_positions: np.ndarray,
    strengths: np.ndarray,
    alive: np.ndarray,
    num_players: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    in_combat = np.zeros(num_players, dtype=bool)
    rewards = np.zeros(num_players, dtype=np.float32)
    winners = np.full(num_players, AgentState.NO_LOSER.value, dtype=np.int32)
    losers = np.full(num_players, AgentState.NO_LOSER.value, dtype=np.int32)

    collision_groups = _find_collision_groups(
        old_positions, new_positions, alive, num_players
    )

    for agents_at_position in collision_groups:
        if len(agents_at_position) == 1:
            continue

        _process_battle(
            agents_at_position,
            strengths,
            num_players,
            in_combat,
            rewards,
            winners,
            losers,
        )

    return in_combat, rewards, winners, losers


def _find_collision_groups(
    old_positions: np.ndarray,
    new_positions: np.ndarray,
    alive: np.ndarray,
    num_players: int,
) -> list[list[int]]:
    """Find all groups of agents that collided (same position or swapped)."""
    position_map = _build_position_map(new_positions, alive, num_players)
    collision_groups = [
        agents for agents in position_map.values() if len(agents) > 1
    ]

    _add_swapped_collisions(
        old_positions, new_positions, alive, num_players, collision_groups
    )

    return collision_groups


def _build_position_map(
    new_positions: np.ndarray, alive: np.ndarray, num_players: int
) -> dict:
    """Build a map of positions to agent indices."""
    position_map = {}
    for index in range(num_players):
        if not alive[index]:
            continue
        position_tuple = tuple(new_positions[index])
        if position_tuple not in position_map:
            position_map[position_tuple] = []
        position_map[position_tuple].append(index)
    return position_map


def _add_swapped_collisions(
    old_positions: np.ndarray,
    new_positions: np.ndarray,
    alive: np.ndarray,
    num_players: int,
    collision_groups: list[list[int]],
) -> None:
    """Find and add agents that swapped positions to collision groups."""
    for first_agent_index in range(num_players):
        if not alive[first_agent_index]:
            continue
        for second_agent_index in range(first_agent_index + 1, num_players):
            if not alive[second_agent_index]:
                continue

            if _agents_swapped(
                old_positions, new_positions, first_agent_index, second_agent_index
            ):
                _add_to_collision_group(
                    collision_groups, first_agent_index, second_agent_index
                )


def _agents_swapped(
    old_positions: np.ndarray,
    new_positions: np.ndarray,
    first_index: int,
    second_index: int,
) -> bool:
    """Check if two agents swapped positions."""
    return np.array_equal(
        old_positions[first_index], new_positions[second_index]
    ) and np.array_equal(old_positions[second_index], new_positions[first_index])


def _add_to_collision_group(
    collision_groups: list[list[int]], first_index: int, second_index: int
) -> None:
    """Add agents to existing collision group or create new one."""
    in_group_first = any(first_index in group for group in collision_groups)
    in_group_second = any(second_index in group for group in collision_groups)

    if not in_group_first and not in_group_second:
        collision_groups.append([first_index, second_index])
    elif in_group_first and not in_group_second:
        for group in collision_groups:
            if first_index in group:
                group.append(second_index)
                break
    elif in_group_second and not in_group_first:
        for group in collision_groups:
            if second_index in group:
                group.append(first_index)
                break


def _process_battle(
    agents_at_position: list[int],
    strengths: np.ndarray,
    num_players: int,
    in_combat: np.ndarray,
    rewards: np.ndarray,
    winners: np.ndarray,
    losers: np.ndarray,
) -> None:
    """Process a battle between agents at the same position."""
    if len(agents_at_position) == 2:
        _process_two_player_battle(
            agents_at_position, strengths, num_players, in_combat, rewards,
            winners, losers
        )
    else:
        _process_multi_player_battle(
            agents_at_position, strengths, in_combat, rewards, winners, losers
        )


def _process_two_player_battle(
    agents: list[int],
    strengths: np.ndarray,
    num_players: int,
    in_combat: np.ndarray,
    rewards: np.ndarray,
    winners: np.ndarray,
    losers: np.ndarray,
) -> None:
    """Resolve a two-player battle."""
    first_index, second_index = agents
    winner_index, loser_index = _resolve_two_player(
        first_index, second_index, strengths, num_players
    )
    winners[winner_index] = winner_index
    losers[loser_index] = loser_index
    rewards[winner_index] = float(strengths[loser_index])
    in_combat[winner_index] = True
    in_combat[loser_index] = True


def _process_multi_player_battle(
    agents: list[int],
    strengths: np.ndarray,
    in_combat: np.ndarray,
    rewards: np.ndarray,
    winners: np.ndarray,
    losers: np.ndarray,
) -> None:
    """Resolve a multi-player battle."""
    survivors, losers_list = _resolve_multi_player(agents, strengths)

    for survivor_index in survivors:
        winners[survivor_index] = survivor_index
        rewards[survivor_index] = float(
            sum(strengths[i] + 1 for i in losers_list)
        ) / max(len(losers_list), 1)
        in_combat[survivor_index] = True

    for loser_index in losers_list:
        losers[loser_index] = loser_index
        in_combat[loser_index] = True


def _resolve_two_player(
    first_player_index: int,
    second_player_index: int,
    strengths: np.ndarray,
    num_players: int,
) -> tuple[int, int]:
    first_strength = strengths[first_player_index]
    second_strength = strengths[second_player_index]

    # Wrap-around rule: strength N loses to strength 1, otherwise higher wins
    if first_strength == num_players and second_strength == 1:
        return second_player_index, first_player_index
    
    if second_strength == num_players and first_strength == 1:
        return first_player_index, second_player_index

    if first_strength > second_strength:
        return first_player_index, second_player_index

    return second_player_index, first_player_index


    first_strength = strengths[first_player_index]
    second_strength = strengths[second_player_index]


def _resolve_multi_player(agents: list, strengths: np.ndarray) -> tuple[list, list]:
    strongest_index = max(agents, key=lambda index: strengths[index])
    strongest_strength = strengths[strongest_index]
    others_strength = sum(
        strengths[index] for index in agents if index != strongest_index
    )

    if strongest_strength > others_strength:
        survivors = [strongest_index]
        losers = [index for index in agents if index != strongest_index]
    else:
        survivors = [index for index in agents if index != strongest_index]
        losers = [strongest_index]

    return survivors, losers
