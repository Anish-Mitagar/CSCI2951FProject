import numpy as np


def calculate_survival_rewards(
    strengths: np.ndarray,
    alive: np.ndarray,
    num_players: int,
    total_steps: int,
) -> np.ndarray:
    max_reward = num_players * (num_players + 1) // 2
    return (max_reward / (2 * strengths * total_steps)) * alive.astype(np.float32)


# def calculate_survival_rewards(
#    survival_time: np.ndarray,
#    strengths: np.ndarray,
#    alive: np.ndarray,
#    num_players: int,
#    survival_decay: float,
#    total_steps: int,
# ) -> np.ndarray:
#    decay = np.power(survival_decay, -survival_time) / (1 + 0.01 * survival_time)
#
#    max_decay = np.power(survival_decay, -total_steps)
#    scale_factor = (num_players - 0) * max_decay / 0.5
#
#    bonus = (num_players - strengths).astype(np.float32)
#    rewards = (decay / scale_factor) * bonus * alive.astype(np.float32)
#
#    return rewards
