import json
from pathlib import Path

import numpy as np

from arena.entities import Settings, Trajectory


class Loader:
    def __init__(self, input_dir: str = "sessions"):
        self.input_dir = Path(input_dir)

    def load_trajectory(
        self, session_name: str, episode_num: int
    ) -> tuple[Trajectory, Path]:
        session_dir = self.input_dir / session_name
        episode_dir = session_dir / str(episode_num)

        # Load metadata
        with open(episode_dir / "metadata.json", "r") as file:
            metadata = json.load(file)

        config = Settings(
            height=metadata["config"]["height"],
            width=metadata["config"]["width"],
            max_steps=metadata["config"]["max_steps"],
            survival_decay=metadata["config"]["survival_decay"],
            vision_radius=metadata["config"]["vision_radius"],
        )

        # Load data
        data = np.load(episode_dir / "data.npz", allow_pickle=False)

        # Reconstruct Trajectory
        trajectory = Trajectory(
            episode=metadata["episode"],
            representations=list(data["representations"]),
            actions=list(data["actions"]),
            rewards=list(data["rewards"]),
            cumulative_returns=list(data["cumulative_returns"]),
            alive_status=list(data["alive_status"]),
            positions=list(data["positions"]),
            strengths=list(data["strengths"]),
            identifiers=list(data["identifiers"]),
            agent_names=metadata["agent_names"],
            config=config,
        )

        # Close the file
        data.close()

        return trajectory, episode_dir

    def load_session(self, session_name: str) -> list[tuple[Trajectory, Path]]:
        session_dir = self.input_dir / session_name
        trajectories = []

        for episode_dir in sorted(session_dir.iterdir()):
            if episode_dir.is_dir():
                try:
                    episode_num = int(episode_dir.name)
                    trajectory, loaded_episode_dir = self.load_trajectory(
                        session_name, episode_num
                    )
                    trajectories.append((trajectory, loaded_episode_dir))
                except (ValueError, FileNotFoundError):
                    continue

        return trajectories

    def list_sessions(self) -> list[Path]:
        if not self.input_dir.exists():
            return []

        return sorted(
            [directory for directory in self.input_dir.iterdir() if directory.is_dir()]
        )
