import json
from datetime import datetime
from pathlib import Path

import numpy as np

from arena.entities import Trajectory


class Saver:
    def __init__(self, output_dir: str = "sessions"):
        self.output_dir = Path(output_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.output_dir / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trajectories: list[Trajectory]) -> None:
        for trajectory in trajectories:
            episode_dir = self.session_dir / str(trajectory.episode)
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Save all numpy data as compressed archive
            np.savez_compressed(
                episode_dir / "data.npz",
                representations=np.array(trajectory.representations),
                actions=np.array(trajectory.actions),
                rewards=np.array(trajectory.rewards),
                cumulative_returns=np.array(trajectory.cumulative_returns),
                alive_status=np.array(trajectory.alive_status),
                positions=np.array(trajectory.positions),
                strengths=np.array(trajectory.strengths),
                identifiers=np.array(trajectory.identifiers),
            )

            # Save metadata as JSON
            metadata = {
                "episode": trajectory.episode,
                "agent_names": trajectory.agent_names,
                "config": {
                    "height": trajectory.config.height,
                    "width": trajectory.config.width,
                    "max_steps": trajectory.config.max_steps,
                    "survival_decay": trajectory.config.survival_decay,
                    "vision_radius": trajectory.config.vision_radius,
                },
            }

            with open(episode_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
