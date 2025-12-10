import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arena.entities import Trajectory

AGENT_COLORS = [
    [100, 149, 237],
    [255, 127, 80],
    [144, 238, 144],
    [255, 215, 0],
    [218, 112, 214],
    [64, 224, 208],
    [255, 105, 180],
    [255, 165, 0],
    [147, 112, 219],
    [60, 179, 113],
    [250, 128, 114],
    [135, 206, 235],
    [240, 230, 140],
    [221, 160, 221],
    [0, 139, 139],
    [205, 92, 92],
    [32, 178, 170],
    [188, 143, 143],
    [106, 90, 205],
    [255, 99, 71],
    [0, 191, 255],
    [152, 251, 152],
    [233, 150, 122],
    [102, 205, 170],
    [238, 130, 238],
    [50, 205, 50],
    [220, 20, 60],
    [0, 128, 128],
    [128, 0, 0],
    [0, 0, 128],
]

GRID_SIZE = 800
AGENT_BAR_WIDTH = 400
STEP_BAR_HEIGHT = 40
PANEL_WIDTH = GRID_SIZE + AGENT_BAR_WIDTH


class Renderer:

    def __init__(self):
        self.font_large = self._load_font(size=12, bold=True)
        self.font_agent_label = self._load_font(size=11, bold=True)
        self.font_normal = self._load_font(size=9, bold=False)

    def _find_font(self, bold: bool = False) -> str:
        font_names_bold = [
            "DejaVuSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
            "ArialBD.ttf",
            "Arial-Bold.ttf",
            "Helvetica.ttc",
        ]
        font_names_normal = [
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
            "Arial.ttf",
            "Helvetica.ttc",
            "NotoSans-Regular.ttf",
        ]

        font_names = font_names_bold if bold else font_names_normal

        # Common font directories
        font_dirs = [
            # Linux
            "/usr/share/fonts/truetype/dejavu/",
            "/usr/share/fonts/truetype/liberation/",
            "/usr/share/fonts/truetype/freefont/",
            "/usr/share/fonts/opentype/noto/",
            # Mac
            "/Library/Fonts/",
            "/System/Library/Fonts/",
            os.path.expanduser("~/Library/Fonts/"),
            # Windows
            "C:\\Windows\\Fonts\\",
            "C:\\Windows\\System32\\",
            # Cross-platform
            os.path.expanduser("~/.fonts/"),
        ]

        # Try to find fonts in common directories
        for font_dir in font_dirs:
            if not os.path.exists(font_dir):
                continue

            for font_name in font_names:
                font_path = os.path.join(font_dir, font_name)
                if os.path.exists(font_path):
                    return font_path

        return None

    def _load_font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        font_path = self._find_font(bold=bold)

        if font_path:
            try:
                return ImageFont.truetype(font_path, size=size)
            except:
                pass

        return ImageFont.load_default()

    def render(self, trajectory: Trajectory, skip_frames: int = 1) -> list:
        frames = []
        for step_index in range(0, len(trajectory.representations), skip_frames):
            frame = self._render_frame(step_index, trajectory)
            frames.append(frame)
        return frames

    def _render_frame(self, step_index: int, trajectory: Trajectory) -> np.ndarray:
        step_bar = self._render_step_bar(step_index)

        grid = trajectory.representations[step_index]
        original_height = grid.shape[0]

        grid_image = self._render_grid(
            grid,
            trajectory.positions[step_index],
            trajectory.alive_status[step_index],
            trajectory.config.vision_radius,
        )
        grid_image = self._resize_to_fixed_size(grid_image, GRID_SIZE, GRID_SIZE)
        grid_image = self._add_agent_labels(
            grid_image,
            trajectory.positions[step_index],
            trajectory.alive_status[step_index],
            trajectory.identifiers[step_index],
            original_height,
        )

        agent_bar = self._render_agent_bar(step_index, trajectory)

        combined = np.concatenate([grid_image, agent_bar], axis=1)
        return np.concatenate([step_bar, combined], axis=0)

    def _render_step_bar(self, step_index: int) -> np.ndarray:
        bar = Image.new("RGB", (PANEL_WIDTH, STEP_BAR_HEIGHT), (240, 240, 240))
        draw = ImageDraw.Draw(bar)
        draw.text((20, 10), f"STEP {step_index}", fill=(0, 0, 0), font=self.font_large)
        return np.array(bar)

    def _render_grid(
        self,
        grid: np.ndarray,
        positions: np.ndarray,
        alive: np.ndarray,
        vision_radius: int,
    ) -> np.ndarray:
        height, width = grid.shape
        rgb = np.full((height, width, 3), [220, 220, 220], dtype=np.uint8)

        # Walls
        rgb[grid == -1] = [64, 64, 64]

        # Pre-compute distance grids once
        row_grid, col_grid = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )

        # Vision highlights (vectorized)
        for agent_index in range(len(alive)):
            if not alive[agent_index]:
                continue
            agent_row, agent_col = int(positions[agent_index][0]), int(
                positions[agent_index][1]
            )
            manhattan = np.abs(row_grid - agent_row) + np.abs(col_grid - agent_col)
            mask = (manhattan <= vision_radius) & (grid != -1)
            rgb[mask] = [200, 220, 255]

        # Place agents (vectorized)
        alive_indices = np.where(alive)[0]
        for agent_index in alive_indices:
            agent_row, agent_col = int(positions[agent_index][0]), int(
                positions[agent_index][1]
            )
            rgb[agent_row, agent_col] = AGENT_COLORS[agent_index % len(AGENT_COLORS)]

        return rgb

    def _resize_to_fixed_size(
        self, image: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        return np.array(Image.fromarray(image).resize((width, height), Image.NEAREST))

    def _add_agent_labels(
        self,
        image: np.ndarray,
        positions: np.ndarray,
        alive: np.ndarray,
        identifiers: np.ndarray,
        original_grid_height: int,
    ) -> np.ndarray:
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        scale = GRID_SIZE / original_grid_height

        for agent_index in np.where(alive)[0]:
            agent_row, agent_col = int(positions[agent_index][0]), int(
                positions[agent_index][1]
            )
            text_x = int((agent_col + 0.5) * scale)
            text_y = int((agent_row + 0.5) * scale)
            draw.text(
                (text_x - 5, text_y - 5),
                str(int(identifiers[agent_index])),
                fill=(0, 0, 0),
                font=self.font_agent_label,
            )

        return np.array(pil_image)

    def _render_agent_bar(self, step_index: int, trajectory: Trajectory) -> np.ndarray:
        bar = Image.new("RGB", (AGENT_BAR_WIDTH, GRID_SIZE), (240, 240, 240))
        draw = ImageDraw.Draw(bar)

        alive = trajectory.alive_status[step_index]
        num_agents = len(alive)
        box_height = GRID_SIZE // num_agents

        for agent_index in range(num_agents):
            color = tuple(AGENT_COLORS[agent_index % len(AGENT_COLORS)])
            box_top = agent_index * box_height

            draw.rectangle(
                [(0, box_top), (AGENT_BAR_WIDTH, box_top + box_height)],
                fill=color,
                outline=(0, 0, 0),
                width=1,
            )

            agent_name = trajectory.agent_names[agent_index]
            agent_id = int(trajectory.identifiers[step_index][agent_index])
            reward = trajectory.cumulative_returns[step_index][agent_index]
            action = int(trajectory.actions[step_index][agent_index])
            strength = int(trajectory.strengths[step_index][agent_index])

            draw.text(
                (4, box_top + 3), agent_name, fill=(0, 0, 0), font=self.font_large
            )

            parts = [
                f"ID: {agent_id}",
                f"ALIVE: {bool(alive[agent_index])}",
                f"REWARD: {reward:.1f}",
                f"ACTION: {action}",
                f"STRENGTH: {strength}",
            ]

            draw.text(
                (4, box_top + 20),
                "       ".join(parts),
                fill=(0, 0, 0),
                font=self.font_normal,
            )

        return np.array(bar)

    def save_gif(
        self,
        trajectory: Trajectory,
        episode_dir: Path | str,
        output_path: str | Path | None = None,
        fps: int = 2,
        skip_frames: int = 1,
    ) -> None:
        frames = self.render(trajectory, skip_frames=skip_frames)

        if output_path is None:
            episode_dir = Path(episode_dir)
            output_path = episode_dir / "recording.gif"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = int(1000 / fps)
        pil_images = [Image.fromarray(frame) for frame in frames]
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0,
        )
