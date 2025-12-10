from enum import Enum

import numpy as np


class Cell(Enum):
    EMPTY = 0
    WALL = -1


class Channel(Enum):
    MAP = 0
    PRE_COMBAT = 1
    POST_COMBAT = 2


class AgentState(Enum):
    NO_LOSER = -1
    INVALID_ACTION = -1


class Direction(Enum):
    UP = (0, [-1, 0])
    RIGHT = (1, [0, 1])
    DOWN = (2, [1, 0])
    LEFT = (3, [0, -1])

    def __new__(cls, action_value: int, vector: list):
        obj = object.__new__(cls)
        obj._value_ = action_value
        obj.vector = np.array(vector, dtype=np.int32)
        return obj


# Actions
NUM_ACTIONS = 4

# Grid and walls
PLAYER_ID_OFFSET = 1

MIN_PLAYERS = 2
MAX_PLAYERS = 20

# Observation
OBSERVATION_CHANNELS = 3

# Grid values (deprecated, use GridCell enum instead)
GRID_EMPTY = 0
GRID_WALL = -1

# Direction vectors (deprecated, use Direction enum instead)
DIRECTION_VECTORS = [
    [-1, 0],  # up
    [0, 1],  # right
    [1, 0],  # down
    [0, -1],  # left
]

# Rendering - Grid and layout
GRID_SIZE = 800
AGENT_BAR_WIDTH = 400
STEP_BAR_HEIGHT = 40
PANEL_WIDTH = GRID_SIZE + AGENT_BAR_WIDTH

# Rendering - Text positioning
STEP_TEXT_X = 20
STEP_TEXT_Y = 10
AGENT_NAME_X = 4
AGENT_NAME_Y = 3
AGENT_STATS_X = 4
AGENT_STATS_Y = 20
AGENT_LABEL_TEXT_OFFSET_X = -5
AGENT_LABEL_TEXT_OFFSET_Y = -5

# Rendering - Agent label positioning
AGENT_LABEL_CENTER_OFFSET = 0.5

# Rendering - Fonts
FONT_SIZE_LARGE = 12
FONT_SIZE_AGENT_LABEL = 11
FONT_SIZE_NORMAL = 9

# Rendering - Colors
AGENT_COLORS = [
    [100, 149, 237],  # cornflower blue
    [255, 127, 80],  # coral
    [144, 238, 144],  # light green
    [255, 215, 0],  # gold
    [218, 112, 214],  # orchid
    [64, 224, 208],  # turquoise
    [255, 105, 180],  # hot pink
    [255, 165, 0],  # orange
    [147, 112, 219],  # medium purple
    [60, 179, 113],  # medium sea green
    [250, 128, 114],  # salmon
    [135, 206, 235],  # sky blue
    [240, 230, 140],  # khaki
    [221, 160, 221],  # plum
    [0, 139, 139],  # dark cyan
    [205, 92, 92],  # indian red
    [32, 178, 170],  # light sea green
    [188, 143, 143],  # rosy brown
    [106, 90, 205],  # slate blue
    [255, 99, 71],  # tomato
    [0, 191, 255],  # deep sky blue
    [152, 251, 152],  # pale green
    [233, 150, 122],  # light salmon
    [102, 205, 170],  # medium aquamarine
    [238, 130, 238],  # violet
    [50, 205, 50],  # lime green
    [220, 20, 60],  # crimson
    [0, 128, 128],  # teal
    [128, 0, 0],  # maroon
    [0, 0, 128],  # navy
]

BACKGROUND_COLOR = [220, 220, 220]
WALL_COLOR = [64, 64, 64]
VISION_HIGHLIGHT_COLOR = [200, 220, 255]
TEXT_COLOR_DARK = (0, 0, 0)
TEXT_COLOR_LIGHT = (255, 255, 255)
AGENT_BOX_OUTLINE_COLOR = (0, 0, 0)
AGENT_BOX_OUTLINE_WIDTH = 1
