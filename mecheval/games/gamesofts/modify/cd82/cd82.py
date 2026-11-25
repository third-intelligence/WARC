"""


























"""

from typing import cast

import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)

from gamesofts._action_log import ActionLogState, default_aggregator, compose_default_summary


TRANSPARENT   = -1
WHITE         =  0
OFF_WHITE     =  1
LIGHT_GRAY    =  2
GRAY          =  3
DARK_GRAY     =  4
BLACK         =  5
MAGENTA       =  6
PINK          =  7
RED           =  8
BLUE          =  9
LIGHT_BLUE    = 10
YELLOW        = 11
ORANGE        = 12
MAROON        = 13
GREEN         = 14
PURPLE        = 15


sprites = {

    "crane_arm": Sprite(
        pixels=[
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, LIGHT_GRAY, WHITE, WHITE, LIGHT_GRAY, LIGHT_GRAY],
            [TRANSPARENT, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, TRANSPARENT],
        ],
        name="crane_arm",
        visible=True,
        collidable=True,
        layer=2,
    ),

    "target_pattern_1": Sprite(
        pixels=[
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
        ],
        name="target_pattern_1",
        visible=True,
        collidable=True,
    ),

    "target_pattern_2": Sprite(
        pixels=[
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, ORANGE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, ORANGE, ORANGE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, ORANGE, ORANGE, ORANGE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, ORANGE, ORANGE, ORANGE, ORANGE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [WHITE, WHITE, WHITE, WHITE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [WHITE, WHITE, WHITE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [WHITE, WHITE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [WHITE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
        ],
        name="target_pattern_2",
        visible=True,
        collidable=True,
    ),

    "target_pattern_3": Sprite(
        pixels=[
            [PURPLE, PURPLE, PURPLE, ORANGE, ORANGE, ORANGE, ORANGE, PURPLE, PURPLE, PURPLE],
            [PURPLE, PURPLE, PURPLE, ORANGE, ORANGE, ORANGE, ORANGE, PURPLE, PURPLE, GREEN],
            [PURPLE, PURPLE, PURPLE, ORANGE, ORANGE, ORANGE, ORANGE, PURPLE, GREEN, GREEN],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, GREEN, GREEN, GREEN],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, GREEN, GREEN, GREEN, GREEN],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, GREEN, GREEN, GREEN, GREEN, GREEN],
            [PURPLE, PURPLE, PURPLE, PURPLE, RED, GREEN, GREEN, GREEN, GREEN, GREEN],
            [PURPLE, PURPLE, PURPLE, RED, RED, GREEN, GREEN, GREEN, GREEN, GREEN],
            [PURPLE, PURPLE, RED, RED, RED, GREEN, GREEN, GREEN, GREEN, GREEN],
            [PURPLE, RED, RED, RED, RED, GREEN, GREEN, GREEN, GREEN, GREEN],
        ],
        name="target_pattern_3",
        visible=True,
        collidable=True,
    ),

    "target_pattern_4": Sprite(
        pixels=[
            [BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE, ORANGE, PURPLE],
            [BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE, PURPLE, PURPLE],
            [BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE, ORANGE, PURPLE, PURPLE, PURPLE],
            [YELLOW, YELLOW, YELLOW, BLUE, BLUE, ORANGE, PURPLE, PURPLE, PURPLE, PURPLE],
            [YELLOW, YELLOW, YELLOW, BLUE, BLUE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [YELLOW, YELLOW, YELLOW, BLUE, BLUE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [YELLOW, YELLOW, YELLOW, BLUE, BLUE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [BLUE, BLUE, BLUE, BLUE, BLUE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [BLUE, BLUE, BLUE, BLUE, BLUE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [BLUE, BLUE, BLUE, BLUE, BLUE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
        ],
        name="target_pattern_4",
        visible=True,
        collidable=True,
    ),

    "target_pattern_5": Sprite(
        pixels=[
            [GREEN, BLUE, BLUE, RED, RED, RED, RED, BLUE, BLUE, ORANGE],
            [GREEN, GREEN, BLUE, RED, RED, RED, RED, BLUE, ORANGE, ORANGE],
            [GREEN, GREEN, GREEN, RED, RED, RED, RED, ORANGE, ORANGE, ORANGE],
            [GREEN, GREEN, GREEN, GREEN, BLUE, BLUE, ORANGE, ORANGE, ORANGE, ORANGE],
            [GREEN, GREEN, GREEN, GREEN, GREEN, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [GREEN, GREEN, GREEN, GREEN, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [GREEN, GREEN, GREEN, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [GREEN, GREEN, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [GREEN, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
            [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE],
        ],
        name="target_pattern_5",
        visible=True,
        collidable=True,
    ),

    "target_pattern_6": Sprite(
        pixels=[
            [RED, RED, RED, PURPLE, PURPLE, PURPLE, PURPLE, RED, RED, RED],
            [RED, RED, RED, PURPLE, PURPLE, PURPLE, PURPLE, RED, RED, GREEN],
            [RED, RED, RED, PURPLE, PURPLE, PURPLE, PURPLE, RED, GREEN, GREEN],
            [YELLOW, YELLOW, YELLOW, RED, RED, RED, RED, GREEN, GREEN, GREEN],
            [YELLOW, YELLOW, YELLOW, RED, RED, RED, GREEN, GREEN, GREEN, GREEN],
            [YELLOW, YELLOW, YELLOW, RED, RED, GREEN, GREEN, GREEN, GREEN, GREEN],
            [YELLOW, YELLOW, YELLOW, RED, WHITE, GREEN, GREEN, GREEN, GREEN, GREEN],
            [RED, RED, RED, WHITE, WHITE, GREEN, GREEN, GREEN, GREEN, GREEN],
            [RED, RED, WHITE, WHITE, WHITE, GREEN, GREEN, GREEN, GREEN, GREEN],
            [RED, WHITE, WHITE, WHITE, WHITE, GREEN, GREEN, GREEN, GREEN, GREEN],
        ],
        name="target_pattern_6",
        visible=True,
        collidable=True,
    ),

    "grid_background": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        name="grid_background",
        visible=True,
        collidable=True,
        layer=-1,
    ),

    "bucket_diagonal": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, -1, LIGHT_GRAY, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, LIGHT_GRAY, LIGHT_GRAY, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, LIGHT_GRAY, LIGHT_GRAY, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, LIGHT_GRAY, LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1, -1, -1],
            [-1, LIGHT_GRAY, LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1, -1],
            [LIGHT_GRAY, LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1],
            [-1, -1, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1],
            [-1, -1, -1, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1],
            [-1, -1, -1, -1, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1],
            [-1, -1, -1, -1, -1, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY],
            [-1, -1, -1, -1, -1, -1, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1],
            [-1, -1, -1, -1, -1, -1, -1, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, PURPLE, PURPLE, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, LIGHT_GRAY, LIGHT_GRAY, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, LIGHT_GRAY, -1, -1, -1, -1, -1, -1],
        ],
        name="bucket_diagonal",
        visible=True,
        collidable=True,
        layer=2,
    ),

    "bucket_horizontal": Sprite(
        pixels=[
            [LIGHT_GRAY, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, LIGHT_GRAY],
            [LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY, LIGHT_GRAY],
        ],
        name="bucket_horizontal",
        visible=True,
        collidable=True,
    ),

    "color_button": Sprite(
        pixels=[
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, WHITE, WHITE, WHITE, DARK_GRAY],
            [DARK_GRAY, WHITE, WHITE, WHITE, DARK_GRAY],
            [DARK_GRAY, WHITE, WHITE, WHITE, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
        ],
        name="color_button",
        visible=True,
        collidable=True,
    ),

    "canvas": Sprite(
        pixels=[
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
        ],
        name="canvas",
        visible=True,
        collidable=True,
    ),

    "color_indicator": Sprite(
        pixels=[
            [WHITE, WHITE, WHITE, WHITE, WHITE],
        ],
        name="color_indicator",
        visible=True,
        collidable=True,
    ),
}


levels = [

    Level(
        sprites=[
            sprites["target_pattern_1"].clone().set_position(3, 3),
            sprites["grid_background"].clone(),
            sprites["bucket_horizontal"].clone().set_position(25, 24).set_rotation(180),
            sprites["color_button"].clone().set_position(35, 2),
            sprites["color_button"].clone().set_position(41, 2).color_remap(0, 15),
            sprites["canvas"].clone().set_position(27, 34),
            sprites["color_indicator"].clone().set_position(41, 7),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["target_pattern_2"].clone().set_position(3, 3),
            sprites["grid_background"].clone(),
            sprites["bucket_horizontal"].clone().set_position(25, 24).set_rotation(180),
            sprites["color_button"].clone().set_position(32, 2),
            sprites["color_button"].clone().set_position(38, 2).color_remap(0, 15),
            sprites["color_button"].clone().set_position(44, 2).color_remap(0, 12),
            sprites["canvas"].clone().set_position(27, 34),
            sprites["color_indicator"].clone().set_position(38, 7),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["crane_arm"].clone().set_position(29, 18).set_rotation(180),
            sprites["target_pattern_3"].clone().set_position(3, 3),
            sprites["grid_background"].clone(),
            sprites["bucket_horizontal"].clone().set_position(25, 24).set_rotation(180),
            sprites["color_button"].clone().set_position(21, 2),
            sprites["color_button"].clone().set_position(27, 2).color_remap(0, 15),
            sprites["color_button"].clone().set_position(33, 2).color_remap(0, 12),
            sprites["color_button"].clone().set_position(39, 2).color_remap(0, 11),
            sprites["color_button"].clone().set_position(45, 2).color_remap(0, 14),
            sprites["color_button"].clone().set_position(51, 2).color_remap(0, 8),
            sprites["color_button"].clone().set_position(57, 2).color_remap(0, 9),
            sprites["canvas"].clone().set_position(27, 34),
            sprites["color_indicator"].clone().set_position(27, 7),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["crane_arm"].clone().set_position(29, 18).set_rotation(180),
            sprites["target_pattern_4"].clone().set_position(3, 3),
            sprites["grid_background"].clone(),
            sprites["bucket_horizontal"].clone().set_position(25, 24).set_rotation(180),
            sprites["color_button"].clone().set_position(21, 2),
            sprites["color_button"].clone().set_position(27, 2).color_remap(0, 15),
            sprites["color_button"].clone().set_position(33, 2).color_remap(0, 12),
            sprites["color_button"].clone().set_position(39, 2).color_remap(0, 11),
            sprites["color_button"].clone().set_position(45, 2).color_remap(0, 14),
            sprites["color_button"].clone().set_position(51, 2).color_remap(0, 8),
            sprites["color_button"].clone().set_position(57, 2).color_remap(0, 9),
            sprites["canvas"].clone().set_position(27, 34),
            sprites["color_indicator"].clone().set_position(27, 7),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["crane_arm"].clone().set_position(29, 18).set_rotation(180),
            sprites["target_pattern_5"].clone().set_position(3, 3),
            sprites["grid_background"].clone(),
            sprites["bucket_horizontal"].clone().set_position(25, 24).set_rotation(180),
            sprites["color_button"].clone().set_position(21, 2),
            sprites["color_button"].clone().set_position(27, 2).color_remap(0, 15),
            sprites["color_button"].clone().set_position(33, 2).color_remap(0, 12),
            sprites["color_button"].clone().set_position(39, 2).color_remap(0, 11),
            sprites["color_button"].clone().set_position(45, 2).color_remap(0, 14),
            sprites["color_button"].clone().set_position(51, 2).color_remap(0, 8),
            sprites["color_button"].clone().set_position(57, 2).color_remap(0, 9),
            sprites["canvas"].clone().set_position(27, 34),
            sprites["color_indicator"].clone().set_position(27, 7),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["crane_arm"].clone().set_position(29, 18).set_rotation(180),
            sprites["target_pattern_6"].clone().set_position(3, 3),
            sprites["grid_background"].clone(),
            sprites["bucket_horizontal"].clone().set_position(25, 24).set_rotation(180),
            sprites["color_button"].clone().set_position(21, 2),
            sprites["color_button"].clone().set_position(27, 2).color_remap(0, 15),
            sprites["color_button"].clone().set_position(33, 2).color_remap(0, 12),
            sprites["color_button"].clone().set_position(39, 2).color_remap(0, 11),
            sprites["color_button"].clone().set_position(45, 2).color_remap(0, 14),
            sprites["color_button"].clone().set_position(51, 2).color_remap(0, 8),
            sprites["color_button"].clone().set_position(57, 2).color_remap(0, 9),
            sprites["canvas"].clone().set_position(27, 34),
            sprites["color_indicator"].clone().set_position(27, 7),
        ],
        grid_size=(64, 64),
    ),
]

BACKGROUND_COLOR = BLACK
PADDING_COLOR    = DARK_GRAY




class StepProgressBar(RenderableUserDisplay):
    """



    """

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.current_steps = max_steps

    def set_steps(self, steps: int) -> None:
        """"""
        self.current_steps = max(0, min(steps, self.max_steps))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """"""
        if self.max_steps == 0:
            return frame
        width = frame.shape[1]

        ratio = self.current_steps / self.max_steps
        filled_width = round(width * ratio)
        filled_width = min(filled_width, width)
        for x in range(width):
            if x < filled_width:
                frame[63, x] = DARK_GRAY
            else:
                frame[63, x] = BLACK
        return frame



COLOR_PALETTE: dict[int, dict] = {
    -1: {"name": "TRANSPARENT", "hex": None,      "label": "Transparent", "label_l2": ''      },
     0: {"name": "WHITE",       "hex": "#FFFFFF",  "label": "White",       "label_l2": ''   },
     1: {"name": "OFF_WHITE",   "hex": "#CCCCCC",  "label": "Off White",   "label_l2": ''                  },
     2: {"name": "LIGHT_GRAY",  "hex": "#999999",  "label": "Light Gray",  "label_l2": ''                  },
     3: {"name": "GRAY",        "hex": "#666666",  "label": "Gray",        "label_l2": ''         },
     4: {"name": "DARK_GRAY",   "hex": "#333333",  "label": "Dark Gray",   "label_l2": ''                  },
     5: {"name": "BLACK",       "hex": "#000000",  "label": "Black",       "label_l2": ''   },
     6: {"name": "MAGENTA",     "hex": "#E53AA3",  "label": "Magenta",     "label_l2": ''            },
     7: {"name": "PINK",        "hex": "#FF7BCC",  "label": "Pink",        "label_l2": ''         },
     8: {"name": "RED",         "hex": "#F93C31",  "label": "Red",         "label_l2": ''   },
     9: {"name": "BLUE",        "hex": "#1E93FF",  "label": "Blue",        "label_l2": ''   },
    10: {"name": "LIGHT_BLUE",  "hex": "#88D8F1",  "label": "Light Blue",  "label_l2": ''                  },
    11: {"name": "YELLOW",      "hex": "#FFDC00",  "label": "Yellow",      "label_l2": ''   },
    12: {"name": "ORANGE",      "hex": "#FF851B",  "label": "Orange",      "label_l2": ''            },
    13: {"name": "MAROON",      "hex": "#921231",  "label": "Maroon",      "label_l2": ''            },
    14: {"name": "GREEN",       "hex": "#4FCC30",  "label": "Green",       "label_l2": ''   },
    15: {"name": "PURPLE",      "hex": "#A356D6",  "label": "Purple",      "label_l2": ''   },
}


TAG_DISPLAY_NAMES: dict[str, str] = {
    "target":    "Target Pattern",
    "bucket":    "Bucket",
    "crane":     "Crane Arm",
    "button":    "Color Button",
    "canvas":    "Canvas",
    "indicator": "Color Indicator",
    "bg":        "Grid Background",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "target":    "",
    "bucket":    "",
    "crane":     "",
    "button":    "",
    "canvas":    "",
    "indicator": "",
    "bg":        "",
}



SPRITE_NAME_DISPLAY: list[tuple[str, str, str]] = [

    ("target_pattern",     "Target Pattern",      ""),
    ("grid_background",    "Grid Background",     ""),
    ("bucket_horizontal",  "Horizontal Bucket",   ""),
    ("bucket_vertical",    "Vertical Bucket",     ""),
    ("bucket_diagonal",    "Diagonal Bucket",     ""),
    ("bucket",             "Bucket",              ""),
    ("ActiveBasket",       "Active Basket",       ""),
    ("ExampleDiagonal",    "Example Diagonal",    ""),
    ("color_button",       "Color Button",        ""),
    ("color_indicator",    "Color Indicator",     ""),
    ("canvas",             "Canvas",              ""),
    ("crane_arm",          "Crane Arm",           ""),
    ("crane",              "Crane",               ""),
]

def _lookup_sprite_label(sprite_name: str) -> tuple[str, str]:
    """"""
    for prefix, en, l2 in SPRITE_NAME_DISPLAY:
        if sprite_name.startswith(prefix):
            return en, l2
    return sprite_name, sprite_name



GAME_KNOWLEDGE: dict = {

    "color_palette": {str(k): v for k, v in COLOR_PALETTE.items()},


    "tag_display_names": TAG_DISPLAY_NAMES,


    "tag_display_names_l2": TAG_DISPLAY_NAMES_L2,


    "global_knowledge": {
        "game_id":    "cd82",
        "title":      "Crane Dipper 82",
        "title_l2":   ''                             ,
        "description": (
            "A painting puzzle where you rotate a crane to dip a bucket into colors "
            "and paint regions of a canvas to match a target pattern."
        ),
        "description_l2": (
            " "
            ""
        ),
        "default_grid_size": [64, 64],
        "num_levels": len(levels),

        "goal": (
            "Paint the canvas to exactly match the target pattern shown in the "
            "top-left corner, within the step limit."
        ),
        "goal_l2": (
            " "
            ""
        ),
        "win_condition":    "Canvas pixel array matches target_pattern pixel array exactly (ignoring diagonal mask).",
        "win_condition_l2": ''                                                                                                                                          ,
        "lose_condition":   "Step counter reaches 0 (no steps remaining).",
        "lose_condition_l2": ''                                                                         ,

        "actions": {
            "ACTION1": {
                "key": "up",
                "effect": "Move the active basket one cell up within the 3x3 perimeter grid (col - 1). The active basket lives on the 8 perimeter cells (center excluded), so each successful move corresponds to a 45° rotation around the canvas; the rotation direction (clockwise vs counter-clockwise) depends on the starting side.",
                "effect_l2": ''                                                                                                                                                                                                                                                                                                                                                   ,
                "restrictions": "No-op if the move would land on the center cell (1,1) or outside the 3x3 grid.",
                "restrictions_l2": ''                                                                                                        ,
                "step_cost": 1,
            },
            "ACTION2": {
                "key": "down",
                "effect": "Move the active basket one cell down within the 3x3 perimeter grid (col + 1). Each successful move is a 45° step around the canvas (direction depends on the starting side).",
                "effect_l2": ''                                                                                                                                                                                                                      ,
                "restrictions": "No-op if the move would land on the center cell (1,1) or outside the 3x3 grid.",
                "restrictions_l2": ''                                                                                                        ,
                "step_cost": 1,
            },
            "ACTION3": {
                "key": "left",
                "effect": "Move the active basket one cell left within the 3x3 perimeter grid (row - 1). Each successful move is a 45° step around the canvas (direction depends on the starting side).",
                "effect_l2": ''                                                                                                                                                                                                                      ,
                "restrictions": "No-op if the move would land on the center cell (1,1) or outside the 3x3 grid.",
                "restrictions_l2": ''                                                                                                        ,
                "step_cost": 1,
            },
            "ACTION4": {
                "key": "right",
                "effect": "Move the active basket one cell right within the 3x3 perimeter grid (row + 1). Each successful move is a 45° step around the canvas (direction depends on the starting side).",
                "effect_l2": ''                                                                                                                                                                                                                      ,
                "restrictions": "No-op if the move would land on the center cell (1,1) or outside the 3x3 grid.",
                "restrictions_l2": ''                                                                                                        ,
                "step_cost": 1,
            },
            "ACTION5": {
                "key": "confirm/dip",
                "effect": "Dip the bucket: animate the active bucket into position and paint a region of the canvas with the selected color.",
                "effect_l2": ''                                                                                                                                                         ,
                "restrictions": "Triggers an animation sequence; cannot act until animation completes.",
                "restrictions_l2": ''                                                                              ,
                "step_cost": 1,
            },
            "ACTION6": {
                "key": "click",
                "effect": "Click a color_button to change the active painting color, or click the crane_arm to trigger a dip animation.",
                "effect_l2": ''                                                                                                                                                               ,
                "restrictions": "Click must land on a valid sprite (color_button or crane_arm).",
                "restrictions_l2": ''                                                                                                            ,
                "step_cost": 1,
            },
        },

        "sprite_roles": {
            "target_pattern":    "The goal image shown in the top-left. Canvas must match this exactly to win.",
            "target_pattern_l2": ''                                                                                                                     ,
            "grid_background":   "L-shaped grid overlay indicating the playfield boundaries.",
            "grid_background_l2":''                                                                                  ,
            "bucket_horizontal": "Straight bucket sprite placed at the active basket position (horizontal/vertical sides).",
            "bucket_horizontal_l2": ''                                                                                                 ,
            "bucket_diagonal":   "Diagonal bucket sprite placed at corner basket positions.",
            "bucket_diagonal_l2":''                                                                        ,
            "crane_arm":         "Crane arm handle. Click (ACTION6) or dip (ACTION5) animates it to paint the canvas.",
            "crane_arm_l2":      ''                                                                                                                                                ,
            "color_button":      "Clickable color picker button. Center pixel color = selectable paint color.",
            "color_button_l2":   ''                                                                                                            ,
            "canvas":            "10x10 painting area. Regions are painted by dipping the bucket.",
            "canvas_l2":         ''                                                                                   ,
            "color_indicator":   "1x5 bar showing the currently selected paint color.",
            "color_indicator_l2":''                                                   ,
        },

        "mechanics": {
            "basket_positions": (
                "8 positions around a circle: top(180°), top-right(0°/diagonal), "
                "right(270°), bottom-right(90°/diagonal), bottom(0°), "
                "bottom-left(180°/diagonal), left(90°), top-left(270°/diagonal). "
                "Mapped via a 3x3 grid (excluding center)."
            ),
            "basket_positions_l2": (
                "8: (180°)()(270°)()"
                "(0°)()(90°)()"
                "3x3"
            ),
            "dip_animation": (
                "When dipping, the active bucket sprite moves toward the canvas center "
                "and back (7 sub-steps). At the farthest point, the corresponding canvas "
                "region is painted with the selected color."
            ),
            "dip_animation_l2": (
                ""
                "7"
            ),
            "crane_dip": (
                "When the crane_arm is clicked or activated by ACTION5, it animates "
                "similarly to a bucket dip, painting a cross-shaped region at the canvas center."
            ),
            "crane_dip_l2": (
                "ACTION5"
                ""
            ),
            "win_check": (
                "After each dip animation completes, the canvas pixels are compared to "
                "the target_pattern pixels (ignoring the diagonal mask positions). "
                "A match advances to the next level."
            ),
            "win_check_l2": (
                ""
                ""
            ),
            "step_limit": "Each level allows 100 steps. Exceeding this causes game over.",
            "step_limit_l2": ''                                                                           ,
        },
    },


    "levels": [
        {
            "level_index": 1,
            "grid_size": [64, 64],
            "target_pattern": "target_pattern_1",
            "colors_available": [WHITE, PURPLE],
            "num_color_buttons": 2,
            "has_crane_arm": False,
            "difficulty": "easy",
            "description": "Two-color split (white top / purple bottom). Use top and bottom bucket positions.",
            "description_l2": ''                                                                            ,
        },
        {
            "level_index": 2,
            "grid_size": [64, 64],
            "target_pattern": "target_pattern_2",
            "colors_available": [WHITE, PURPLE, ORANGE],
            "num_color_buttons": 3,
            "has_crane_arm": False,
            "difficulty": "easy",
            "description": "Three-color diagonal split (purple/orange/white). Use diagonal bucket positions.",
            "description_l2": ''                                                                                  ,
        },
        {
            "level_index": 3,
            "grid_size": [64, 64],
            "target_pattern": "target_pattern_3",
            "colors_available": [WHITE, PURPLE, ORANGE, YELLOW, GREEN, RED, BLUE],
            "num_color_buttons": 7,
            "has_crane_arm": True,
            "difficulty": "medium",
            "description": "Four-color complex pattern (purple/orange/green/red). Crane arm available.",
            "description_l2": ''                                                                                  ,
        },
        {
            "level_index": 4,
            "grid_size": [64, 64],
            "target_pattern": "target_pattern_4",
            "colors_available": [WHITE, PURPLE, ORANGE, YELLOW, GREEN, RED, BLUE],
            "num_color_buttons": 7,
            "has_crane_arm": True,
            "difficulty": "medium",
            "description": "Four-color pattern (blue/orange/yellow/purple). Crane arm available.",
            "description_l2": ''                                                                            ,
        },
        {
            "level_index": 5,
            "grid_size": [64, 64],
            "target_pattern": "target_pattern_5",
            "colors_available": [WHITE, PURPLE, ORANGE, YELLOW, GREEN, RED, BLUE],
            "num_color_buttons": 7,
            "has_crane_arm": True,
            "difficulty": "hard",
            "description": "Four-color complex pattern (green/blue/red/orange). Crane arm available.",
            "description_l2": ''                                                                                  ,
        },
        {
            "level_index": 6,
            "grid_size": [64, 64],
            "target_pattern": "target_pattern_6",
            "colors_available": [WHITE, PURPLE, ORANGE, YELLOW, GREEN, RED, BLUE],
            "num_color_buttons": 7,
            "has_crane_arm": True,
            "difficulty": "hard",
            "description": "Four-color complex pattern (red/white/green/yellow). Crane arm available.",
            "description_l2": ''                                                                                  ,
        },
    ],
}





_BASKET_POS_DESC_EN = {
    0: "the top",
    1: "the top-right corner",
    2: "the right",
    3: "the bottom-right corner",
    4: "the bottom",
    5: "the bottom-left corner",
    6: "the left",
    7: "the top-left corner",
}
_BASKET_POS_DESC_L2 = {
    0: "",
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: "",
    7: "",
}



_CANVAS_PAINT_REGION_EN = {
    (0, False): "top half",
    (1, False): "top-right triangle",
    (2, False): "right half",
    (3, False): "bottom-right triangle",
    (4, False): "bottom half",
    (5, False): "bottom-left triangle",
    (6, False): "left half",
    (7, False): "top-left triangle",
    (0, True):  "top-center small region",
    (2, True):  "right-center small region",
    (4, True):  "bottom-center small region",
    (6, True):  "left-center small region",
}
_CANVAS_PAINT_REGION_L2 = {
    (0, False): "",
    (1, False): "",
    (2, False): "",
    (3, False): "",
    (4, False): "",
    (5, False): "",
    (6, False): "",
    (7, False): "",
    (0, True):  "",
    (2, True):  "",
    (4, True):  "",
    (6, True):  "",
}




class Cd82(ARCBaseGame):
    """



















    """

    def __init__(self) -> None:






        self.basket_positions = [
            ("horizontal", 25, 24, 180, 0, 1),
            ("diagonal",   33, 21,   0, -1, 1),
            ("horizontal", 38, 32, 270, -1, 0),
            ("diagonal",   33, 40,  90, -1, -1),
            ("horizontal", 25, 45,   0,  0, -1),
            ("diagonal",   14, 40, 180,  1, -1),
            ("horizontal", 17, 32,  90,  1,  0),
            ("diagonal",   14, 21, 270,  1,  1),
        ]


        self.grid_pos_to_idx = {
            0: (0, 1),
            1: (0, 2),
            2: (1, 2),
            3: (2, 2),
            4: (2, 1),
            5: (2, 0),
            6: (1, 0),
            7: (0, 0),
        }

        self.idx_to_grid_pos = {v: k for k, v in self.grid_pos_to_idx.items()}


        self.active_basket_idx: int = 0
        self.selected_color: int = PURPLE
        self.is_dipping: bool = False
        self.dip_step: int = 0
        self.dip_direction: int = 1
        self.dip_painted: bool = False
        self.bucket_start_pos: tuple = (0, 0)
        self.has_crane_arm: bool = False
        self.max_steps: int = 100
        self.progress_bar = StepProgressBar(max_steps=self.max_steps)
        self.step_move_count: int = 0
        self.click_used: bool = False




        self._log_state = ActionLogState(
            aggregator   = default_aggregator,
            post_process = self._compose_summary,
        )
        self._frame_sprite_log:      list[list[dict]] = []
        self._frame_nonsprite_log:   list[list[dict]] = []


        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.progress_bar],
        )
        super().__init__(
            game_id="cd82",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5, 6],
        )

    def on_set_level(self, level: Level) -> None:
        """"""
        self.active_basket_idx = 0
        self.selected_color = PURPLE
        self.is_dipping = False
        self.dip_step = 0
        self.dip_direction = 1
        self.dip_painted = False
        self.bucket_start_pos = (0, 0)
        self.click_used = False

        self.has_crane_arm = any(s.name == "crane_arm" for s in level.get_sprites())
        self._rebuild_basket_sprites()
        self.step_move_count = 0
        self.click_used = False
        self.progress_bar.set_steps(self.max_steps)







        _LAYOUT_NAME_RULES: list[tuple[str, str, str]] = [

            ("target_pattern",  "target",           ""),
            ("color_button",    "color_button",     ""),
            ("color_indicator", "color_indicator",  ""),
            ("canvas",          "canvas",           ""),
            ("ActiveBasket",    "selected_bucket",  ""),
            ("crane_arm",       "crane_arm",        ""),
        ]
        _layout_rows: list[dict] = []
        for _spr in self.current_level.get_sprites():
            if not _spr.is_visible:
                continue
            _match = next(
                ((r, rj) for pref, r, rj in _LAYOUT_NAME_RULES
                 if _spr.name.startswith(pref)),
                None,
            )
            if _match is None:
                continue
            _role, _role_l2 = _match
            _dx, _dy, _dw, _dh = self._to_display_bbox(_spr)
            _layout_rows.append({
                "role":    _role,
                "role_l2": _role_l2,
                "sname":   _spr.name,
                "dx":      _dx, "dy": _dy,
                "dw":      _dw, "dh": _dh,
            })


        try:
            _ns_status = self.get_nonsprite_status() or []
        except Exception:
            _ns_status = []
        for _ns in _ns_status:
            if _ns.get("name") != "step_budget":
                continue
            _layout_rows.append({
                "role":    "steps",
                "role_l2": ''               ,
                "sname":   "step_budget",
                "dx":      int(_ns.get("x", 0)),
                "dy":      int(_ns.get("y", 0)),
                "dw":      int(_ns.get("width", 0)),
                "dh":      int(_ns.get("height", 0)),
            })

        if _layout_rows:
            self._set_event("initial_layout", _layout_rows)

    def _rebuild_basket_sprites(self) -> None:
        """







        """
        level = self.current_level

        stale = [
            s for s in level.get_sprites()
            if s.name.startswith("bucket_")
            or s.name.startswith("ActiveBasket")
            or s.name.startswith("ExampleDiagonal")
            or s.name == "crane_arm"
        ]
        for spr in stale:
            level.remove_sprite(spr)
        self._place_active_bucket()
        self._place_crane_arm()

    def _place_active_bucket(self) -> None:
        """"""
        btype, bx, by, brot, _, _ = self.basket_positions[self.active_basket_idx]
        if btype == "horizontal":
            bucket = sprites["bucket_horizontal"].clone()
        else:
            bucket = sprites["bucket_diagonal"].clone()
        bucket.set_position(bx, by).set_rotation(brot)
        bucket.color_remap(PURPLE, self.selected_color)
        bucket.set_name("ActiveBasket")
        bucket.set_layer(2)
        self.current_level.add_sprite(bucket)

    def _refresh_active_bucket(self) -> None:
        """"""
        existing = self.current_level.get_sprites_by_name("ActiveBasket")
        for spr in existing:
            self.current_level.remove_sprite(spr)
        self._place_active_bucket()
        self._place_crane_arm()

    def _place_crane_arm(self) -> None:
        """
        """
        if not self.has_crane_arm:
            return

        existing = self.current_level.get_sprites_by_name("crane_arm")
        for spr in existing:
            self.current_level.remove_sprite(spr)

        if self.active_basket_idx not in [0, 2, 4, 6]:
            return

        arm_config: dict[int, dict] = {
            0: {"offset": (4, -6), "rotation": 180},
            2: {"offset": (10, 4), "rotation": 270},
            4: {"offset": (4, 10), "rotation": 0},
            6: {"offset": (-6, 4), "rotation": 90},
        }
        cfg = arm_config[self.active_basket_idx]
        _, bx, by, _, _, _ = self.basket_positions[self.active_basket_idx]
        arm = sprites["crane_arm"].clone()
        rotation = cast(int, cfg["rotation"])
        offset = cast(tuple[int, int], cfg["offset"])
        arm.set_rotation(rotation)
        arm.set_position(bx + offset[0], by + offset[1])
        arm.color_remap(PURPLE, self.selected_color)
        self.current_level.add_sprite(arm)

    def _get_crane_arm_at(self, basket_idx: int) -> Sprite | None:
        """"""
        rotation_map = {0: 180, 2: 270, 4: 0, 6: 90}
        if basket_idx not in rotation_map:
            return None
        target_rot = rotation_map[basket_idx]
        for spr in self.current_level.get_sprites():
            if spr.name == "crane_arm" and spr.rotation == target_rot:
                return spr
        return None

    def _navigate_basket(self, direction: int) -> None:
        """


























        """
        col, row = self.grid_pos_to_idx[self.active_basket_idx]
        if direction == 1:
            new_col, new_row = (max(0, col - 1), row)
        elif direction == 2:
            new_col, new_row = (min(2, col + 1), row)
        elif direction == 3:
            new_col, new_row = (col, max(0, row - 1))
        elif direction == 4:
            new_col, new_row = (col, min(2, row + 1))
        else:
            return

        if (new_col, new_row) == (1, 1):
            return
        new_idx = self.idx_to_grid_pos.get((new_col, new_row))
        if new_idx is not None and new_idx != self.active_basket_idx:
            self.active_basket_idx = new_idx
            self._refresh_active_bucket()

    def _handle_click(self, screen_x: int, screen_y: int) -> None:
        """"""
        grid_pos = self.camera.display_to_grid(screen_x, screen_y)
        if grid_pos is None:
            return
        gx, gy = grid_pos
        hit = self.current_level.get_sprite_at(gx, gy, ignore_collidable=True)
        if hit is None:
            return
        if hit.name == "crane_arm":

            arm_rotation_to_idx = {180: 0, 270: 2, 0: 4, 90: 6}
            arm_idx = arm_rotation_to_idx.get(hit.rotation)
            if arm_idx is not None:
                self._start_dip(arm_idx)
            return
        if hit.name.startswith("color_button"):

            center_color = int(hit.pixels[2, 2])
            self.selected_color = center_color

            indicators = self.current_level.get_sprites_by_name("color_indicator")
            if indicators:
                indicators[0].set_position(hit.x, hit.y + 5)
            self._refresh_active_bucket()

    def _start_dip(self, basket_idx: int) -> None:
        """"""
        arm = self._get_crane_arm_at(basket_idx)
        if not arm:
            return
        self.is_dipping = True
        self.dip_step = 0
        self.dip_direction = 1
        self.bucket_start_pos = (arm.x, arm.y)
        self.dip_painted = False

        self.dipping_sprite = arm
        self.dipping_basket_idx = basket_idx

    def _paint_canvas_crane(self, basket_idx: int) -> None:
        """


        """
        canvas_sprites = self.current_level.get_sprites_by_name("canvas")
        if not canvas_sprites:
            return
        canvas = canvas_sprites[0]
        old_pixels = canvas.pixels.copy()
        pixels = canvas.pixels.copy()
        if basket_idx == 0:
            pixels[0:3, 3:7] = self.selected_color
        elif basket_idx == 4:
            pixels[7:10, 3:7] = self.selected_color
        elif basket_idx == 6:
            pixels[3:7, 0:3] = self.selected_color
        elif basket_idx == 2:
            pixels[3:7, 7:10] = self.selected_color
        canvas.pixels = pixels

        if not np.array_equal(old_pixels, pixels):
            self._log_canvas_paint(basket_idx, is_crane=True)

    def _log_canvas_paint(self, basket_idx: int, is_crane: bool) -> None:
        """"""
        c_label    = COLOR_PALETTE.get(self.selected_color, {}).get("label",    str(self.selected_color)).lower()
        c_label_l2 = COLOR_PALETTE.get(self.selected_color, {}).get("label_l2", str(self.selected_color))
        region    = _CANVAS_PAINT_REGION_EN.get((basket_idx, is_crane), "a region")
        region_l2 = _CANVAS_PAINT_REGION_L2.get((basket_idx, is_crane), "")

        canvas_sprites = self.current_level.get_sprites_by_name("canvas")
        if canvas_sprites:
            cdx, cdy, cdw, cdh = self._to_display_bbox(canvas_sprites[0])
        else:
            cdx, cdy = 0, 0
        self.action_log.append(
            f"[paint]:[canvas]:[sname(canvas);position({cdx}_{cdy});bbox({cdw}_{cdh});color({c_label})]:"
            f"The {region} of the canvas at ({cdx},{cdy}) is repainted {c_label}."
            f"|The {region} of the Object at ({cdx},{cdy}) is repainted {c_label}."
        )
        self.action_log_l2.append(
            f"[paint]:[canvas]:[sname(canvas);position({cdx}_{cdy});bbox({cdw}_{cdh});color({c_label})]:"
            f"({cdx},{cdy}){region_l2}{c_label_l2}"
            f"|({cdx},{cdy})Object{region_l2}{c_label_l2}"
        )

    def step(self) -> None:
        """









        """

        self.action_log    = []
        self.action_log_l2 = []




        _before_sprites = {
            spr.name: {
                "x": spr.x, "y": spr.y,
                "width": spr.width, "height": spr.height,
                "rotation": spr.rotation,
                "visible": spr.is_visible,
            }
            for spr in self.current_level.get_sprites()
        }
        _before_steps = self.progress_bar.current_steps
        _before_color = self.selected_color
        _before_idx   = self.active_basket_idx






        self.progress_bar.set_steps(self.max_steps - self._action_count)


        if self._action_count >= self.max_steps:
            self.lose()
            self.complete_action()
            self._set_event("gameover", {
                "en": "[gameover]:[game]:[]:The step budget is exhausted, game over.|Game over.",
                "l2": "[gameover]:[game]:[]:|",
            })
            if hasattr(self, "_frame_sprite_log"):
                self._frame_sprite_log.append(self.get_sprite_status())
            if hasattr(self, "_frame_nonsprite_log"):
                self._frame_nonsprite_log.append(self.get_nonsprite_status())
            self._log_state.snapshot_step()
            return


        if self.is_dipping:
            self._tick_dip_animation()

            self._detect_and_log_diffs(_before_sprites, _before_steps, _before_color, _before_idx)
            if hasattr(self, "_frame_sprite_log"):
                self._frame_sprite_log.append(self.get_sprite_status())
            if hasattr(self, "_frame_nonsprite_log"):
                self._frame_nonsprite_log.append(self.get_nonsprite_status())
            self._log_state.snapshot_step()
            return


        action = self.action.id
        if action == GameAction.ACTION1:
            self._navigate_basket(1)
            self.step_move_count += 1
            self.complete_action()
        elif action == GameAction.ACTION2:
            self._navigate_basket(2)
            self.step_move_count += 1
            self.complete_action()
        elif action == GameAction.ACTION3:
            self._navigate_basket(3)
            self.step_move_count += 1
            self.complete_action()
        elif action == GameAction.ACTION4:
            self._navigate_basket(4)
            self.step_move_count += 1
            self.complete_action()
        elif action == GameAction.ACTION5:

            self._start_dip_from_bucket()
            self.step_move_count = 0
            self.click_used = False
        elif action == GameAction.ACTION6:
            data = self.action.data
            if "x" in data and "y" in data:
                self._handle_click(data["x"], data["y"])
                self.click_used = True
            if not self.is_dipping:
                self.complete_action()
        else:
            self.complete_action()




        self._detect_and_log_diffs(_before_sprites, _before_steps, _before_color, _before_idx)

        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()

    def _start_dip_from_bucket(self) -> None:
        """"""
        btype, bx, by, brot, dx, dy = self.basket_positions[self.active_basket_idx]

        bucket_spr = None
        for spr in self.current_level.get_sprites():
            if spr.name == "ActiveBasket":
                bucket_spr = spr
                break
        if not bucket_spr:
            self.complete_action()
            return
        self.is_dipping = True
        self.dip_step = 0
        self.dip_direction = 1
        self.bucket_start_pos = (bucket_spr.x, bucket_spr.y)
        self.dip_painted = False
        self.dipping_sprite = bucket_spr
        self.dipping_basket_idx = self.active_basket_idx
        self._dip_dx = dx
        self._dip_dy = dy

    def _tick_dip_animation(self) -> None:
        """"""
        if not hasattr(self, "dipping_sprite") or self.dipping_sprite is None:
            self.is_dipping = False
            self.complete_action()
            return


        if hasattr(self, "_dip_dx"):
            dx, dy = self._dip_dx, self._dip_dy
        else:
            arm_dir_map = {0: (0, 2), 2: (-2, 0), 4: (0, -2), 6: (2, 0)}
            dx, dy = arm_dir_map.get(self.dipping_basket_idx, (0, 0))

        if self.dip_direction == 1:

            self.dipping_sprite.move(dx, dy)
            self.dip_step += 1
            if self.dip_step >= 7:
                if not self.dip_painted:


                    if hasattr(self, "_dip_dx"):
                        self._paint_canvas(self.dipping_basket_idx)
                    else:
                        self._paint_canvas_crane(self.dipping_basket_idx)
                    self.dip_painted = True
                self.dip_direction = -1
        else:

            self.dipping_sprite.move(-dx, -dy)
            self.dip_step -= 1
            if self.dip_step <= 0:
                self.dipping_sprite.set_position(*self.bucket_start_pos)
                self.dipping_sprite = None
                self.is_dipping = False
                if hasattr(self, "_dip_dx"):
                    del self._dip_dx
                    del self._dip_dy
                self._check_win_condition()
                self.complete_action()

    def _paint_canvas(self, basket_idx: int) -> None:
        """







        """
        canvas_sprites = self.current_level.get_sprites_by_name("canvas")
        if not canvas_sprites:
            return
        canvas = canvas_sprites[0]
        old_pixels = canvas.pixels.copy()
        pixels = canvas.pixels.copy()
        btype = self.basket_positions[basket_idx][0]
        _, _, _, rot, _, _ = self.basket_positions[basket_idx]
        if btype == "horizontal":
            if rot == 180:
                pixels[0:5, :] = self.selected_color
            elif rot == 0:
                pixels[5:10, :] = self.selected_color
            elif rot == 90:
                pixels[:, 0:5] = self.selected_color
            elif rot == 270:
                pixels[:, 5:10] = self.selected_color
        else:

            if rot == 180:
                for i in range(10):
                    pixels[i, 0 : i + 1] = self.selected_color
            elif rot == 90:
                for i in range(10):
                    pixels[i, 9 - i : 10] = self.selected_color
            elif rot == 0:
                for i in range(10):
                    pixels[i, i:10] = self.selected_color
            elif rot == 270:
                for i in range(10):
                    pixels[i, 0 : 10 - i] = self.selected_color
        canvas.pixels = pixels

        if not np.array_equal(old_pixels, pixels):
            self._log_canvas_paint(basket_idx, is_crane=False)

    def _check_win_condition(self) -> None:
        """"""
        canvas_sprites = self.current_level.get_sprites_by_name("canvas")

        target_sprites = [
            s for s in self.current_level.get_sprites()
            if s.name.startswith("target_pattern_")
        ]
        if not canvas_sprites or not target_sprites:
            return
        canvas_pix = canvas_sprites[0].pixels
        target_pix = target_sprites[0].pixels

        mask = np.ones((10, 10), dtype=bool)
        for i in range(10):
            mask[i, i] = False
            mask[i, 9 - i] = False
        if np.array_equal(canvas_pix[mask], target_pix[mask]):
            self._set_event("win", {
                "en": "[win]:[game]:[]:The canvas matches the target pattern, the level is cleared.|The level is cleared.",
                "l2": "[win]:[game]:[]:|",
            })
            self.next_level()

    def _to_display_xy(self, x: int, y: int) -> tuple[int, int]:
        """



        """
        cam = self.camera
        arc_scale = min(64 // cam.width, 64 // cam.height)
        x_off = (64 - cam.width  * arc_scale) // 2
        y_off = (64 - cam.height * arc_scale) // 2
        return (x - cam.x) * arc_scale + x_off, (y - cam.y) * arc_scale + y_off


    def _to_display_bbox(self, spr) -> tuple[int, int, int, int]:
        """

        """
        cam = self.camera
        arc_scale = min(64 // cam.width, 64 // cam.height)
        x_off = (64 - cam.width  * arc_scale) // 2
        y_off = (64 - cam.height * arc_scale) // 2
        dx = (spr.x - cam.x) * arc_scale + x_off
        dy = (spr.y - cam.y) * arc_scale + y_off
        dw = spr.width  * arc_scale
        dh = spr.height * arc_scale
        return dx, dy, dw, dh

    def _bar_pixel_len(self, value: int) -> int:
        """



        """
        width = 64
        max_steps = self.progress_bar.max_steps
        if max_steps == 0:
            return 0
        ratio = value / max_steps
        filled = round(width * ratio)
        return min(filled, width)

    def _detect_and_log_diffs(
        self,
        before_sprites: dict,
        before_steps: int,
        before_color: int,
        before_idx: int,
    ) -> None:
        """





        """
        after_sprites = {spr.name: spr for spr in self.current_level.get_sprites()}


        for name, spr in after_sprites.items():
            b = before_sprites.get(name)
            if b is None or (not b["visible"] and spr.is_visible):
                dx, dy, dw, dh = self._to_display_bbox(spr)
                self.action_log.append(
                    f"[appear]:[{name}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh})]:"
                    f"{name} appears at ({dx},{dy})."
                    f"|Object appears at ({dx},{dy})."
                )
                self.action_log_l2.append(
                    f"[appear]:[{name}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh})]:"
                    f"{name}({dx},{dy})"
                    f"|Object({dx},{dy})"
                )



        cam = self.camera
        _arc_scale = min(64 // cam.width, 64 // cam.height)
        for name, b in before_sprites.items():
            spr = after_sprites.get(name)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = b["width"]  * _arc_scale
                bdh = b["height"] * _arc_scale
                self.action_log.append(
                    f"[disappear]:[{name}]:[sname({name});position({bdx}_{bdy});bbox({bdw}_{bdh})]:"
                    f"{name} disappears from ({bdx},{bdy})."
                    f"|Object disappears from ({bdx},{bdy})."
                )
                self.action_log_l2.append(
                    f"[disappear]:[{name}]:[sname({name});position({bdx}_{bdy});bbox({bdw}_{bdh})]:"
                    f"{name}({bdx},{bdy})"
                    f"|Object({bdx},{bdy})"
                )









        nav_happened = (before_idx != self.active_basket_idx)
        for name, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            b = before_sprites.get(name)
            if b is None or not b["visible"]:
                continue
            if name == "ActiveBasket" and nav_happened:
                continue
            if (spr.x, spr.y) != (b["x"], b["y"]):
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                adx, ady, adw, adh = self._to_display_bbox(spr)
                self.action_log.append(
                    f"[move]:[{name}]:[sname({name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:"
                    f"{name} moves from ({bdx},{bdy}) to ({adx},{ady})."
                    f"|Object moves from ({bdx},{bdy}) to ({adx},{ady})."
                )
                self.action_log_l2.append(
                    f"[move]:[{name}]:[sname({name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:"
                    f"{name}({bdx},{bdy})({adx},{ady})"
                    f"|Object({bdx},{bdy})({adx},{ady})"
                )
            if spr.rotation != b["rotation"]:
                _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
                self.action_log.append(
                    f"[rotate]:[{name}]:[sname({name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:"
                    f"{name} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°."
                    f"|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°."
                )
                self.action_log_l2.append(
                    f"[rotate]:[{name}]:[sname({name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:"
                    f"({_rdx},{_rdy}){name}{b['rotation']}°{spr.rotation}°"
                    f"|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°"
                )


        if self.progress_bar.current_steps != before_steps and self._bar_pixel_len(before_steps) != self._bar_pixel_len(self.progress_bar.current_steps):
            _after_steps = self.progress_bar.current_steps
            if _after_steps < before_steps:
                self.action_log.append(
                    f"[change]:[steps]:[sname(step_budget);count({before_steps})_({_after_steps})]:"
                    f"The dark-gray step bar at the bottom shrinks from the right ({before_steps}→{_after_steps} steps remain)."
                    f"|The dark-gray Object at the bottom shrinks from the right ({before_steps}→{_after_steps})."
                )
                self.action_log_l2.append(
                    f"[change]:[steps]:[sname(step_budget);count({before_steps})_({_after_steps})]:"
                    f"{before_steps}→{_after_steps}"
                    f"|Object{before_steps}→{_after_steps}"
                )
            else:
                self.action_log.append(
                    f"[change]:[steps]:[sname(step_budget);count({before_steps})_({_after_steps})]:"
                    f"The dark-gray step bar at the bottom extends to the right ({before_steps}→{_after_steps} steps)."
                    f"|The dark-gray Object at the bottom extends to the right ({before_steps}→{_after_steps})."
                )
                self.action_log_l2.append(
                    f"[change]:[steps]:[sname(step_budget);count({before_steps})_({_after_steps})]:"
                    f"{before_steps}→{_after_steps}"
                    f"|Object{before_steps}→{_after_steps}"
                )
        if self.selected_color != before_color:
            _b_name    = COLOR_PALETTE.get(before_color,      {}).get("label",    str(before_color)).lower()
            _b_name_l2 = COLOR_PALETTE.get(before_color,      {}).get("label_l2", str(before_color))
            _a_name    = COLOR_PALETTE.get(self.selected_color, {}).get("label",    str(self.selected_color)).lower()
            _a_name_l2 = COLOR_PALETTE.get(self.selected_color, {}).get("label_l2", str(self.selected_color))




            self._set_event("selected_color_change", {
                "en": f"[change]:[selected_color]:[color({_b_name})_({_a_name})]:The color indicator bar moves to the {_a_name} button (selected color changes from {_b_name} to {_a_name}).|The Object moves to the {_a_name} Object (color changes from {_b_name} to {_a_name}).",
                "l2": f"[change]:[selected_color]:[color({_b_name})_({_a_name})]:{_a_name_l2}{_b_name_l2}{_a_name_l2}|Object{_a_name_l2}Object{_b_name_l2}{_a_name_l2}",
            })


            for _crname in ("ActiveBasket", "crane_arm"):
                for _spr in self.current_level.get_sprites_by_name(_crname):
                    if not _spr.is_visible:
                        continue
                    _sdx, _sdy, _sdw, _sdh = self._to_display_bbox(_spr)
                    self.action_log.append(
                        f"[change]:[{_spr.name}]:[sname({_spr.name});color({_b_name})_({_a_name});position({_sdx}_{_sdy});bbox({_sdw}_{_sdh})]:"
                        f"{_spr.name} at ({_sdx},{_sdy}) changes color from {_b_name} to {_a_name}."
                        f"|Object at ({_sdx},{_sdy}) changes color from {_b_name} to {_a_name}."
                    )
                    self.action_log_l2.append(
                        f"[change]:[{_spr.name}]:[sname({_spr.name});color({_b_name})_({_a_name});position({_sdx}_{_sdy});bbox({_sdw}_{_sdh})]:"
                        f"({_sdx},{_sdy}){_spr.name}{_b_name_l2}{_a_name_l2}"
                        f"|({_sdx},{_sdy})Object{_b_name_l2}{_a_name_l2}"
                    )
        if self.active_basket_idx != before_idx:





            _delta = (self.active_basket_idx - before_idx) % 8
            _b_gx, _b_gy = self.basket_positions[before_idx][1],          self.basket_positions[before_idx][2]
            _a_gx, _a_gy = self.basket_positions[self.active_basket_idx][1], self.basket_positions[self.active_basket_idx][2]
            _b_dx, _b_dy = self._to_display_xy(_b_gx, _b_gy)
            _a_dx, _a_dy = self._to_display_xy(_a_gx, _a_gy)


            _bsprs = self.current_level.get_sprites_by_name("ActiveBasket")
            _ab_dw, _ab_dh = (0, 0)
            if _bsprs:
                _, _, _ab_dw, _ab_dh = self._to_display_bbox(_bsprs[0])
            _b_desc    = _BASKET_POS_DESC_EN[before_idx]
            _a_desc    = _BASKET_POS_DESC_EN[self.active_basket_idx]
            _b_desc_l2 = _BASKET_POS_DESC_L2[before_idx]
            _a_desc_l2 = _BASKET_POS_DESC_L2[self.active_basket_idx]
            if _delta == 1:
                _en_dir, _l2_dir = "clockwise", ""
            elif _delta == 7:
                _en_dir, _l2_dir = "counter-clockwise", ""
            else:
                _en_dir, _l2_dir = None, None
            if _en_dir is not None:
                self.action_log.append(
                    f"[move]:[active_basket]:[sname(ActiveBasket);position({_b_dx}_{_b_dy})_({_a_dx}_{_a_dy});bbox({_ab_dw}_{_ab_dh})]:"
                    f"The active bucket position rotates 45° {_en_dir} around the canvas from {_b_desc} to {_a_desc}."
                    f"|Object rotates 45° {_en_dir} around the Object from {_b_desc} to {_a_desc}."
                )
                self.action_log_l2.append(
                    f"[move]:[active_basket]:[sname(ActiveBasket);position({_b_dx}_{_b_dy})_({_a_dx}_{_a_dy});bbox({_ab_dw}_{_ab_dh})]:"
                    f"{_l2_dir}45°{_b_desc_l2}{_a_desc_l2}"
                    f"|ObjectObject{_l2_dir}45°{_b_desc_l2}{_a_desc_l2}"
                )
            else:
                self.action_log.append(
                    f"[move]:[active_basket]:[sname(ActiveBasket);position({_b_dx}_{_b_dy})_({_a_dx}_{_a_dy});bbox({_ab_dw}_{_ab_dh})]:"
                    f"The active bucket position moves from {_b_desc} to {_a_desc} around the canvas."
                    f"|Object moves from {_b_desc} to {_a_desc} around the Object."
                )
                self.action_log_l2.append(
                    f"[move]:[active_basket]:[sname(ActiveBasket);position({_b_dx}_{_b_dy})_({_a_dx}_{_a_dy});bbox({_ab_dw}_{_ab_dh})]:"
                    f"{_b_desc_l2}{_a_desc_l2}"
                    f"|ObjectObject{_b_desc_l2}{_a_desc_l2}"
                )

    def _get_valid_actions(self) -> list[ActionInput]:
        """"""
        actions: list[ActionInput] = []


        if self.click_used:
            actions.append(ActionInput(id=GameAction.ACTION5))
            return actions


        if self.step_move_count >= 6:
            actions.extend(self._get_clickable_actions())
            return actions


        if self.step_move_count >= 3:
            for action_id in [1, 2, 3, 4]:
                actions.append(ActionInput(id=GameAction.from_id(action_id)))
            actions.extend(self._get_clickable_actions())
            actions.extend(self._get_crane_click_actions())
            return actions


        for action_id in [1, 2, 3, 4]:
            actions.append(ActionInput(id=GameAction.from_id(action_id)))
        actions.extend(self._get_crane_click_actions())
        return actions

    def _get_clickable_actions(self) -> list[ActionInput]:
        """"""
        result: list[ActionInput] = []
        scale, offset_x, offset_y = self.camera._calculate_scale_and_offset()
        for spr in self.current_level.get_sprites():
            if spr.name.startswith("color_button"):
                dx = (spr.x + 2) * scale + offset_x
                dy = (spr.y + 2) * scale + offset_y
                result.append(ActionInput(id=GameAction.ACTION6, data={"x": dx, "y": dy}))
        return result

    def _get_crane_click_actions(self) -> list[ActionInput]:
        """"""
        result: list[ActionInput] = []
        if self.active_basket_idx not in [0, 2, 4, 6]:
            return result
        arm = self._get_crane_arm_at(self.active_basket_idx)
        if arm:
            scale, offset_x, offset_y = self.camera._calculate_scale_and_offset()
            cx = (arm.x + arm.width // 2) * scale + offset_x
            cy = (arm.y + arm.height // 2) * scale + offset_y
            result.append(ActionInput(id=GameAction.ACTION6, data={"x": cx, "y": cy}))
        return result

    def _get_hidden_state(self) -> np.ndarray:
        """"""
        state = np.zeros((4, 4), dtype=np.int16)
        state[0, 0] = self.progress_bar.current_steps
        return state

    def _set_event(self, key: str, value) -> None:
        """
        """
        self._log_state.set_event(key, value)




    @property
    def action_log(self) -> list[str]:
        return self._log_state._step_log
    @action_log.setter
    def action_log(self, value: list[str]) -> None:
        self._log_state._step_log = value

    @property
    def action_log_l2(self) -> list[str]:
        return self._log_state._step_log_l2
    @action_log_l2.setter
    def action_log_l2(self, value: list[str]) -> None:
        self._log_state._step_log_l2 = value

    @property
    def _frame_action_logs(self) -> list[list[str]]:
        return self._log_state._frames
    @_frame_action_logs.setter
    def _frame_action_logs(self, value: list[list[str]]) -> None:
        self._log_state._frames = value

    @property
    def _frame_action_logs_l2(self) -> list[list[str]]:
        return self._log_state._frames_l2
    @_frame_action_logs_l2.setter
    def _frame_action_logs_l2(self, value: list[list[str]]) -> None:
        self._log_state._frames_l2 = value


    def reset_logs(self) -> None:
        self._log_state.reset()

    def get_action_log_frame(self) -> tuple[list[list[str]], list[list[str]]]:
        return self._log_state.get_frame_logs()

    def get_action_log(self) -> tuple[list[str], list[str]]:
        return self._log_state.get_action_log()

    def _compose_summary(
        self,
        agg_en: list[str],
        agg_l2: list[str],
        events: dict,
    ) -> tuple[list[str], list[str]]:
        """


        """
        out_en = list(agg_en)
        out_l2 = list(agg_l2)
        sc_evt = events.get("selected_color_change")
        if isinstance(sc_evt, dict):
            if sc_evt.get("en"):
                out_en.append(sc_evt["en"])
            if sc_evt.get("l2"):
                out_l2.append(sc_evt["l2"])
        return compose_default_summary(out_en, out_l2, events)

    def get_sprite_status(self) -> list[dict]:
        """



        """
        result = []
        for spr in self.current_level.get_sprites():
            if not spr.is_visible:
                continue
            tags = list(spr._tags) if spr._tags else []


            label = next(
                (TAG_DISPLAY_NAMES[t] for t in tags if t in TAG_DISPLAY_NAMES),
                None,
            )
            label_l2 = next(
                (TAG_DISPLAY_NAMES_L2[t] for t in tags if t in TAG_DISPLAY_NAMES_L2),
                None,
            )
            if label is None or label_l2 is None:
                _en, _l2 = _lookup_sprite_label(spr.name)
                if label is None:
                    label = _en
                if label_l2 is None:
                    label_l2 = _l2
            result.append({
                "spr_id":   id(spr),
                "name":     spr.name,
                "label":    label,
                "label_l2": label_l2,
                "tags":     tags,
                "x":        spr.x,
                "y":        spr.y,
                "width":    spr.width,
                "height":   spr.height,
                "rotation": spr.rotation,
            })
        return result

    def get_nonsprite_status(self) -> list[dict]:
        """

        """
        result = []

        result.append({
            "name":     "step_budget",
            "label":    "Steps Remaining",
            "label_l2": ''                     ,
            "value":    self.progress_bar.current_steps,
            "max":      self.progress_bar.max_steps,
            "x":        0, "y":      63,
            "width":    64, "height": 1,
        })

        result.append({
            "name":     "selected_color",
            "label":    "Selected Color",
            "label_l2": ''               ,
            "value":    self.selected_color,
            "max":      15,
            "x":        0, "y":      0,
            "width":    1, "height": 1,
        })

        result.append({
            "name":     "active_basket_idx",
            "label":    "Active Basket Position",
            "label_l2": ''                                    ,
            "value":    self.active_basket_idx,
            "max":      7,
            "x":        0, "y":      0,
            "width":    1, "height": 1,
        })
        return result

    def get_game_knowledge(self) -> dict:
        """

        """
        return GAME_KNOWLEDGE
