"""



















































"""

from typing import Any

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

sprites = {
    "0": Sprite(
        pixels=[
            [10],
        ],
        name="0",
        visible=True,
        collidable=True,
        tags=["0", "fruit"],
        layer=10,
    ),
    "1": Sprite(
        pixels=[
            [6, 6],
            [6, 6],
        ],
        name="1",
        visible=True,
        collidable=True,
        tags=["fruit", "1"],
        layer=9,
    ),
    "2": Sprite(
        pixels=[
            [15, 15, 15],
            [15, 15, 15],
            [15, 15, 15],
        ],
        name="2",
        visible=True,
        collidable=True,
        tags=["2", "fruit"],
    ),
    "3": Sprite(
        pixels=[
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
        ],
        name="3",
        visible=True,
        collidable=True,
        tags=["3", "fruit"],
    ),
    "4": Sprite(
        pixels=[
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
        ],
        name="4",
        visible=True,
        collidable=True,
        tags=["fruit", "4"],
    ),
    "5": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
        ],
        name="5",
        visible=True,
        collidable=True,
        tags=["fruit", "5"],
    ),
    "6": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="6",
        visible=True,
        collidable=True,
        tags=["fruit", "6"],
    ),
    "7": Sprite(
        pixels=[
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
        ],
        name="7",
        visible=True,
        collidable=True,
        tags=["7", "fruit"],
    ),
    "8": Sprite(
        pixels=[
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
            [14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
        ],
        name="8",
        visible=True,
        collidable=True,
        tags=["8", "fruit"],
        layer=2,
    ),
    "goal_9x9": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
        ],
        name="goal_9x9",
        visible=True,
        collidable=True,
        tags=["goal"],
        layer=-3,
    ),
    "scoreboard_4x12": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 7, 7, 5, 5, 14, 14, 5, 5, 13, 13, 5],
            [5, 7, 7, 5, 5, 14, 14, 5, 5, 13, 13, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        name="scoreboard_4x12",
        visible=True,
        collidable=True,
    ),
    "key_white_3x3": Sprite(
        pixels=[
            [15, 15, 15],
            [15, 15, 15],
            [15, 15, 15],
        ],
        name="key_white_3x3",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "enemy": Sprite(
        pixels=[
            [-2, -2, 7, -2, -2],
            [-2, 7, -2, 7, -2],
            [7, -2, -2, -2, 7],
            [-2, 7, 7, 7, -2],
        ],
        name="enemy",
        visible=True,
        collidable=True,
        tags=["enemy"],
        layer=-1,
    ),
    "enemy2": Sprite(
        pixels=[
            [-1, -1, 14, -1, -1],
            [-1, 14, -1, 14, -1],
            [14, -1, 14, -1, 14],
            [-1, 14, -1, 14, -1],
        ],
        name="enemy2",
        visible=True,
        collidable=True,
        tags=["enemy2"],
        layer=-1,
    ),
    "enemy3": Sprite(
        pixels=[
            [13, -2, 13, -2, 13],
            [-2, 13, -2, 13, -2],
            [13, 13, 13, 13, 13],
            [-2, 13, 13, 13, -2],
        ],
        name="enemy3",
        visible=True,
        collidable=True,
        tags=["enemy3"],
        layer=-1,
    ),
    "hint_arrow_4x4": Sprite(
        pixels=[
            [-2, -2, 0, -2],
            [-2, 0, 0, 0],
            [-2, -2, 0, -2],
            [-2, -2, -2, -2],
        ],
        name="hint_arrow_4x4",
        visible=True,
        collidable=True,
        tags=["hint"],
        layer=5,
    ),
    "block_pattern_5x5": Sprite(
        pixels=[
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        name="block_pattern_5x5",
        visible=True,
        collidable=True,
    ),
    "canvas_pixel_1x1": Sprite(
        pixels=[
            [0],
        ],
        name="canvas_pixel_1x1",
        visible=True,
        collidable=True,
    ),
    "bar_blue_4x8": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="bar_blue_4x8",
        visible=True,
        collidable=True,
    ),
    "key_orange_5x5": Sprite(
        pixels=[
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12],
        ],
        name="key_orange_5x5",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "bar_blue_alt_4x8": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="bar_blue_alt_4x8",
        visible=True,
        collidable=True,
    ),
    "frame_4x24": Sprite(
        pixels=[
            [-1, -1, -1, -1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1],
            [4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4],
            [4, 4, 4, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 4, 4, 4],
            [-1, -1, -1, -1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1],
        ],
        name="frame_4x24",
        visible=True,
        collidable=True,
    ),
    "key_enemy_5x5": Sprite(
        pixels=[
            [-2, -2, 7, -2, -2],
            [-2, 7, -2, 7, -2],
            [7, -2, -2, -2, 7],
            [-2, 7, 7, 7, -2],
        ],
        name="key_enemy_5x5",
        visible=True,
        collidable=True,
        tags=["key"],
        layer=-1,
    ),
    "block_magenta_2x2": Sprite(
        pixels=[
            [6, 6],
            [6, 6],
        ],
        name="block_magenta_2x2",
        visible=True,
        collidable=True,
    ),
    "plus_3x4": Sprite(
        pixels=[
            [-1, -1, 10, -1],
            [10, 10, 10, 10],
            [-1, -1, 10, -1],
        ],
        name="plus_3x4",
        visible=True,
        collidable=True,
    ),
    "bar_blue_b_4x8": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="bar_blue_b_4x8",
        visible=True,
        collidable=True,
    ),
    "key_maroon_4x5": Sprite(
        pixels=[
            [13, -2, 13, -2, 13],
            [-2, 13, -2, 13, -2],
            [13, 13, 13, 13, 13],
            [-2, 13, 13, 13, -2],
        ],
        name="key_maroon_4x5",
        visible=True,
        collidable=True,
        tags=["key"],
        layer=-1,
    ),
    "bar_blue_wide_4x24": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="bar_blue_wide_4x24",
        visible=True,
        collidable=True,
    ),
    "bar_blue_c_4x8": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="bar_blue_c_4x8",
        visible=True,
        collidable=True,
    ),
    "key_yellow_4x4": Sprite(
        pixels=[
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
            [11, 11, 11, 11],
        ],
        name="key_yellow_4x4",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "goal_9x9_b": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, 9, -1, -1],
        ],
        name="goal_9x9_b",
        visible=True,
        collidable=True,
        tags=["goal"],
        layer=-3,
    ),
    "floor_10x64": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="floor_10x64",
        visible=True,
        collidable=True,
        layer=-2,
    ),
    "key_orange_4x5": Sprite(
        pixels=[
            [-1, -1, 14, -1, -1],
            [-1, 14, -1, 14, -1],
            [14, -1, 14, -1, 14],
            [-1, 14, -1, 14, -1],
        ],
        name="key_orange_4x5",
        visible=True,
        collidable=True,
        tags=["key"],
    ),
    "block_offwhite_5x5": Sprite(
        pixels=[
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        name="block_offwhite_5x5",
        visible=True,
        collidable=True,
    ),
    "diagonal_trace_43x43": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        name="diagonal_trace_43x43",
        visible=True,
        collidable=True,
        layer=-2,
    ),
    "color_strip_1x11": Sprite(
        pixels=[
            [10, -1, 6, -1, 15, -1, 11, -1, 12, -1, 8],
        ],
        name="color_strip_1x11",
        visible=True,
        collidable=True,
    ),
    "goal_9x18": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, -1, -1],
        ],
        name="goal_9x18",
        visible=True,
        collidable=True,
        tags=["goal"],
        layer=-2,
    ),
    "block_gray_3x3": Sprite(
        pixels=[
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        name="block_gray_3x3",
        visible=True,
        collidable=True,
        layer=-1,
    ),
    "bar_blue_row_4x36": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="bar_blue_row_4x36",
        visible=True,
        collidable=True,
        layer=-2,
    ),
    "color_bar_1x9": Sprite(
        pixels=[
            [10, 6, 15, 11, 12, 8, 9, 7, 14],
        ],
        name="color_bar_1x9",
        visible=True,
        collidable=True,
    ),
    "hint_diamond_11x11": Sprite(
        pixels=[
            [-1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1],
            [-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1],
            [-1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [-1, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1],
            [-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1],
            [-1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1],
        ],
        name="hint_diamond_11x11",
        visible=True,
        collidable=True,
        tags=["hint"],
    ),
    "goal_8x8": Sprite(
        pixels=[
            [-1, -1, 9, 9, 9, 9, -1, -1],
            [-1, 9, 9, 9, 9, 9, 9, -1],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [-1, 9, 9, 9, 9, 9, 9, -1],
            [-1, -1, 9, 9, 9, 9, -1, -1],
        ],
        name="goal_8x8",
        visible=True,
        collidable=True,
        layer=-1,
    ),
}
levels = [

    Level(
        sprites=[
            sprites["2"].clone().set_position(3, 58),
            sprites["goal_9x9"].clone().set_position(44, 11),
            sprites["key_white_3x3"].clone().set_position(30, 4),
            sprites["hint_arrow_4x4"].clone().set_position(8, 52),
            sprites["bar_blue_wide_4x24"].clone(),
            sprites["floor_10x64"].clone(),
            sprites["diagonal_trace_43x43"].clone().set_position(6, 15),
            sprites["block_gray_3x3"].clone().set_position(47, 14),
        ],
        grid_size=(64, 64),
        data={
            "goal": [2, 1],
            "steps": 32,
        },
    ),

    Level(
        sprites=[
            sprites["0"].clone().set_position(41, 37),
            sprites["0"].clone().set_position(18, 37),
            sprites["0"].clone().set_position(37, 40),
            sprites["0"].clone().set_position(16, 41),
            sprites["0"].clone().set_position(14, 55),
            sprites["0"].clone().set_position(16, 57),
            sprites["0"].clone().set_position(49, 54),
            sprites["0"].clone().set_position(47, 56),
            sprites["goal_9x9"].clone().set_position(29, 23),
            sprites["bar_blue_b_4x8"].clone().set_position(16, 0),
            sprites["key_yellow_4x4"].clone().set_position(30, 3),
            sprites["floor_10x64"].clone(),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 1],
            "steps": 32,
        },
    ),

    Level(
        sprites=[
            sprites["0"].clone().set_position(55, 23),
            sprites["0"].clone().set_position(61, 23),
            sprites["0"].clone().set_position(31, 22),
            sprites["0"].clone().set_position(31, 15),
            sprites["0"].clone().set_position(12, 23),
            sprites["0"].clone().set_position(8, 28),
            sprites["1"].clone().set_position(46, 22),
            sprites["1"].clone().set_position(30, 32),
            sprites["1"].clone().set_position(18, 16),
            sprites["goal_9x9"].clone().set_position(5, 46),
            sprites["goal_9x9"].clone().set_position(19, 46),
            sprites["key_white_3x3"].clone().set_position(36, 4),
            sprites["bar_blue_alt_4x8"].clone().set_position(16, 0),
            sprites["key_yellow_4x4"].clone().set_position(30, 3),
            sprites["floor_10x64"].clone(),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[3, 1], [2, 1]],
            "steps": 48,
        },
    ),

    Level(
        sprites=[
            sprites["0"].clone().set_position(5, 26),
            sprites["0"].clone().set_position(11, 26),
            sprites["0"].clone().set_position(31, 27),
            sprites["0"].clone().set_position(36, 29),
            sprites["0"].clone().set_position(33, 47),
            sprites["0"].clone().set_position(30, 51),
            sprites["0"].clone().set_position(12, 47),
            sprites["0"].clone().set_position(8, 41),
            sprites["goal_9x9"].clone().set_position(1, 53),
            sprites["enemy"].clone().set_position(52, 19),
            sprites["bar_blue_4x8"].clone().set_position(16, 0),
            sprites["key_yellow_4x4"].clone().set_position(30, 3),
            sprites["floor_10x64"].clone(),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 1],
            "steps": 48,
        },
    ),

    Level(
        sprites=[
            sprites["0"].clone().set_position(58, 59),
            sprites["0"].clone().set_position(44, 53),
            sprites["0"].clone().set_position(3, 60),
            sprites["0"].clone().set_position(14, 54),
            sprites["1"].clone().set_position(14, 28),
            sprites["1"].clone().set_position(53, 26),
            sprites["1"].clone().set_position(6, 25),
            sprites["1"].clone().set_position(42, 26),
            sprites["goal_9x9"].clone().set_position(28, 11),
            sprites["enemy"].clone().set_position(4, 37),
            sprites["enemy"].clone().set_position(46, 37),
            sprites["bar_blue_c_4x8"].clone().set_position(16, 0),
            sprites["key_yellow_4x4"].clone().set_position(30, 3),
            sprites["floor_10x64"].clone(),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 1],
            "steps": 32,
        },
    ),

    Level(
        sprites=[
            sprites["5"].clone().set_position(33, 32),
            sprites["goal_9x9"].clone().set_position(2, 12),
            sprites["goal_9x9"].clone().set_position(52, 53),
            sprites["enemy"].clone().set_position(16, 34),
            sprites["key_enemy_5x5"].clone().set_position(36, 3),
            sprites["key_yellow_4x4"].clone().set_position(30, 3),
            sprites["floor_10x64"].clone(),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[3, 1], ["vnjbdkorwc", 1]],
            "steps": 32,
        },
    ),

    Level(
        sprites=[
            sprites["1"].clone().set_position(9, 25),
            sprites["1"].clone().set_position(20, 35),
            sprites["1"].clone().set_position(6, 35),
            sprites["1"].clone().set_position(30, 37),
            sprites["5"].clone().set_position(51, 46),
            sprites["goal_9x9"].clone().set_position(19, 13),
            sprites["goal_9x9"].clone().set_position(40, 18),
            sprites["enemy"].clone().set_position(12, 51),
            sprites["enemy"].clone().set_position(52, 56),
            sprites["key_yellow_4x4"].clone().set_position(30, 3),
            sprites["key_yellow_4x4"].clone().set_position(36, 3),
            sprites["floor_10x64"].clone(),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [3, 2],
            "steps": 32,
        },
    ),

    Level(
        sprites=[
            sprites["3"].clone().set_position(13, 42),
            sprites["3"].clone().set_position(3, 40),
            sprites["5"].clone().set_position(20, 24),
            sprites["goal_9x9"].clone().set_position(52, 15),
            sprites["goal_9x9"].clone().set_position(3, 15),
            sprites["goal_9x9"].clone().set_position(52, 51),
            sprites["goal_9x9"].clone().set_position(3, 51),
            sprites["scoreboard_4x12"].clone().set_position(0, 5),
            sprites["enemy"].clone().set_position(43, 31),
            sprites["enemy"].clone().set_position(29, 53),
            sprites["enemy"].clone().set_position(47, 48),
            sprites["key_orange_5x5"].clone().set_position(30, 2),
            sprites["key_orange_5x5"].clone().set_position(37, 2),
            sprites["floor_10x64"].clone(),
            sprites["key_orange_4x5"].clone().set_position(44, 3),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[4, 2], ["yckgseirmu", 1]],
            "steps": 48,
        },
    ),

    Level(
        sprites=[
            sprites["1"].clone().set_position(18, 46),
            sprites["1"].clone().set_position(23, 52),
            sprites["5"].clone().set_position(35, 48),
            sprites["goal_9x9"].clone().set_position(7, 37),
            sprites["goal_9x9"].clone().set_position(49, 51),
            sprites["goal_9x9"].clone().set_position(7, 51),
            sprites["scoreboard_4x12"].clone().set_position(0, 5),
            sprites["key_white_3x3"].clone().set_position(37, 4),
            sprites["enemy"].clone().set_position(51, 13),
            sprites["enemy"].clone().set_position(14, 12),
            sprites["enemy"].clone().set_position(15, 22),
            sprites["enemy"].clone().set_position(54, 33),
            sprites["key_orange_5x5"].clone().set_position(30, 2),
            sprites["key_maroon_4x5"].clone().set_position(42, 3),
            sprites["floor_10x64"].clone(),
            sprites["color_strip_1x11"].clone().set_position(1, 1).set_scale(2),
        ],
        grid_size=(64, 64),
        data={
            "goal": [[4, 1], ["vptxjilzzk", 1], [2, 1]],
            "steps": 48,
        },
    ),
]


class StepCounterUI(RenderableUserDisplay):
    def __init__(self, game_ref: "Su15", max_steps: int):
        self.max_steps = max_steps
        self.current_steps = max_steps
        self.game_ref = game_ref
        self.penalty: int = 0

    def set_steps(self, value: int) -> None:
        self.current_steps = max(0, min(value, self.max_steps))

    def decrement_step(self) -> bool:
        if self.current_steps > 0:
            self.current_steps -= 1
        return self.current_steps > 0

    def decrement_step_penalty(self) -> bool:
        if self.current_steps > 0:
            self.current_steps -= 2 + self.penalty * 2
            self.penalty += 1
        return self.current_steps > 0

    def reset_steps(self) -> None:
        self.current_steps = self.max_steps
        self.penalty = 0

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        oazbqthsrx: float = self.current_steps / self.max_steps
        start_y: int = 63
        ipapfhrgsf = int(np.ceil(oazbqthsrx * 64))
        frame[start_y, 0:ipapfhrgsf] = WHITE
        return frame


BACKGROUND_COLOR = 5
PADDING_COLOR = 3


TRANSPARENT: int = -1
WHITE:       int =  0
OFF_WHITE:   int =  1
LIGHT_GRAY:  int =  2
GRAY:        int =  3
DARK_GRAY:   int =  4
BLACK:       int =  5
MAGENTA:     int =  6
PINK:        int =  7
RED:         int =  8
BLUE:        int =  9
LIGHT_BLUE:  int = 10
YELLOW:      int = 11
ORANGE:      int = 12
MAROON:      int = 13
GREEN:       int = 14
PURPLE:      int = 15


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
    "fruit":    "Fruit",
    "goal":     "Goal Zone",
    "enemy":    "Enemy (Type A)",
    "enemy2":   "Enemy (Type B, Orange)",
    "enemy3":   "Enemy (Type C, Maroon)",
    "key":      "Key Sprite",
    "hint":     "Hint Marker",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "fruit":    "",
    "goal":     "",
    "enemy":    "A",
    "enemy2":   "B",
    "enemy3":   "C",
    "key":      "",
    "hint":     "",
}


MERGE_COLORS: list[int] = [
    LIGHT_BLUE,
    MAGENTA,
    PURPLE,
    YELLOW,
    ORANGE,
    RED,
    BLUE,
    PINK,
    GREEN,
]

FRUIT_SPRITE_KEYS: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]


TAG_FRUIT: str = "fruit"
TAG_GOAL:  str = "goal"
TAG_NONE:  str = ""


ENEMY_TYPE_A: str = "vnjbdkorwc"
ENEMY_TYPE_B: str = "yckgseirmu"
ENEMY_TYPE_C: str = "vptxjilzzk"

TAG_ENEMY:  str = "enemy"
TAG_ENEMY2: str = "enemy2"
TAG_ENEMY3: str = "enemy3"


VACUUM_STEPS:    int = 8
ENEMY_WAVE_SIZE: int = 4
ENEMY_MOVE_SPEED: int = 4
FLASH_DURATION:  int = 16
GRID_TOP:        int = 10
GRID_BOTTOM:     int = 63




def _pixel_list_to_mask(pixels: list) -> list:
    """"""
    if pixels and isinstance(pixels[0], list):
        return [[0 if v == -1 else 1 for v in row] for row in pixels]
    return [0 if v == -1 else 1 for v in pixels]


def _build_sprite_knowledge(spr: Sprite) -> dict:
    """"""
    pix = spr.pixels.tolist()
    tags = list(spr._tags) if spr._tags else []
    return {
        "pixels":      pix,
        "mask_pixels": _pixel_list_to_mask(pix),
        "width":       int(spr.width),
        "height":      int(spr.height),
        "tags":        tags,
        "layer":       int(getattr(spr, "_layer", 0)),
    }


def _build_level_knowledge(level_index: int, level) -> dict:
    """"""
    grid = getattr(level, "grid_size", (64, 64)) or (64, 64)
    goal_spec = None
    try:
        goal_spec = level.get_data("goal")
    except Exception:
        pass
    steps = None
    try:
        steps = level.get_data("steps")
    except Exception:
        pass

    level_sprites_info: list[dict] = []
    for spr in getattr(level, "_sprites", []) or []:
        tags = list(spr._tags) if spr._tags else []
        level_sprites_info.append({
            "name":   spr.name,
            "x":      spr.x,
            "y":      spr.y,
            "width":  int(spr.width),
            "height": int(spr.height),
            "tags":   tags,
        })

    _level_analyses = [
        {
            "gimmicks":     ["Single fruit on open field, one goal zone"],
            "gimmicks_l2":  [''                                                                          ],
            "subgoals":     ["Click near fruit to vacuum it toward goal zone"],
            "subgoals_l2":  [''                                                                                    ],
            "goal_flow":    "Click to create a vacuum ring near the fruit and attract it into the goal zone.",
            "goal_flow_l2": ''                                                                                                            ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Multiple small fruits, goal requires size-3 fruit"],
            "gimmicks_l2":  [''                                                                  ],
            "subgoals":     ["Merge small fruits into larger ones, then guide to goal"],
            "subgoals_l2":  [''                                                                     ],
            "goal_flow":    "Vacuum multiple same-level fruits together to merge them, increasing their level until goal condition is met.",
            "goal_flow_l2": ''                                                                                                                  ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Multiple fruits of different sizes, multi-goal condition"],
            "gimmicks_l2":  [''                                                         ],
            "subgoals":     ["Selectively merge specific fruit groups to match goal spec"],
            "subgoals_l2":  [''                                                                                             ],
            "goal_flow":    "Plan merges carefully to achieve the multi-goal condition without wasting fruits.",
            "goal_flow_l2": ''                                                                                                                  ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Enemies introduced, fruits need protection"],
            "gimmicks_l2":  [''                                                   ],
            "subgoals":     ["Avoid enemies while merging fruits and guiding to goal"],
            "subgoals_l2":  [''                                                                     ],
            "goal_flow":    "Use vacuum timing to attract fruits away from enemies, merge them, and place in goal zone.",
            "goal_flow_l2": ''                                                                                                                        ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Multiple enemy types, complex merge chain required"],
            "gimmicks_l2":  [''                                                            ],
            "subgoals":     ["Handle enemy interference, perform long merge chains"],
            "subgoals_l2":  [''                                                                        ],
            "goal_flow":    "Systematically merge fruits through multiple levels while managing enemy threats.",
            "goal_flow_l2": ''                                                                                                               ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Enemy in goal zone, goal requires enemy presence"],
            "gimmicks_l2":  [''                                                               ],
            "subgoals":     ["Guide enemy into goal zone alongside specific fruits"],
            "subgoals_l2":  [''                                                                        ],
            "goal_flow":    "Attract and position both enemy and fruit sprites into the goal zone simultaneously.",
            "goal_flow_l2": ''                                                                                                      ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Large fruit plus multiple small fruits, multi-row goal"],
            "gimmicks_l2":  [''                                                               ],
            "subgoals":     ["Build up large fruit through merges, fill multiple goal zones"],
            "subgoals_l2":  [''                                                                           ],
            "goal_flow":    "Merge fruits to achieve the required level, then guide them to fill all goal zones.",
            "goal_flow_l2": ''                                                                                                ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Multiple enemy types and many fruits, complex goal spec"],
            "gimmicks_l2":  [''                                                                        ],
            "subgoals":     ["Precisely manage merges and enemy positions to match goal"],
            "subgoals_l2":  [''                                                                        ],
            "goal_flow":    "Execute a precise sequence of vacuums to merge correct fruits and position enemies per goal spec.",
            "goal_flow_l2": ''                                                                                                                        ,
            "difficulty":   "very_hard",
        },
        {
            "gimmicks":     ["All enemy types, maximum complexity"],
            "gimmicks_l2":  [''                              ],
            "subgoals":     ["Master all game mechanics simultaneously"],
            "subgoals_l2":  [''                                                   ],
            "goal_flow":    "Combine all learned strategies to satisfy the complex multi-condition goal.",
            "goal_flow_l2": ''                                                                                             ,
            "difficulty":   "very_hard",
        },
    ]
    analysis = _level_analyses[level_index] if level_index < len(_level_analyses) else {}
    return {
        "level_index": level_index + 1,
        "grid_size":   list(grid),
        "steps":       steps,
        "goal_spec":   goal_spec,
        "sprites":     level_sprites_info,
        **analysis,
    }



GAME_KNOWLEDGE: dict = {


    "color_palette": {str(k): v for k, v in COLOR_PALETTE.items()},


    "tag_display_names": TAG_DISPLAY_NAMES,


    "tag_display_names_l2": TAG_DISPLAY_NAMES_L2,


    "sprites": {
        name: _build_sprite_knowledge(spr)
        for name, spr in sprites.items()
    },


    "global_knowledge": {
        "game_id":          "su15",
        "title":            "Suika Vacuum",
        "title_l2":         ''                        ,
        "description":      "A puzzle game where you create vacuum rings by clicking to attract and merge fruits. Same-level fruits merge into higher-level ones. Fill the goal zone with the required fruit types to win.",
        "description_l2":   ''                                                                                                                                                                                                                                                         ,
        "default_grid_size": [64, 64],
        "num_levels":        len(levels),

        "goal":             "Place the required number and type of fruits (and/or enemies) in the goal zone by merging fruits and managing enemies.",
        "goal_l2":          ''                                                                                                                                             ,
        "win_condition":    "_check_win() returns True when the goal zone contains the exact fruits/enemies specified in goal_spec.",
        "win_condition_l2": ''                                                                                                                                       ,
        "lose_condition":   "Steps reach 0, or all fruits are eliminated.",
        "lose_condition_l2":''                                                       ,

        "actions": {
            "ACTION6": {
                "key":             "click",
                "effect":          "Create a vacuum ring at the clicked position that attracts nearby fruits and enemies toward it.",
                "effect_l2":       ''                                                                                                      ,
                "restrictions":    "Costs 1 step. Click must be within the playable grid (y=10 to 63).",
                "restrictions_l2": ''                                                                                                    ,
                "step_cost":       1,
            },
            "ACTION7": {
                "key":             "undo",
                "effect":          "Undo the last move, restoring the previous state.",
                "effect_l2":       ''                                                         ,
                "restrictions":    "Costs 2 steps plus a penalty that increases with each undo.",
                "restrictions_l2": ''                                                                                         ,
                "step_cost":       "2 + penalty * 2",
            },
        },

        "sprite_roles": {
            "fruit":          "Collectible circular sprite. Same-level fruits merge when brought together.",
            "fruit_l2":       ''                                                                                                ,
            "goal":           "Goal zone marker. The goal is achieved when the right fruits are inside it.",
            "goal_l2":        ''                                                                                             ,
            "enemy":          "Type A enemy (pink diamond). Moves toward nearest fruit and knocks it back.",
            "enemy_l2":       ''                                                                                                             ,
            "enemy2":         "Type B enemy (orange). Moves slower than Type A.",
            "enemy2_l2":      ''                                                           ,
            "enemy3":         "Type C enemy (maroon). Moves at speed 2, faster than others.",
            "enemy3_l2":      ''                                                           ,
            "key":            "Key sprite displayed in the UI header area.",
            "key_l2":         ''                                                              ,
            "hint":           "Hint marker shown in level 1 to guide the player.",
            "hint_l2":        ''                                                                   ,
        },

        "mechanics": {
            "vacuum":              "Clicking creates a vacuum ring that shrinks over VACUUM_STEPS frames. Fruits and enemies within range are pulled toward the center.",
            "vacuum_l2":           ''                                                                                                                                                           ,
            "fruit_merging":       "When same-level fruits overlap, they merge into one fruit of the next level. Level 8 fruits disappear when merged.",
            "fruit_merging_l2":    ''                                                                                                                                      ,
            "enemy_ai":            "Enemies chase the nearest non-knockback fruit each frame. On collision, the fruit enters knockback state and bounces.",
            "enemy_ai_l2":         ''                                                                                                                                                                           ,
            "enemy_evolution":     "Same-type enemies that overlap evolve: Type A -> Type B -> Type C.",
            "enemy_evolution_l2":  ''                                                                                     ,
            "knockback":           "Hit fruits shake for knockback_shake_frames, then fly for knockback_fly_frames in the opposite direction.",
            "knockback_l2":        ''                                                                                                                                 ,
            "undo":                "ACTION7 restores the last snapshot. Each undo costs 2 + penalty*2 steps, penalty increases by 1 per undo.",
            "undo_l2":             ''                                                                                                                                           ,
            "step_bar":            "The step progress bar fills the bottom row (y=63). It shrinks as steps are used.",
            "step_bar_l2":         ''                                                                                                 ,
        },
    },


    "levels": [
        _build_level_knowledge(i, lv)
        for i, lv in enumerate(levels)
    ],
}


class Su15(ARCBaseGame):
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

    def _primary_color_name(self, pixels: np.ndarray) -> str:
        """"""
        flat = pixels.flatten()
        visible = flat[flat >= 0]
        if visible.size == 0:
            return "transparent"
        values, counts = np.unique(visible, return_counts=True)
        dominant = int(values[np.argmax(counts)])
        palette = COLOR_PALETTE.get(dominant)
        return palette["label"].lower() if palette else f"color{dominant}"

    def _color_l2(self, name: str) -> str:
        """"""
        for entry in COLOR_PALETTE.values():
            if entry["label"].lower() == name:
                return entry["label_l2"]
        return name

    def _step_bar_pixel_len(self, steps: int) -> int:
        """




        """
        max_steps = self.step_counter_ui.max_steps
        if max_steps <= 0:
            return 0
        return int(np.ceil(steps / max_steps * 64))

    def __init__(self) -> None:
        self.step_counter_ui = StepCounterUI(self, 128)
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.step_counter_ui],
        )
        self.grid_width: int = 64
        self.grid_height: int = 64
        self.fruits: list[Sprite] = []
        self.enemies: list[Sprite] = []
        self.goals: list[Sprite] = []
        self.enemy_type_map: dict[Sprite, str] = {}
        self.fruit_level_map: dict[Sprite, int] = {}
        self.fruit_positions: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        self.fruit_levels_arr: np.ndarray = np.zeros((0,), dtype=np.int64)
        self.vacuum_center: tuple[int, int] = (0, 0)
        self.attracted_fruits: list[Sprite] = []
        self.attracted_enemies: list[Sprite] = []
        self.vacuum_tick: int = 0
        self.vacuum_total_steps: int = ENEMY_WAVE_SIZE
        self.vacuum_radius: int = VACUUM_STEPS
        self.sprite_move_speed: int = ENEMY_MOVE_SPEED
        self.vacuum_done: bool = False
        self.enemy_dir_map: dict[Sprite, tuple[float, float]] = {}
        self.enemy_pos_map: dict[Sprite, tuple[float, float]] = {}
        self.knockback_set: set[Sprite] = set()
        self.knockback_tick_map: dict[Sprite, int] = {}
        self.knockback_origin_map: dict[Sprite, tuple[float, float]] = {}
        self.knockback_dir_map: dict[Sprite, tuple[float, float]] = {}
        self.knockback_cur_pos_map: dict[Sprite, tuple[float, float]] = {}
        self.knockback_shake_frames: int = 4
        self.knockback_fly_frames: int = 4
        self.knockback_fly_speed: float = 10.0
        self.killed_fruits: set[Sprite] = set()
        self.enemy_cooldown_map: dict[Sprite, int] = {}
        self.fruit_pending_level_map: dict[Sprite, int] = {}
        self.vacuum_ring_sprite: Sprite | None = None
        self.vacuum_radius_cur: float = float(self.vacuum_radius)
        self.vacuum_radius_step: float = float(self.vacuum_radius)
        self.mesh_x: np.ndarray | None = None
        self.mesh_y: np.ndarray | None = None
        self.flash_sprites: list[Sprite] = []
        self.flash_pixels_backup: dict[Sprite, np.ndarray] = {}
        self.flash_tick: int = 0
        self.flash_total_frames: int = FLASH_DURATION
        self.animating: bool = False
        self.anim_state: str = "none"
        self.goal_spec: list[Any] | None = None
        self.state_history: list[list[tuple[str, int, int, int]]] = []



        self._log_state = ActionLogState(aggregator=default_aggregator, post_process=compose_default_summary)


        self._frame_sprite_log:    list[list[dict]] = []
        self._frame_nonsprite_log: list[list[dict]] = []

        super().__init__(game_id="su15", levels=levels, camera=camera, available_actions=[6, 7])

    def _init_step_counter(self) -> None:
        oifmnnjbvl = self.current_level.get_data("steps")
        if oifmnnjbvl:
            self.step_counter_ui.max_steps = oifmnnjbvl
            self.step_counter_ui.reset_steps()

    def on_set_level(self, level: Level) -> None:
        self.actions: list[ActionInput] = []
        for i in range(16):
            for ituakcbkov in range(14):
                y = 10 + ituakcbkov * 4
                self.actions.append(ActionInput(id=GameAction.ACTION6.value, data={"x": i * 4, "y": y}))
        self._init_step_counter()
        oiasidva = self.current_level.grid_size
        if oiasidva is not None:
            self.grid_width = oiasidva[0]
            self.grid_height = oiasidva[1]
        self.goal_spec = self.current_level.get_data("goal")
        self.key_sprites = self.current_level.get_sprites_by_tag("key")
        self.hint_sprite: Sprite | None = None
        sgxxybqgsv = self.current_level.get_sprites_by_tag("hint")
        if self.level_index == 0:
            self.hint_sprite = sgxxybqgsv[0]
        self.grayed = False
        self.fruits = self.current_level.get_sprites_by_tag(TAG_FRUIT)
        lrxgzrrgul: list[Sprite] = self.current_level.get_sprites_by_tag(TAG_ENEMY)
        dflvyelvmq: list[Sprite] = self.current_level.get_sprites_by_tag(TAG_ENEMY2)
        eiqlefvkeq: list[Sprite] = self.current_level.get_sprites_by_tag(TAG_ENEMY3)
        self.enemies = []
        self.enemies.extend(lrxgzrrgul)
        self.enemies.extend(dflvyelvmq)
        self.enemies.extend(eiqlefvkeq)
        self.goals = self.current_level.get_sprites_by_tag(TAG_GOAL)
        self.enemy_type_map = {}
        for vxxrprjnav in lrxgzrrgul:
            self.enemy_type_map[vxxrprjnav] = ENEMY_TYPE_A
        for vxxrprjnav in dflvyelvmq:
            self.enemy_type_map[vxxrprjnav] = ENEMY_TYPE_B
        for vxxrprjnav in eiqlefvkeq:
            self.enemy_type_map[vxxrprjnav] = ENEMY_TYPE_C
        self.fruit_level_map.clear()
        for sprite in self.fruits:
            kcacvefoot = self._get_fruit_level(sprite)
            self.fruit_level_map[sprite] = kcacvefoot
        self._update_fruit_positions_cache()
        self.attracted_fruits = []
        self.attracted_enemies = []
        self.enemy_dir_map = {}
        self.enemy_pos_map = {}
        self.vacuum_center = (0, 0)
        self.vacuum_tick = 0
        self.vacuum_radius_cur = float(self.vacuum_radius)
        if self.vacuum_total_steps > 1:
            self.vacuum_radius_step = float(self.vacuum_radius) / float(self.vacuum_total_steps - 1)
        else:
            self.vacuum_radius_step = float(self.vacuum_radius)
        self.knockback_set = set()
        self.knockback_tick_map = {}
        self.knockback_origin_map = {}
        self.knockback_dir_map = {}
        self.knockback_cur_pos_map = {}
        self.enemy_cooldown_map = {}
        self.fruit_pending_level_map = {}
        self.killed_fruits = set()
        self.vacuum_ring_sprite = sprites["canvas_pixel_1x1"].clone()
        self.vacuum_ring_sprite.pixels = np.full((64, 64), TRANSPARENT, dtype=np.int64)
        self.current_level.add_sprite(self.vacuum_ring_sprite)
        wskvwndunw = self.vacuum_ring_sprite.pixels
        canvas_height, canvas_width = wskvwndunw.shape
        gwxlhgjbac = np.arange(canvas_height, dtype=np.float32)
        afgzzcccpo = np.arange(canvas_width, dtype=np.float32)
        self.mesh_y, self.mesh_x = np.meshgrid(gwxlhgjbac, afgzzcccpo, indexing="ij")
        self._clear_vacuum_ring()
        self.flash_sprites = []
        self.flash_pixels_backup = {}
        self.flash_tick = 0
        self.animating = False
        self.anim_state = "none"
        self.vacuum_done = False
        self.win_frame = 0
        self.state_history = []
        self._emit_initial_layout()
        self._snapshot_state()


    def _emit_initial_layout(self) -> None:
        """















        """
        _layout_rows: list[dict] = []

        for _tag, _role, _role_l2 in (
            (TAG_FRUIT,  "fruit", ""),
            (TAG_ENEMY,  "enemy", ""),
            (TAG_ENEMY2, "enemy", ""),
            (TAG_ENEMY3, "enemy", ""),
            (TAG_GOAL,   "goal",  ""),
            ("key",      "key",   ""),
            ("hint",     "hint",  ""),
        ):
            for _spr in self.current_level.get_sprites_by_tag(_tag):
                if not _spr.is_visible:
                    continue
                _dx, _dy, _dw, _dh = self._to_display_bbox(_spr)
                _layout_rows.append({
                    "role":    _role,
                    "role_l2": _role_l2,
                    "sname":   _spr.name,
                    "dx":      _dx, "dy": _dy,
                    "dw":      _dw, "dh": _dh,
                })




        for _spr in self.current_level.get_sprites():
            if not _spr.is_visible or not _spr.name.startswith("color_strip"):
                continue
            _dx, _dy, _dw, _dh = self._to_display_bbox(_spr)
            _layout_rows.append({
                "role":    "color_strip",
                "role_l2": ''                        ,
                "sname":   _spr.name,
                "dx":      _dx, "dy": _dy,
                "dw":      _dw, "dh": _dh,
            })


        try:
            _ns_status = self.get_nonsprite_status() or []
        except Exception:
            _ns_status = []
        for _ns in _ns_status:
            if _ns.get("name") != "step_counter":
                continue
            _layout_rows.append({
                "role":    "remaining_step",
                "role_l2": ''                     ,
                "sname":   "step_counter",
                "dx":      int(_ns.get("x", 0)),
                "dy":      int(_ns.get("y", 0)),
                "dw":      int(_ns.get("width", 0)),
                "dh":      int(_ns.get("height", 0)),
            })

        if _layout_rows:
            self._set_event("initial_layout", _layout_rows)

    def step(self) -> None:
        """




        """

        self.action_log    = []
        self.action_log_l2 = []



        _before_sprites: dict[int, dict] = {
            id(spr): {
                "name":     spr.name,
                "visible":  spr.is_visible,
                "x":        spr.x,
                "y":        spr.y,
                "width":    spr.width,
                "height":   spr.height,
                "rotation": spr.rotation,
                "pixels_bytes": spr.pixels.tobytes(),
                "primary_color": self._primary_color_name(spr.pixels),
            }
            for spr in self.current_level.get_sprites()
        }
        _before_steps = self.step_counter_ui.current_steps



        self._before_vacuum_radius_cur = self.vacuum_radius_cur
        self._before_anim_state        = self.anim_state


        if self.animating:
            if self.anim_state == "win":
                index = 0
                for k in self.key_sprites:
                    mpalcyrcyw = k.pixels >= 0
                    k.pixels[mpalcyrcyw] = WHITE
                    index += 1
                for dmanzruhiz in self.goals:
                    mpalcyrcyw = dmanzruhiz.pixels >= 0
                    dmanzruhiz.pixels[mpalcyrcyw] = WHITE
                self.win_frame += 1
                if self.win_frame > 10:
                    self._set_event("win", {
                        "en": "[win]:[game]:[]:The level is cleared, advancing to the next level.|The level is cleared.",
                        "l2": "[win]:[game]:[]:|",
                    })
                    self.next_level()
                    self.complete_action()
                self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                return
            elif self.anim_state == "vacuum":
                kjmljovana = self._move_attracted_sprites()
                if not self.vacuum_done and self.vacuum_tick >= self.vacuum_total_steps:
                    self._process_vacuum_fruits()
                    self.vacuum_done = True
                    if self.anim_state == "flash":
                        self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                        return
                if self._check_win():
                    self._clear_vacuum_ring()
                    self.animating = True
                    self.anim_state = "win"
                    self._set_event("win", {
                        "en": "[win]:[game]:[]:The goal condition is met and the level is cleared.|The level is cleared.",
                        "l2": "[win]:[game]:[]:|",
                    })
                    self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                    return
                if kjmljovana:
                    self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                    return
                self.animating = False
                self.anim_state = "none"
                if not self._has_enough_fruits():
                    self.grayed = True
                    for k in self.key_sprites:
                        mpalcyrcyw = k.pixels >= 0
                        k.pixels[mpalcyrcyw] = LIGHT_GRAY
                    for dmanzruhiz in self.goals:
                        mpalcyrcyw = dmanzruhiz.pixels >= 0
                        dmanzruhiz.pixels[mpalcyrcyw] = LIGHT_GRAY
                if self.level_index == 0 and (self.fruits[0]._x != 3 or self.fruits[0]._y != 58):
                    if self.hint_sprite:
                        self.hint_sprite._x = 500
                if self._check_win():
                    self.animating = True
                    self.anim_state = "win"
                    self._set_event("win", {
                        "en": "[win]:[game]:[]:The goal condition is met and the level is cleared.|The level is cleared.",
                        "l2": "[win]:[game]:[]:|",
                    })
                    self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                    return
                if len(self.current_level.get_sprites_by_tag("fruit")) == 0:
                    self._set_event("gameover", {
                        "en": "[gameover]:[game]:[]:No fruits remain, game over.|Game over.",
                        "l2": "[gameover]:[game]:[]:|",
                    })
                    self.lose()
                if not self.step_counter_ui.decrement_step():
                    self._set_event("gameover", {
                        "en": "[gameover]:[game]:[]:Step counter reached zero, game over.|The Object reached zero, game over.",
                        "l2": "[gameover]:[game]:[]:0|Object0",
                    })
                    self.lose()
                self.complete_action()
                self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                return
            if self.anim_state == "flash":
                if self._tick_flash_animation():
                    self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                    return
                self.anim_state = "none"
                self.animating = False
                self._finish_flash_animation()
                if not self.step_counter_ui.decrement_step_penalty():
                    self._set_event("gameover", {
                        "en": "[gameover]:[game]:[]:Step counter reached zero after a penalty, game over.|The Object reached zero, game over.",
                        "l2": "[gameover]:[game]:[]:0|Object0",
                    })
                    self.lose()
                    self.complete_action()
                    self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                    return
                self._undo_last_move()
                self.complete_action()
                self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                return
            self.animating = False
            self.anim_state = "none"
            self.complete_action()
            self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
            return
        if self.action.id == GameAction.ACTION6:
            wqolgdvhew = self.action.data.get("x", 0)
            vpudlxuvpd = self.action.data.get("y", 0)
            zeyfpujhqd = self.camera.display_to_grid(wqolgdvhew, vpudlxuvpd)
            if zeyfpujhqd is not None:
                grid_x, grid_y = zeyfpujhqd
                self._snapshot_state()
                if self.level_index == 0:
                    rzwdpukxyy = self.current_level.get_sprite_at(grid_x, grid_y, "hint")
                    if rzwdpukxyy is not None:
                        rzwdpukxyy._x = 500
                self._on_click_grid(grid_x, grid_y)



                if self.animating:
                    _cdx, _cdy = self._to_display_xy(grid_x, grid_y)








                    _arc_scale = min(64 // self.camera.width, 64 // self.camera.height)
                    _vac_diam  = (2 * int(self.vacuum_radius + 0.5) + 1) * _arc_scale
                    self.action_log.append(f"[appear]:[vacuum]:[sname(vacuum);position({_cdx}_{_cdy});bbox({_vac_diam}_{_vac_diam})]:A vacuum ring appears centered at ({_cdx},{_cdy}).|An Object appears centered at ({_cdx},{_cdy}).")
                    self.action_log_l2.append(f"[appear]:[vacuum]:[sname(vacuum);position({_cdx}_{_cdy});bbox({_vac_diam}_{_vac_diam})]:({_cdx},{_cdy})|({_cdx},{_cdy})Object")




                    self._vacuum_initial_diam = _vac_diam
                    self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
                    return
            self.complete_action()
            self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
            return
        if self.action.id == GameAction.ACTION7:
            self._undo_last_move()
            self.complete_action()
            self._emit_diff_and_frame_logs(_before_sprites, _before_steps)
            return
        self.complete_action()
        self._emit_diff_and_frame_logs(_before_sprites, _before_steps)

    def _emit_diff_and_frame_logs(self, _before_sprites: dict, _before_steps: int) -> None:
        """



        """
        after_sprites: dict[int, object] = {
            id(spr): spr
            for spr in self.current_level.get_sprites()
        }



        def _is_canvas_pixel(name: str) -> bool:
            return name.startswith("canvas_pixel_")




        def _on_screen(dx: int, dy: int) -> bool:
            return 0 <= dx < 64 and 0 <= dy < 64


        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            if _is_canvas_pixel(spr.name):
                continue
            b = _before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                dx, dy, dw, dh = self._to_display_bbox(spr)
                if not _on_screen(dx, dy):
                    continue
                self.action_log.append(f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).")
                self.action_log_l2.append(f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name}({dx},{dy})|Object({dx},{dy})")



        _arc_scale_d = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in _before_sprites.items():
            if _is_canvas_pixel(b["name"]):
                continue
            spr = after_sprites.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = int(b.get("width", 0))  * _arc_scale_d
                bdh = int(b.get("height", 0)) * _arc_scale_d
                self.action_log.append(f"[disappear]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).")
                self.action_log_l2.append(f"[disappear]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']}({bdx},{bdy})|Object({bdx},{bdy})")


        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            if _is_canvas_pixel(spr.name):
                continue
            b = _before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                continue
            if (spr.x, spr.y) != (b["x"], b["y"]):
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                adx, ady, adw, adh = self._to_display_bbox(spr)
                before_on = _on_screen(bdx, bdy)
                after_on  = _on_screen(adx, ady)
                if before_on and after_on:
                    self.action_log.append(f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady}).")
                    self.action_log_l2.append(f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})")
                elif before_on and not after_on:



                    self.action_log.append(f"[disappear]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy});bbox({adw}_{adh})]:{spr.name} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).")
                    self.action_log_l2.append(f"[disappear]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy});bbox({adw}_{adh})]:({bdx},{bdy}){spr.name}|({bdx},{bdy})Object")
                elif not before_on and after_on:
                    self.action_log.append(f"[appear]:[{spr.name}]:[sname({spr.name});position({adx}_{ady});bbox({adw}_{adh})]:{spr.name} appears at ({adx},{ady}) (moved on-screen).|Object appears at ({adx},{ady}).")
                    self.action_log_l2.append(f"[appear]:[{spr.name}]:[sname({spr.name});position({adx}_{ady});bbox({adw}_{adh})]:{spr.name}({adx},{ady})|Object({adx},{ady})")

            if spr.rotation != b["rotation"]:
                _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
                self.action_log.append(f"[rotate]:[{spr.name}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:{spr.name} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.")
                self.action_log_l2.append(f"[rotate]:[{spr.name}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:({_rdx},{_rdy}){spr.name}{b['rotation']}°{spr.rotation}°|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°")

            if "pixels_bytes" in b and spr.pixels.tobytes() != b["pixels_bytes"]:
                after_color = self._primary_color_name(spr.pixels)
                before_color = b["primary_color"]
                if after_color != before_color:
                    adx, ady, adw, adh = self._to_display_bbox(spr)


                    if after_color == "white":
                        self.action_log.append(f"[bright]:[{spr.name}]:[sname({spr.name});state(flash);position({adx}_{ady});bbox({adw}_{adh})]:{spr.name} at ({adx},{ady}) flashes white (level-clear celebration).|Object at ({adx},{ady}) flashes white.")
                        self.action_log_l2.append(f"[bright]:[{spr.name}]:[sname({spr.name});state(flash);position({adx}_{ady});bbox({adw}_{adh})]:({adx},{ady}){spr.name}|({adx},{ady})Object")
                    else:
                        self.action_log.append(f"[change]:[{spr.name}]:[sname({spr.name});color({before_color})_({after_color});position({adx}_{ady});bbox({adw}_{adh})]:{spr.name} at ({adx},{ady}) changes color from {before_color} to {after_color}.|Object at ({adx},{ady}) changes color from {before_color} to {after_color}.")
                        self.action_log_l2.append(f"[change]:[{spr.name}]:[sname({spr.name});color({before_color})_({after_color});position({adx}_{ady});bbox({adw}_{adh})]:({adx},{ady}){spr.name}{self._color_l2(before_color)}{self._color_l2(after_color)}|({adx},{ady})Object{self._color_l2(before_color)}{self._color_l2(after_color)}")







        before_anim = getattr(self, "_before_anim_state", "none")
        before_radius = float(getattr(self, "_before_vacuum_radius_cur", 0.0))
        after_radius  = float(self.vacuum_radius_cur)
        _vacuum_cut_by_win = (before_anim == "vacuum" and self.anim_state == "win")
        if before_anim == "vacuum" and abs(before_radius - after_radius) > 1e-6 and not _vacuum_cut_by_win:
            _arc_scale_v = min(64 // self.camera.width, 64 // self.camera.height)



            def _diam(r: float) -> int:
                if r <= 0.0:
                    return 0
                return (2 * int(r + 0.5) + 1) * _arc_scale_v
            _bd = _diam(before_radius)
            _ad = _diam(after_radius)
            _cx, _cy = self._to_display_xy(*self.vacuum_center)
            if _bd != _ad:







                _from_diam = int(getattr(self, "_vacuum_initial_diam", _bd) or _bd)
                if after_radius < before_radius:
                    self.action_log.append(f"[shrink]:[vacuum]:[sname(vacuum);position({_cx}_{_cy});bbox({_from_diam}_{_from_diam})_({_ad}_{_ad})]:The vacuum ring centered at ({_cx},{_cy}) contracts from {_from_diam}x{_from_diam} to {_ad}x{_ad}.|The Object centered at ({_cx},{_cy}) contracts from {_from_diam}x{_from_diam} to {_ad}x{_ad}.")
                    self.action_log_l2.append(f"[shrink]:[vacuum]:[sname(vacuum);position({_cx}_{_cy});bbox({_from_diam}_{_from_diam})_({_ad}_{_ad})]:({_cx},{_cy}){_from_diam}x{_from_diam}{_ad}x{_ad}|({_cx},{_cy})Object{_from_diam}x{_from_diam}{_ad}x{_ad}")
                else:
                    self.action_log.append(f"[extend]:[vacuum]:[sname(vacuum);position({_cx}_{_cy});bbox({_from_diam}_{_from_diam})_({_ad}_{_ad})]:The vacuum ring centered at ({_cx},{_cy}) expands from {_from_diam}x{_from_diam} to {_ad}x{_ad}.|The Object centered at ({_cx},{_cy}) expands from {_from_diam}x{_from_diam} to {_ad}x{_ad}.")
                    self.action_log_l2.append(f"[extend]:[vacuum]:[sname(vacuum);position({_cx}_{_cy});bbox({_from_diam}_{_from_diam})_({_ad}_{_ad})]:({_cx},{_cy}){_from_diam}x{_from_diam}{_ad}x{_ad}|({_cx},{_cy})Object{_from_diam}x{_from_diam}{_ad}x{_ad}")






        if before_anim == "vacuum" and self.anim_state != "vacuum":
            _arc_v = min(64 // self.camera.width, 64 // self.camera.height)
            _cx2, _cy2 = self._to_display_xy(*self.vacuum_center)
            self.action_log.append(f"[disappear]:[vacuum]:[sname(vacuum);position({_cx2}_{_cy2});bbox({_arc_v}_{_arc_v})]:The vacuum ring at ({_cx2},{_cy2}) finishes contracting and disappears.|The Object at ({_cx2},{_cy2}) finishes contracting and disappears.")
            self.action_log_l2.append(f"[disappear]:[vacuum]:[sname(vacuum);position({_cx2}_{_cy2});bbox({_arc_v}_{_arc_v})]:({_cx2},{_cy2})|({_cx2},{_cy2})Object")
            self._vacuum_initial_diam = None


        after_steps = self.step_counter_ui.current_steps
        if after_steps != _before_steps:
            before_px = self._step_bar_pixel_len(_before_steps)
            after_px  = self._step_bar_pixel_len(after_steps)
            if before_px != after_px:
                if after_steps < _before_steps:
                    self.action_log.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:The white step bar at the bottom shrinks from the right ({_before_steps}→{after_steps} steps remain).|The white Object at the bottom shrinks from the right ({_before_steps}→{after_steps}).")
                    self.action_log_l2.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:{_before_steps}→{after_steps}|Object{_before_steps}→{after_steps}")
                else:
                    self.action_log.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:The white step bar at the bottom extends to the right ({_before_steps}→{after_steps} steps).|The white Object at the bottom extends to the right ({_before_steps}→{after_steps}).")
                    self.action_log_l2.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:{_before_steps}→{after_steps}|Object{_before_steps}→{after_steps}")


        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()

    def _on_click_grid(self, grid_x: int, grid_y: int) -> None:
        if grid_y <= GRID_TOP - 1 or grid_y >= GRID_BOTTOM:
            return
        self._update_fruit_positions_cache()
        gvflqukvtu = self.vacuum_radius
        self.knockback_set = set()
        self.knockback_tick_map = {}
        self.knockback_origin_map = {}
        self.knockback_dir_map = {}
        self.knockback_cur_pos_map = {}
        self.enemy_cooldown_map = {}
        self.fruit_pending_level_map = {}
        self.killed_fruits = set()
        self.attracted_fruits = []
        for vgtpakybrc in self.fruits:
            if self._point_in_radius_of_sprite(grid_x, grid_y, gvflqukvtu, vgtpakybrc):
                self.attracted_fruits.append(vgtpakybrc)
        self.attracted_enemies = []
        self.enemy_dir_map = {}
        self.enemy_pos_map = {}
        if self.enemies:
            for vxxrprjnav in self.enemies:
                if not self._point_in_radius_of_sprite(grid_x, grid_y, gvflqukvtu, vxxrprjnav):
                    continue
                self.attracted_enemies.append(vxxrprjnav)
                tjgvnkouuz = self.enemy_type_map.get(vxxrprjnav, ENEMY_TYPE_A)
                cptzxsjeyh = self._enemy_speed_for_type(tjgvnkouuz)
                if cptzxsjeyh == 1:
                    enemy_center_x, enemy_center_y = self._sprite_center(vxxrprjnav)
                    vpnrfpppjj = float(grid_x) - float(enemy_center_x)
                    flhganpjxt = float(grid_y) - float(enemy_center_y)
                    frslhefhml = vpnrfpppjj * vpnrfpppjj + flhganpjxt * flhganpjxt
                    if frslhefhml > 0.0:
                        smdamdpvbs = frslhefhml**0.5
                        okhjylumdn = vpnrfpppjj / smdamdpvbs
                        sfxtvvwind = flhganpjxt / smdamdpvbs
                    else:
                        okhjylumdn = 0.0
                        sfxtvvwind = 0.0
                    self.enemy_dir_map[vxxrprjnav] = (
                        okhjylumdn,
                        sfxtvvwind,
                    )
                    self.enemy_pos_map[vxxrprjnav] = (
                        float(vxxrprjnav.x),
                        float(vxxrprjnav.y),
                    )
        self.vacuum_center = (grid_x, grid_y)
        self.vacuum_tick = 0
        self.vacuum_radius_cur = float(self.vacuum_radius)
        if self.vacuum_total_steps > 1:
            self.vacuum_radius_step = float(self.vacuum_radius) / float(self.vacuum_total_steps - 1)
        else:
            self.vacuum_radius_step = float(self.vacuum_radius)
        self.vacuum_done = False
        self.animating = True
        self.anim_state = "vacuum"
        self._update_vacuum_ring()

    def _update_vacuum_ring(self) -> None:
        if self.vacuum_ring_sprite is None:
            return
        if self.vacuum_ring_sprite.pixels is None:
            return
        if self.mesh_x is None or self.mesh_y is None:
            return
        pixels = self.vacuum_ring_sprite.pixels
        pixels[:, :] = TRANSPARENT
        gvflqukvtu = self.vacuum_radius_cur
        if gvflqukvtu <= 0.0:
            return
        umyncnjoxc, fbynmfmnkb = self.vacuum_center
        dx = self.mesh_x - float(umyncnjoxc)
        dy = self.mesh_y - float(fbynmfmnkb)
        jojfalzkuj = dx * dx + dy * dy
        nsplbotuaq = gvflqukvtu + 0.5
        hlliyirjkc = max(0.0, gvflqukvtu - 0.5)
        ajgubownbe = nsplbotuaq * nsplbotuaq
        jfnwcbjlft = hlliyirjkc * hlliyirjkc
        gcpsvytzxh = (jojfalzkuj <= ajgubownbe) & (jojfalzkuj >= jfnwcbjlft)
        pixels[gcpsvytzxh] = WHITE

    def _clear_vacuum_ring(self) -> None:
        if self.vacuum_ring_sprite is None:
            return
        if self.vacuum_ring_sprite.pixels is None:
            return
        self.vacuum_ring_sprite.pixels[:, :] = TRANSPARENT

    def _process_vacuum_fruits(self) -> None:
        self._process_vacuum_enemies()
        if self.anim_state == "flash":
            return
        if not self.attracted_fruits:
            return
        self._update_fruit_positions_cache()
        nuekotmpxi = len(self.attracted_fruits)
        if nuekotmpxi < 2:
            return
        ukrrnnuysj = [self.fruit_level_map.get(sprite, 0) for sprite in self.attracted_fruits]
        rrsxgaznjn = list(range(nuekotmpxi))

        def ikmvvzofng(index: int) -> int:
            while rrsxgaznjn[index] != index:
                rrsxgaznjn[index] = rrsxgaznjn[rrsxgaznjn[index]]
                index = rrsxgaznjn[index]
            return index

        def icliwhbnqx(thavbderam: int, ekdpitzaju: int) -> None:
            vdvfmqirba = ikmvvzofng(thavbderam)
            cbtkkwnuig = ikmvvzofng(ekdpitzaju)
            if vdvfmqirba == cbtkkwnuig:
                return
            rrsxgaznjn[cbtkkwnuig] = vdvfmqirba

        for i in range(nuekotmpxi):
            uypdqbxdnt = self.attracted_fruits[i]
            for ituakcbkov in range(i + 1, nuekotmpxi):
                dpcwawzfil = self.attracted_fruits[ituakcbkov]
                if not self._sprites_overlap(uypdqbxdnt, dpcwawzfil):
                    continue
                lztrvkkkzz = ukrrnnuysj[i]
                aggcncmcfq = ukrrnnuysj[ituakcbkov]
                if lztrvkkkzz != aggcncmcfq:
                    self._start_flash_animation([uypdqbxdnt, dpcwawzfil])
                    return
                icliwhbnqx(i, ituakcbkov)
        ixerilvqvr: dict[int, list[int]] = {}
        for index in range(nuekotmpxi):
            jtwusevbhw = ikmvvzofng(index)
            if jtwusevbhw not in ixerilvqvr:
                ixerilvqvr[jtwusevbhw] = []
            ixerilvqvr[jtwusevbhw].append(index)
        sqkcnvjkcn: list[list[Sprite]] = []
        for mncdxcqyyu in ixerilvqvr.values():
            if len(mncdxcqyyu) < 2:
                continue
            qwjzfehqwx: list[Sprite] = []
            for index in mncdxcqyyu:
                qwjzfehqwx.append(self.attracted_fruits[index])
            sqkcnvjkcn.append(qwjzfehqwx)
        if not sqkcnvjkcn:
            return
        self._merge_fruit_groups(sqkcnvjkcn)
        self._update_fruit_positions_cache()

    def _process_vacuum_enemies(self) -> None:
        ecbrzskltw = len(self.enemies)
        if ecbrzskltw < 2:
            return
        qvguicvytj: list[str] = [self.enemy_type_map.get(vxxrprjnav, ENEMY_TYPE_A) for vxxrprjnav in self.enemies]
        rrsxgaznjn = list(range(ecbrzskltw))

        def ikmvvzofng(index: int) -> int:
            while rrsxgaznjn[index] != index:
                rrsxgaznjn[index] = rrsxgaznjn[rrsxgaznjn[index]]
                index = rrsxgaznjn[index]
            return index

        def icliwhbnqx(thavbderam: int, ekdpitzaju: int) -> None:
            vdvfmqirba = ikmvvzofng(thavbderam)
            cbtkkwnuig = ikmvvzofng(ekdpitzaju)
            if vdvfmqirba == cbtkkwnuig:
                return
            rrsxgaznjn[cbtkkwnuig] = vdvfmqirba

        for i in range(ecbrzskltw):
            jsypthuvlv = self.enemies[i]
            for ituakcbkov in range(i + 1, ecbrzskltw):
                wddpazkfuz = self.enemies[ituakcbkov]
                if not self._sprites_overlap(jsypthuvlv, wddpazkfuz):
                    continue
                jdnnpadnur = qvguicvytj[i]
                ccvpmdiwjt = qvguicvytj[ituakcbkov]
                if jdnnpadnur != ccvpmdiwjt:
                    self._start_flash_animation([jsypthuvlv, wddpazkfuz])
                    return
                icliwhbnqx(i, ituakcbkov)
        ixerilvqvr: dict[int, list[int]] = {}
        for index in range(ecbrzskltw):
            jtwusevbhw = ikmvvzofng(index)
            if jtwusevbhw not in ixerilvqvr:
                ixerilvqvr[jtwusevbhw] = []
            ixerilvqvr[jtwusevbhw].append(index)
        sqkcnvjkcn: list[list[Sprite]] = []
        for mncdxcqyyu in ixerilvqvr.values():
            if len(mncdxcqyyu) < 2:
                continue
            qwjzfehqwx: list[Sprite] = []
            for index in mncdxcqyyu:
                qwjzfehqwx.append(self.enemies[index])
            sqkcnvjkcn.append(qwjzfehqwx)
        if not sqkcnvjkcn:
            return
        self._evolve_enemy_groups(sqkcnvjkcn)

    def _merge_fruit_groups(self, sqkcnvjkcn: list[list[Sprite]]) -> None:
        nobhaarcjr = len(FRUIT_SPRITE_KEYS) - 1

        def mdknvglful(sprite: Sprite) -> None:
            if sprite in self.knockback_set:
                self.knockback_set.discard(sprite)
            if sprite in self.knockback_tick_map:
                del self.knockback_tick_map[sprite]
            if sprite in self.knockback_origin_map:
                del self.knockback_origin_map[sprite]
            if sprite in self.knockback_dir_map:
                del self.knockback_dir_map[sprite]
            if sprite in self.knockback_cur_pos_map:
                del self.knockback_cur_pos_map[sprite]
            if sprite in self.fruit_pending_level_map:
                del self.fruit_pending_level_map[sprite]
            if sprite in self.killed_fruits:
                self.killed_fruits.discard(sprite)

        for qwjzfehqwx in sqkcnvjkcn:
            if not qwjzfehqwx:
                continue
            kcacvefoot = self.fruit_level_map.get(qwjzfehqwx[0], 0)
            if kcacvefoot >= nobhaarcjr:
                for sprite in qwjzfehqwx:
                    mdknvglful(sprite)
                    self.current_level.remove_sprite(sprite)
                    if sprite in self.fruits:
                        self.fruits.remove(sprite)
                    if sprite in self.fruit_level_map:
                        del self.fruit_level_map[sprite]
                    if sprite in self.attracted_fruits:
                        self.attracted_fruits.remove(sprite)
                continue
            fqjuxupjdi = kcacvefoot + 1
            bwtdecdvlt = FRUIT_SPRITE_KEYS[fqjuxupjdi]
            eacpgvdnya = sprites[bwtdecdvlt]
            isvnxzxwks = eacpgvdnya.clone()
            rcfwyjbamh = 0
            jfwksyrdzk = 0
            for sprite in qwjzfehqwx:
                umyncnjoxc, fbynmfmnkb = self._sprite_center(sprite)
                rcfwyjbamh += umyncnjoxc
                jfwksyrdzk += fbynmfmnkb
            umyncnjoxc = rcfwyjbamh // len(qwjzfehqwx)
            fbynmfmnkb = jfwksyrdzk // len(qwjzfehqwx)
            if isvnxzxwks.pixels is not None:
                sprite_height, sprite_width = isvnxzxwks.pixels.shape
            else:
                sprite_height, sprite_width = (1, 1)
            qfcgjyyogx = umyncnjoxc - sprite_width // 2
            tvfxhjorfu = fbynmfmnkb - sprite_height // 2
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.grid_width - sprite_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < GRID_TOP:
                tvfxhjorfu = GRID_TOP
            if tvfxhjorfu > GRID_BOTTOM:
                tvfxhjorfu = GRID_BOTTOM
            max_y = self.grid_height - sprite_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            isvnxzxwks.set_position(qfcgjyyogx, tvfxhjorfu)
            self.current_level.add_sprite(isvnxzxwks)
            self.fruits.append(isvnxzxwks)
            self.fruit_level_map[isvnxzxwks] = fqjuxupjdi
            for sprite in qwjzfehqwx:
                mdknvglful(sprite)
                self.current_level.remove_sprite(sprite)
                if sprite in self.fruits:
                    self.fruits.remove(sprite)
                if sprite in self.fruit_level_map:
                    del self.fruit_level_map[sprite]
                if sprite in self.attracted_fruits:
                    self.attracted_fruits.remove(sprite)

    def _evolve_enemy_groups(self, sqkcnvjkcn: list[list[Sprite]]) -> None:
        for qwjzfehqwx in sqkcnvjkcn:
            if not qwjzfehqwx:
                continue
            nvgwiuywco = qwjzfehqwx[0]
            tjgvnkouuz = self.enemy_type_map.get(nvgwiuywco, ENEMY_TYPE_A)
            zketwqvgup = self._next_enemy_type(tjgvnkouuz)
            if zketwqvgup is None:
                for vxxrprjnav in qwjzfehqwx:
                    self._remove_enemy(vxxrprjnav)
                continue
            cgmmpmplah = self._enemy_sprite_for_type(zketwqvgup)
            yagfbqunge = cgmmpmplah.clone()
            rcfwyjbamh = 0
            jfwksyrdzk = 0
            for vxxrprjnav in qwjzfehqwx:
                umyncnjoxc, fbynmfmnkb = self._sprite_center(vxxrprjnav)
                rcfwyjbamh += umyncnjoxc
                jfwksyrdzk += fbynmfmnkb
            umyncnjoxc = rcfwyjbamh // len(qwjzfehqwx)
            fbynmfmnkb = jfwksyrdzk // len(qwjzfehqwx)
            if yagfbqunge.pixels is not None:
                sprite_height, sprite_width = yagfbqunge.pixels.shape
            else:
                sprite_height, sprite_width = (1, 1)
            qfcgjyyogx = umyncnjoxc - sprite_width // 2
            tvfxhjorfu = fbynmfmnkb - sprite_height // 2
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.grid_width - sprite_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < GRID_TOP:
                tvfxhjorfu = GRID_TOP
            if tvfxhjorfu > GRID_BOTTOM:
                tvfxhjorfu = GRID_BOTTOM
            max_y = self.grid_height - sprite_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            yagfbqunge.set_position(qfcgjyyogx, tvfxhjorfu)
            self.current_level.add_sprite(yagfbqunge)
            self.enemies.append(yagfbqunge)
            self.enemy_type_map[yagfbqunge] = zketwqvgup
            if self.animating and self.anim_state == "vacuum":
                self.enemy_cooldown_map[yagfbqunge] = self.vacuum_total_steps
            for vxxrprjnav in qwjzfehqwx:
                self._remove_enemy(vxxrprjnav)

    def _start_flash_animation(self, vuzbgsqdwt: list[Sprite]) -> None:
        self._clear_vacuum_ring()
        self.flash_sprites = []
        self.flash_pixels_backup = {}
        for sprite in vuzbgsqdwt:
            if sprite.pixels is None:
                continue
            if sprite in self.flash_pixels_backup:
                continue
            self.flash_sprites.append(sprite)
            self.flash_pixels_backup[sprite] = sprite.pixels.copy()
        for sprite in self.flash_sprites:
            pixels = sprite.pixels
            mpalcyrcyw = pixels > -1
            pixels[mpalcyrcyw] = WHITE
        self.flash_tick = 0
        self.animating = True
        self.anim_state = "flash"

    def _tick_flash_animation(self) -> bool:
        if not self.flash_sprites:
            return False
        wykwcvbijt = self.flash_tick // 2 % 2 == 0
        for sprite in self.flash_sprites:
            if sprite.pixels is None:
                continue
            pixels = sprite.pixels
            if sprite not in self.flash_pixels_backup:
                continue
            if wykwcvbijt:
                mpalcyrcyw = pixels > -1
                pixels[mpalcyrcyw] = WHITE
            else:
                shobdbrmxz = self.flash_pixels_backup[sprite]
                pixels[:, :] = shobdbrmxz
        self.flash_tick += 1
        if self.flash_tick < self.flash_total_frames:
            return True
        return False

    def _finish_flash_animation(self) -> None:
        for sprite, shobdbrmxz in self.flash_pixels_backup.items():
            if sprite.pixels is None:
                continue
            sprite.pixels[:, :] = shobdbrmxz
        self.flash_sprites = []
        self.flash_pixels_backup = {}

    def _move_attracted_sprites(self) -> bool:
        umyncnjoxc, fbynmfmnkb = self.vacuum_center
        zittpgpamq: list[Sprite] = []
        if self.attracted_fruits:
            zittpgpamq.extend(self.attracted_fruits)
        if self.attracted_enemies:
            zittpgpamq.extend(self.attracted_enemies)
        qfkdcxgkgj = self._is_knockback_active()
        for sprite in zittpgpamq:
            if sprite in self.knockback_set:
                continue
            if sprite in self.enemies and qfkdcxgkgj:
                continue
            qhffhgeifo = sprite in self.enemy_dir_map and sprite in self.enemy_pos_map
            if qhffhgeifo:
                okhjylumdn, sfxtvvwind = self.enemy_dir_map[sprite]
                hwnebnrvfl, svxmgnbxtk = self.enemy_pos_map[sprite]
                cmfhziahuk = float(self.sprite_move_speed * 0.85)
                hwnebnrvfl += okhjylumdn * cmfhziahuk
                svxmgnbxtk += sfxtvvwind * cmfhziahuk
                qfcgjyyogx = int(round(hwnebnrvfl))
                tvfxhjorfu = int(round(svxmgnbxtk))
            else:
                sprite_center_x, sprite_center_y = self._sprite_center(sprite)
                dx = umyncnjoxc - sprite_center_x
                dy = fbynmfmnkb - sprite_center_y
                tewydphtje = 0
                rzcfxufysa = 0
                if dx > 0:
                    tewydphtje = min(self.sprite_move_speed, dx)
                elif dx < 0:
                    tewydphtje = max(-self.sprite_move_speed, dx)
                if dy > 0:
                    rzcfxufysa = min(self.sprite_move_speed, dy)
                elif dy < 0:
                    rzcfxufysa = max(-self.sprite_move_speed, dy)
                qfcgjyyogx = sprite.x + tewydphtje
                tvfxhjorfu = sprite.y + rzcfxufysa
            if sprite.pixels is not None:
                sprite_height, sprite_width = sprite.pixels.shape
            else:
                sprite_height, sprite_width = (1, 1)
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.grid_width - sprite_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < GRID_TOP:
                tvfxhjorfu = GRID_TOP
            if tvfxhjorfu > GRID_BOTTOM:
                tvfxhjorfu = GRID_BOTTOM
            max_y = self.grid_height - sprite_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            if qhffhgeifo:
                self.enemy_pos_map[sprite] = (
                    float(qfcgjyyogx),
                    float(tvfxhjorfu),
                )
            sprite.set_position(qfcgjyyogx, tvfxhjorfu)
        self._tick_enemy_ai()
        if self.anim_state != "vacuum":
            self._clear_vacuum_ring()
            return True
        self.vacuum_tick += 1
        self.vacuum_radius_cur = max(0.0, self.vacuum_radius_cur - self.vacuum_radius_step)
        self._update_vacuum_ring()
        if self.vacuum_tick < self.vacuum_total_steps:
            return True
        if self.knockback_set:
            self._clear_vacuum_ring()
            return True
        self._clear_vacuum_ring()
        return False

    def _tick_enemy_ai(self) -> None:
        myvsuxlxul = self._is_knockback_active()
        if self.knockback_set:
            self._tick_knockback()
        if self.enemy_cooldown_map:
            akmfbjzmzl: list[Sprite] = []
            for vxxrprjnav, wenjougvog in self.enemy_cooldown_map.items():
                if wenjougvog > 0:
                    self.enemy_cooldown_map[vxxrprjnav] = wenjougvog - 1
                if self.enemy_cooldown_map[vxxrprjnav] <= 0:
                    akmfbjzmzl.append(vxxrprjnav)
            for vxxrprjnav in akmfbjzmzl:
                del self.enemy_cooldown_map[vxxrprjnav]
        if self.vacuum_tick >= self.vacuum_total_steps:
            return
        if not self.enemies:
            return
        if not self.fruits:
            return
        if myvsuxlxul:
            return
        plklcqdime = set(self.attracted_enemies)
        scfdcrilps = set(self.enemy_cooldown_map.keys())
        for vxxrprjnav in self.enemies:
            if vxxrprjnav in plklcqdime:
                continue
            if vxxrprjnav in scfdcrilps:
                continue
            emjjkgbxif = False
            for vgtpakybrc in self.fruits:
                if vgtpakybrc not in self.knockback_set:
                    emjjkgbxif = True
                    break
            if not emjjkgbxif:
                break
            enemy_center_x, enemy_center_y = self._sprite_center(vxxrprjnav)
            gnrmbrgloh: Sprite | None = None
            yzqonqsxci: int | None = None
            for vgtpakybrc in self.fruits:
                if vgtpakybrc in self.knockback_set:
                    continue
                fruit_center_x, fruit_center_y = self._sprite_center(vgtpakybrc)
                dx = fruit_center_x - enemy_center_x
                dy = fruit_center_y - enemy_center_y
                jojfalzkuj = dx * dx + dy * dy
                if yzqonqsxci is None or jojfalzkuj < yzqonqsxci:
                    yzqonqsxci = jojfalzkuj
                    gnrmbrgloh = vgtpakybrc
            if gnrmbrgloh is None:
                continue
            target_center_x, target_center_y = self._sprite_center(gnrmbrgloh)
            tjgvnkouuz = self.enemy_type_map.get(vxxrprjnav, ENEMY_TYPE_A)
            lvkjczbkpm = 2 if tjgvnkouuz == ENEMY_TYPE_C else 1
            cvdguhkpob = 0
            msscqgccsv = 0
            if target_center_x > enemy_center_x:
                cvdguhkpob = lvkjczbkpm
            elif target_center_x < enemy_center_x:
                cvdguhkpob = -lvkjczbkpm
            if target_center_y > enemy_center_y:
                msscqgccsv = lvkjczbkpm
            elif target_center_y < enemy_center_y:
                msscqgccsv = -lvkjczbkpm
            qfcgjyyogx = vxxrprjnav.x + cvdguhkpob
            tvfxhjorfu = vxxrprjnav.y + msscqgccsv
            if vxxrprjnav.pixels is not None:
                enemy_height, enemy_width = vxxrprjnav.pixels.shape
            else:
                enemy_height, enemy_width = (1, 1)
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.grid_width - enemy_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < GRID_TOP:
                tvfxhjorfu = GRID_TOP
            if tvfxhjorfu > GRID_BOTTOM:
                tvfxhjorfu = GRID_BOTTOM
            max_y = self.grid_height - enemy_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            vxxrprjnav.set_position(qfcgjyyogx, tvfxhjorfu)
        if not self.fruits:
            return
        for vxxrprjnav in self.enemies:
            if vxxrprjnav in scfdcrilps:
                continue
            if vxxrprjnav in plklcqdime:
                continue
            for vgtpakybrc in self.fruits:
                if vgtpakybrc in self.knockback_set:
                    continue
                if not self._sprites_overlap(vxxrprjnav, vgtpakybrc):
                    continue
                self._apply_enemy_hit_fruit(vxxrprjnav, vgtpakybrc)
        self._update_fruit_positions_cache()
        self._process_vacuum_enemies()
        if self.anim_state == "flash":
            return

    def _is_knockback_active(self) -> bool:
        if not self.knockback_set:
            return False
        if self.knockback_fly_frames <= 0:
            return False
        for vgtpakybrc in self.knockback_set:
            if vgtpakybrc in self.killed_fruits:
                continue
            syelanegrq = self.knockback_tick_map.get(vgtpakybrc, 0)
            if syelanegrq >= self.knockback_shake_frames:
                jzcfcenqzz = self.knockback_shake_frames + self.knockback_fly_frames
                if syelanegrq < jzcfcenqzz:
                    return True
        return False

    def _apply_enemy_hit_fruit(self, vxxrprjnav: Sprite, vgtpakybrc: Sprite) -> None:
        if vgtpakybrc not in self.fruits:
            return
        if vgtpakybrc in self.knockback_set:
            return
        amnmgwpkeb = self.fruit_level_map.get(vgtpakybrc, 0)
        rzdkhogqmi = amnmgwpkeb <= 0
        if not rzdkhogqmi:
            biawrretvq = amnmgwpkeb - 1
            self.fruit_pending_level_map[vgtpakybrc] = biawrretvq
        fruit_center_x, fruit_center_y = self._sprite_center(vgtpakybrc)
        enemy_center_x, enemy_center_y = self._sprite_center(vxxrprjnav)
        okhjylumdn = float(fruit_center_x - enemy_center_x)
        sfxtvvwind = float(fruit_center_y - enemy_center_y)
        frslhefhml = okhjylumdn * okhjylumdn + sfxtvvwind * sfxtvvwind
        if frslhefhml > 0.0:
            smdamdpvbs = frslhefhml**0.5
            okhjylumdn /= smdamdpvbs
            sfxtvvwind /= smdamdpvbs
        else:
            okhjylumdn = 0.0
            sfxtvvwind = -1.0
        if okhjylumdn == 0.0 and sfxtvvwind == 0.0:
            sfxtvvwind = -1.0
        self.knockback_set.add(vgtpakybrc)
        self.knockback_tick_map[vgtpakybrc] = 0
        eemwekvrse = float(vgtpakybrc.x)
        bctbqavhmy = float(vgtpakybrc.y)
        self.knockback_origin_map[vgtpakybrc] = (eemwekvrse, bctbqavhmy)
        self.knockback_dir_map[vgtpakybrc] = (okhjylumdn, sfxtvvwind)
        self.knockback_cur_pos_map[vgtpakybrc] = (eemwekvrse, bctbqavhmy)
        if rzdkhogqmi:
            self.killed_fruits.add(vgtpakybrc)
        jzcfcenqzz = self.knockback_shake_frames + self.knockback_fly_frames + 1
        self.enemy_cooldown_map[vxxrprjnav] = jzcfcenqzz

    def _tick_knockback(self) -> None:
        if not self.knockback_set:
            return
        fxhuvimusc: list[Sprite] = []
        jzcfcenqzz = self.knockback_shake_frames + self.knockback_fly_frames
        for vgtpakybrc in list(self.knockback_set):
            if vgtpakybrc not in self.knockback_tick_map:
                fxhuvimusc.append(vgtpakybrc)
                continue
            syelanegrq = self.knockback_tick_map[vgtpakybrc]
            tijhbemqxz = self.knockback_origin_map.get(vgtpakybrc)
            hfgsbvmopf = self.knockback_dir_map.get(vgtpakybrc)
            lvwhnuzrsl = self.knockback_cur_pos_map.get(vgtpakybrc)
            if tijhbemqxz is None or hfgsbvmopf is None or lvwhnuzrsl is None:
                fxhuvimusc.append(vgtpakybrc)
                continue
            kcxfklpxii = jzcfcenqzz
            if vgtpakybrc in self.killed_fruits:
                kcxfklpxii = self.knockback_shake_frames
            eemwekvrse, bctbqavhmy = tijhbemqxz
            okhjylumdn, sfxtvvwind = hfgsbvmopf
            hwnebnrvfl, svxmgnbxtk = lvwhnuzrsl
            if syelanegrq < self.knockback_shake_frames:
                dzvmzsimej = syelanegrq % 4
                aieskpwdyr = 0.0
                if dzvmzsimej == 1:
                    aieskpwdyr = -1.0
                elif dzvmzsimej == 3:
                    aieskpwdyr = 1.0
                hwnebnrvfl = eemwekvrse
                svxmgnbxtk = bctbqavhmy + aieskpwdyr
                if self.knockback_fly_frames == 0 and syelanegrq + 1 == self.knockback_shake_frames and (vgtpakybrc in self.fruit_pending_level_map):
                    biawrretvq = self.fruit_pending_level_map[vgtpakybrc]
                    self.fruit_level_map[vgtpakybrc] = biawrretvq
                    bwtdecdvlt = FRUIT_SPRITE_KEYS[biawrretvq]
                    cgmmpmplah = sprites[bwtdecdvlt]
                    if cgmmpmplah.pixels is not None:
                        vgtpakybrc.pixels = cgmmpmplah.pixels.copy()
                    del self.fruit_pending_level_map[vgtpakybrc]
            else:
                if vgtpakybrc in self.fruit_pending_level_map:
                    biawrretvq = self.fruit_pending_level_map[vgtpakybrc]
                    self.fruit_level_map[vgtpakybrc] = biawrretvq
                    bwtdecdvlt = FRUIT_SPRITE_KEYS[biawrretvq]
                    cgmmpmplah = sprites[bwtdecdvlt]
                    if cgmmpmplah.pixels is not None:
                        vgtpakybrc.pixels = cgmmpmplah.pixels.copy()
                    del self.fruit_pending_level_map[vgtpakybrc]
                if self.knockback_fly_frames > 0:
                    qowlxepxpm = self.knockback_fly_speed / float(self.knockback_fly_frames)
                else:
                    qowlxepxpm = self.knockback_fly_speed
                hwnebnrvfl += okhjylumdn * qowlxepxpm
                svxmgnbxtk += sfxtvvwind * qowlxepxpm
                self.knockback_cur_pos_map[vgtpakybrc] = (hwnebnrvfl, svxmgnbxtk)
            qfcgjyyogx = int(round(hwnebnrvfl))
            tvfxhjorfu = int(round(svxmgnbxtk))
            if vgtpakybrc.pixels is not None:
                fruit_height, fruit_width = vgtpakybrc.pixels.shape
            else:
                fruit_height, fruit_width = (1, 1)
            if qfcgjyyogx < 0:
                qfcgjyyogx = 0
            max_x = self.grid_width - fruit_width
            if qfcgjyyogx > max_x:
                qfcgjyyogx = max_x
            if tvfxhjorfu < GRID_TOP:
                tvfxhjorfu = GRID_TOP
            if tvfxhjorfu > GRID_BOTTOM:
                tvfxhjorfu = GRID_BOTTOM
            max_y = self.grid_height - fruit_height
            if tvfxhjorfu > max_y:
                tvfxhjorfu = max_y
            vgtpakybrc.set_position(qfcgjyyogx, tvfxhjorfu)
            syelanegrq += 1
            self.knockback_tick_map[vgtpakybrc] = syelanegrq
            if syelanegrq >= kcxfklpxii:
                fxhuvimusc.append(vgtpakybrc)
        if not fxhuvimusc:
            return
        for vgtpakybrc in fxhuvimusc:
            self.knockback_set.discard(vgtpakybrc)
            if vgtpakybrc in self.knockback_tick_map:
                del self.knockback_tick_map[vgtpakybrc]
            if vgtpakybrc in self.knockback_origin_map:
                del self.knockback_origin_map[vgtpakybrc]
            if vgtpakybrc in self.knockback_dir_map:
                del self.knockback_dir_map[vgtpakybrc]
            if vgtpakybrc in self.knockback_cur_pos_map:
                del self.knockback_cur_pos_map[vgtpakybrc]
            if vgtpakybrc in self.fruit_pending_level_map:
                del self.fruit_pending_level_map[vgtpakybrc]
            if vgtpakybrc in self.killed_fruits:
                self.killed_fruits.discard(vgtpakybrc)
                self.current_level.remove_sprite(vgtpakybrc)
                if vgtpakybrc in self.fruits:
                    self.fruits.remove(vgtpakybrc)
                if vgtpakybrc in self.fruit_level_map:
                    del self.fruit_level_map[vgtpakybrc]
                if vgtpakybrc in self.attracted_fruits:
                    self.attracted_fruits.remove(vgtpakybrc)

    def _get_fruit_level(self, sprite: Sprite) -> int:
        for tag in sprite.tags:
            if tag == "fruit":
                continue
            return int(tag)
        return 0

    def _update_fruit_positions_cache(self) -> None:
        qplkqujnlq = len(self.fruits)
        xqefcotwls = np.zeros((qplkqujnlq, 2), dtype=np.int64)
        ukrrnnuysj = np.zeros((qplkqujnlq,), dtype=np.int64)
        for i, sprite in enumerate(self.fruits):
            umyncnjoxc, fbynmfmnkb = self._sprite_center(sprite)
            xqefcotwls[i, 0] = umyncnjoxc
            xqefcotwls[i, 1] = fbynmfmnkb
            ukrrnnuysj[i] = self.fruit_level_map.get(sprite, 0)
        self.fruit_positions = xqefcotwls
        self.fruit_levels_arr = ukrrnnuysj

    def _sprite_center(self, sprite: Sprite) -> tuple[int, int]:
        if sprite.pixels is not None:
            sprite_height, sprite_width = sprite.pixels.shape
        else:
            sprite_height, sprite_width = (1, 1)
        umyncnjoxc = sprite.x + sprite_width // 2
        fbynmfmnkb = sprite.y + sprite_height // 2
        return (umyncnjoxc, fbynmfmnkb)

    def _sprites_overlap(self, uypdqbxdnt: Sprite, dpcwawzfil: Sprite) -> bool:
        if uypdqbxdnt.pixels is not None:
            height_a, width_a = uypdqbxdnt.pixels.shape
        else:
            height_a, width_a = (1, 1)
        if dpcwawzfil.pixels is not None:
            height_b, width_b = dpcwawzfil.pixels.shape
        else:
            height_b, width_b = (1, 1)
        oldwyeshdz = uypdqbxdnt.x
        fuhwxypsmw = uypdqbxdnt.y
        dgcogkqmbd = uypdqbxdnt.x + width_a
        rrznqaqqkg = uypdqbxdnt.y + height_a
        zabmerqpsb = dpcwawzfil.x
        ujlimziild = dpcwawzfil.y
        camvkzoytz = dpcwawzfil.x + width_b
        feinskarfd = dpcwawzfil.y + height_b
        if dgcogkqmbd <= zabmerqpsb:
            return False
        if camvkzoytz <= oldwyeshdz:
            return False
        if rrznqaqqkg <= ujlimziild:
            return False
        if feinskarfd <= fuhwxypsmw:
            return False
        return True

    def _point_in_radius_of_sprite(self, umyncnjoxc: int, fbynmfmnkb: int, gvflqukvtu: int, sprite: Sprite) -> bool:
        if sprite.pixels is not None:
            sprite_height, sprite_width = sprite.pixels.shape
        else:
            sprite_height, sprite_width = (1, 1)
        obgidvqngh = sprite.x
        xzewyycdue = sprite.y
        zwzbcbwpyh = sprite.x + sprite_width
        khfytdaqqf = sprite.y + sprite_height
        if umyncnjoxc < obgidvqngh:
            pormmcwelo = obgidvqngh
        elif umyncnjoxc > zwzbcbwpyh - 1:
            pormmcwelo = zwzbcbwpyh - 1
        else:
            pormmcwelo = umyncnjoxc
        if fbynmfmnkb < xzewyycdue:
            bcwunhexxc = xzewyycdue
        elif fbynmfmnkb > khfytdaqqf - 1:
            bcwunhexxc = khfytdaqqf - 1
        else:
            bcwunhexxc = fbynmfmnkb
        dx = umyncnjoxc - pormmcwelo
        dy = fbynmfmnkb - bcwunhexxc
        return dx * dx + dy * dy <= gvflqukvtu * gvflqukvtu

    def _enemy_speed_for_type(self, tjgvnkouuz: str) -> int:
        if tjgvnkouuz == ENEMY_TYPE_A:
            return 1
        if tjgvnkouuz == ENEMY_TYPE_B:
            return 2
        if tjgvnkouuz == ENEMY_TYPE_C:
            return 3
        return 1

    def _enemy_type_for_level(self, level: int) -> str:
        if level <= 1:
            return ENEMY_TYPE_A
        if level == 2:
            return ENEMY_TYPE_B
        return ENEMY_TYPE_C

    def _next_enemy_type(self, tjgvnkouuz: str) -> str | None:
        if tjgvnkouuz == ENEMY_TYPE_A:
            return ENEMY_TYPE_B
        if tjgvnkouuz == ENEMY_TYPE_B:
            return ENEMY_TYPE_C
        return None

    def _enemy_sprite_for_type(self, tjgvnkouuz: str) -> Sprite:
        if tjgvnkouuz == ENEMY_TYPE_A:
            return sprites[TAG_ENEMY]
        if tjgvnkouuz == ENEMY_TYPE_B:
            return sprites[TAG_ENEMY2]
        if tjgvnkouuz == ENEMY_TYPE_C:
            return sprites[TAG_ENEMY3]
        return sprites[TAG_ENEMY]

    def _remove_enemy(self, vxxrprjnav: Sprite) -> None:
        self.current_level.remove_sprite(vxxrprjnav)
        if vxxrprjnav in self.enemies:
            self.enemies.remove(vxxrprjnav)
        if vxxrprjnav in self.enemy_type_map:
            del self.enemy_type_map[vxxrprjnav]
        if vxxrprjnav in self.attracted_enemies:
            self.attracted_enemies.remove(vxxrprjnav)
        if vxxrprjnav in self.enemy_cooldown_map:
            del self.enemy_cooldown_map[vxxrprjnav]
        if vxxrprjnav in self.enemy_dir_map:
            del self.enemy_dir_map[vxxrprjnav]
        if vxxrprjnav in self.enemy_pos_map:
            del self.enemy_pos_map[vxxrprjnav]

    def _check_goal_overlap(self) -> bool:
        if not self.goals:
            return False
        if self.goal_spec is None:
            return False
        ifeuhzhcxr = self.goal_spec
        krdxxrzrlz = ifeuhzhcxr[0]
        gmjltxweel: list[tuple[str, int]]
        if isinstance(krdxxrzrlz, (list, tuple)):
            gmjltxweel = []
            for mvrnytgvvx, aqfhmtmxhi in ifeuhzhcxr:
                gmjltxweel.append((str(mvrnytgvvx), int(aqfhmtmxhi)))
        else:
            mvrnytgvvx = str(ifeuhzhcxr[0])
            aqfhmtmxhi = int(ifeuhzhcxr[1])
            gmjltxweel = [(mvrnytgvvx, aqfhmtmxhi)]
        nhefdqizsw: dict[int, int] = {}
        xbxnoponrg: dict[str, int] = {}
        for vgtpakybrc in self.fruits:
            naojwrjpdq = False
            for coitlkmvpm in self.goals:
                if self._sprites_overlap(coitlkmvpm, vgtpakybrc):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            thujeoyajx = self.fruit_level_map.get(vgtpakybrc, 0)
            if thujeoyajx in nhefdqizsw:
                nhefdqizsw[thujeoyajx] += 1
            else:
                nhefdqizsw[thujeoyajx] = 1
        for vxxrprjnav in self.enemies:
            naojwrjpdq = False
            for coitlkmvpm in self.goals:
                if self._sprites_overlap(coitlkmvpm, vxxrprjnav):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            tjgvnkouuz = self.enemy_type_map.get(vxxrprjnav, ENEMY_TYPE_A)
            if tjgvnkouuz in xbxnoponrg and xbxnoponrg is not None:
                xbxnoponrg[tjgvnkouuz] += 1
            else:
                xbxnoponrg[tjgvnkouuz] = 1
        for xecuihaefu, aqfhmtmxhi in gmjltxweel:
            if xecuihaefu in (ENEMY_TYPE_A, ENEMY_TYPE_B, ENEMY_TYPE_C):
                if xbxnoponrg.get(xecuihaefu, 0) != aqfhmtmxhi:
                    return False
            else:
                thujeoyajx = int(xecuihaefu)
                if nhefdqizsw.get(thujeoyajx, 0) != aqfhmtmxhi:
                    return False
        return True

    def _point_in_sprite(self, x: int, y: int, sprite: Sprite) -> bool:
        if sprite.pixels is not None:
            sprite_height, sprite_width = sprite.pixels.shape
        else:
            sprite_height, sprite_width = (1, 1)
        obgidvqngh = sprite.x
        xzewyycdue = sprite.y
        zwzbcbwpyh = sprite.x + sprite_width
        khfytdaqqf = sprite.y + sprite_height
        if x < obgidvqngh:
            return False
        if x >= zwzbcbwpyh:
            return False
        if y < xzewyycdue:
            return False
        if y >= khfytdaqqf:
            return False
        return True

    def _check_win(self) -> bool:
        if not self.goals:
            return False
        if self.goal_spec is None:
            return False
        ifeuhzhcxr = self.goal_spec
        krdxxrzrlz = ifeuhzhcxr[0]
        gmjltxweel: list[tuple[str, int]]
        if isinstance(krdxxrzrlz, (list, tuple)):
            gmjltxweel = []
            for mvrnytgvvx, aqfhmtmxhi in ifeuhzhcxr:
                gmjltxweel.append((str(mvrnytgvvx), int(aqfhmtmxhi)))
        else:
            mvrnytgvvx = str(ifeuhzhcxr[0])
            aqfhmtmxhi = int(ifeuhzhcxr[1])
            gmjltxweel = [(mvrnytgvvx, aqfhmtmxhi)]
        nhefdqizsw: dict[int, int] = {}
        xbxnoponrg: dict[str, int] = {}
        for vgtpakybrc in self.fruits:
            umyncnjoxc, fbynmfmnkb = self._sprite_center(vgtpakybrc)
            naojwrjpdq = False
            for coitlkmvpm in self.goals:
                if self._point_in_sprite(umyncnjoxc, fbynmfmnkb, coitlkmvpm):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            thujeoyajx = self.fruit_level_map.get(vgtpakybrc, 0)
            nhefdqizsw[thujeoyajx] = nhefdqizsw.get(thujeoyajx, 0) + 1
        for vxxrprjnav in self.enemies:
            umyncnjoxc, fbynmfmnkb = self._sprite_center(vxxrprjnav)
            naojwrjpdq = False
            for coitlkmvpm in self.goals:
                if self._point_in_sprite(umyncnjoxc, fbynmfmnkb, coitlkmvpm):
                    naojwrjpdq = True
                    break
            if not naojwrjpdq:
                continue
            tjgvnkouuz = self.enemy_type_map.get(vxxrprjnav, ENEMY_TYPE_A)
            xbxnoponrg[tjgvnkouuz] = xbxnoponrg.get(tjgvnkouuz, 0) + 1
        for xecuihaefu, aqfhmtmxhi in gmjltxweel:
            if xecuihaefu in (ENEMY_TYPE_A, ENEMY_TYPE_B, ENEMY_TYPE_C):
                if xbxnoponrg.get(xecuihaefu, 0) != aqfhmtmxhi:
                    return False
            else:
                thujeoyajx = int(xecuihaefu)
                if nhefdqizsw.get(thujeoyajx, 0) != aqfhmtmxhi:
                    return False
        return True

    def _snapshot_state(self) -> None:
        juvlspuhfl: list[tuple[str, int, int, int]] = []
        for vgtpakybrc in self.fruits:
            kcacvefoot = self.fruit_level_map.get(vgtpakybrc, 0)
            juvlspuhfl.append(("fruit", int(kcacvefoot), int(vgtpakybrc.x), int(vgtpakybrc.y)))
        for vxxrprjnav in self.enemies:
            maybkcbmig = self.enemy_type_map.get(vxxrprjnav, ENEMY_TYPE_A)
            level = self._enemy_speed_for_type(maybkcbmig)
            juvlspuhfl.append(("enemy", int(level), int(vxxrprjnav.x), int(vxxrprjnav.y)))
        self.state_history.append(juvlspuhfl)

    def _undo_last_move(self) -> None:
        self._clear_vacuum_ring()
        if not self.state_history:
            return
        ivsyxrygbg = self.state_history.pop()
        for vgtpakybrc in list(self.fruits):
            self.current_level.remove_sprite(vgtpakybrc)
        self.fruits = []
        self.fruit_level_map = {}
        for vxxrprjnav in list(self.enemies):
            self._remove_enemy(vxxrprjnav)
        self.enemies = []
        self.enemy_type_map = {}
        for hzlnmxhwpt, xcuvllvbjg, x, y in ivsyxrygbg:
            if hzlnmxhwpt == "fruit":
                thujeoyajx = xcuvllvbjg
                if thujeoyajx < 0:
                    thujeoyajx = 0
                if thujeoyajx >= len(FRUIT_SPRITE_KEYS):
                    thujeoyajx = len(FRUIT_SPRITE_KEYS) - 1
                bwtdecdvlt = FRUIT_SPRITE_KEYS[thujeoyajx]
                cgmmpmplah = sprites[bwtdecdvlt]
                vgtpakybrc = cgmmpmplah.clone()
                vgtpakybrc.set_position(x, y)
                self.current_level.add_sprite(vgtpakybrc)
                self.fruits.append(vgtpakybrc)
                self.fruit_level_map[vgtpakybrc] = thujeoyajx
            elif hzlnmxhwpt == "enemy":
                cptzxsjeyh = xcuvllvbjg
                tjgvnkouuz = self._enemy_type_for_level(cptzxsjeyh)
                nvgwiuywco = self._enemy_sprite_for_type(tjgvnkouuz)
                vxxrprjnav = nvgwiuywco.clone()
                vxxrprjnav.set_position(x, y)
                self.current_level.add_sprite(vxxrprjnav)
                self.enemies.append(vxxrprjnav)
                self.enemy_type_map[vxxrprjnav] = tjgvnkouuz
        self._update_fruit_positions_cache()
        if self.grayed and self._has_enough_fruits():
            self.grayed = False
            for k in self.key_sprites:
                k.pixels = sprites[k.name].pixels.copy()
            for dmanzruhiz in self.goals:
                dmanzruhiz.pixels = sprites[dmanzruhiz.name].pixels.copy()

    def _fruit_value(self, thujeoyajx: int) -> int:
        if thujeoyajx < 0:
            thujeoyajx = 0
        return 1 << thujeoyajx

    def _has_enough_fruits(self) -> bool:
        if self.goal_spec is None:
            return False
        ifeuhzhcxr = self.goal_spec
        krdxxrzrlz = ifeuhzhcxr[0]
        lzepagmnxr: list[tuple[int, int]] = []
        if isinstance(krdxxrzrlz, (list, tuple)):
            for mvrnytgvvx, aqfhmtmxhi in ifeuhzhcxr:
                xecuihaefu = str(mvrnytgvvx)
                if xecuihaefu in (ENEMY_TYPE_A, ENEMY_TYPE_B, ENEMY_TYPE_C):
                    continue
                lzepagmnxr.append((int(xecuihaefu), int(aqfhmtmxhi)))
        else:
            mvrnytgvvx = ifeuhzhcxr[0]
            aqfhmtmxhi = ifeuhzhcxr[1]
            xecuihaefu = str(mvrnytgvvx)
            if xecuihaefu not in (ENEMY_TYPE_A, ENEMY_TYPE_B, ENEMY_TYPE_C):
                lzepagmnxr.append((int(xecuihaefu), int(aqfhmtmxhi)))
        if not lzepagmnxr:
            return True
        yaakzbpcpp: int = 0
        for thujeoyajx, vbcanllumj in lzepagmnxr:
            eqyvskhdxh = self._fruit_value(thujeoyajx)
            yaakzbpcpp += eqyvskhdxh * vbcanllumj
        tvuutcmlpj: int = 0
        for vgtpakybrc in self.fruits:
            thujeoyajx = self.fruit_level_map.get(vgtpakybrc, 0)
            value = self._fruit_value(thujeoyajx)
            tvuutcmlpj += value
        return tvuutcmlpj >= yaakzbpcpp

    def _get_valid_actions(self) -> list[ActionInput]:
        return self.actions

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

    def get_sprite_status(self) -> list[dict]:
        """




        """
        result = []
        for spr in self.current_level.get_sprites():
            if not spr.is_visible:
                continue
            tags     = list(spr._tags) if spr._tags else []
            label    = next(
                (TAG_DISPLAY_NAMES[t] for t in tags if t in TAG_DISPLAY_NAMES),
                spr.name,
            )
            label_l2 = next(
                (TAG_DISPLAY_NAMES_L2[t] for t in tags if t in TAG_DISPLAY_NAMES_L2),
                spr.name,
            )
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
            "name":     "step_counter",
            "label":    "Steps Remaining",
            "label_l2": ''            ,
            "value":    self.step_counter_ui.current_steps,
            "max":      self.step_counter_ui.max_steps,
            "x":        0, "y":      63,
            "width":    64, "height": 1,
        })
        return result

    def get_game_knowledge(self) -> dict:
        """

        """
        return GAME_KNOWLEDGE
