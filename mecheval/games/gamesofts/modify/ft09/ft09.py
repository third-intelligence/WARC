"""
































"""

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
    "orange_2x2": Sprite(
        pixels=[
            [12, 12],
            [12, 12],
        ],
        name="orange_2x2",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "corner_tl_2x2": Sprite(
        pixels=[
            [0, 14],
            [0, 0],
        ],
        name="corner_tl_2x2",
        visible=True,
        collidable=True,
        tags=["sample_tile"],
    ),
    "border_tile_blue_center": Sprite(
        pixels=[
            [2, 2, 0],
            [2, 15, 2],
            [2, 2, 2],
        ],
        name="border_tile_blue_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_gray_yellow": Sprite(
        pixels=[
            [3, 3, 3],
            [3, 11, 0],
            [3, 0, 2],
        ],
        name="border_tile_gray_yellow",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_red_center": Sprite(
        pixels=[
            [2, 0, 0],
            [0, 8, 2],
            [2, 0, 2],
        ],
        name="border_tile_red_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_gray_purple": Sprite(
        pixels=[
            [0, 3, 3],
            [2, 15, 3],
            [0, 2, 0],
        ],
        name="border_tile_gray_purple",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_blue2_center": Sprite(
        pixels=[
            [2, 0, 2],
            [2, 9, 2],
            [2, 2, 0],
        ],
        name="border_tile_blue2_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),

    "border_tile_green_center": Sprite(
        pixels=[
            [2, 0, 0],
            [3, 14, 2],
            [3, 3, 3],
        ],
        name="border_tile_green_center",
        visible=True,
        collidable=False,
        tags=["sample_tile", "border_tile"],
    ),

    "border_tile_green_center_b": Sprite(
        pixels=[
            [2, 0, 2],
            [0, 14, 0],
            [0, 0, 2],
        ],
        name="border_tile_green_center_b",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),

    "border_tile_green_center_c": Sprite(
        pixels=[
            [3, 3, 3],
            [2, 14, 3],
            [0, 0, 2],
        ],
        name="border_tile_green_center_c",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "purple_2x2": Sprite(
        pixels=[
            [15, 15],
            [15, 15],
        ],
        name="purple_2x2",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "border_tile_gray_yellow2": Sprite(
        pixels=[
            [3, 0, 3],
            [2, 11, 2],
            [3, 0, 3],
        ],
        name="border_tile_gray_yellow2",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "orange_2x2_b": Sprite(
        pixels=[
            [12, 12],
            [12, 12],
        ],
        name="orange_2x2_b",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "progress_dot_5w": Sprite(
        pixels=[
            [0, -1, 0, -1, 0],
        ],
        name="progress_dot_5w",
        visible=True,
        collidable=True,
        tags=["progress_dot"],
    ),
    "border_tile_green2_center": Sprite(
        pixels=[
            [2, 0, 2],
            [3, 14, 0],
            [3, 3, 3],
        ],
        name="border_tile_green2_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "decoration_magenta_dot": Sprite(
        pixels=[
            [-1, -1, 6, -1, -1],
            [-1, -1, -1, -1, -1],
            [6, -1, -1, -1, 6],
            [-1, -1, -1, -1, -1],
            [-1, -1, 6, -1, -1],
            [-1, -1, -1, -1, -1],
            [6, -1, -1, -1, 6],
            [-1, -1, -1, -1, -1],
            [-1, -1, 6, -1, -1],
        ],
        name="decoration_magenta_dot",
        visible=True,
        collidable=True,
        tags=["decoration"],
    ),
    "score_bg_18h16w": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        name="score_bg_18h16w",
        visible=True,
        collidable=False,
        layer=-2,
        tags=["score_bg"],
    ),
    "score_bg_16h32w": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        name="score_bg_16h32w",
        visible=True,
        collidable=False,
        layer=-2,
        tags=["score_bg"],
    ),
    "special_tile_light_blue_corner": Sprite(
        pixels=[
            [6, 7, 6],
            [7, 10, 7],
            [6, 7, 6],
        ],
        name="special_tile_light_blue_corner",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile"],
    ),
    "border_tile_red2_corner": Sprite(
        pixels=[
            [3, 0, 2],
            [3, 8, 0],
            [3, 3, 3],
        ],
        name="border_tile_red2_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_blue3_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [0, 9, 0],
            [0, 2, 0],
        ],
        name="border_tile_blue3_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_magenta_pink_corner": Sprite(
        pixels=[
            [6, 6, 6],
            [6, 7, 6],
            [6, 7, 6],
        ],
        name="tile_magenta_pink_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),

    "border_tile_orange_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [0, 12, 2],
            [0, 0, 0],
        ],
        name="border_tile_orange_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),

    "border_tile_orange_corner_b": Sprite(
        pixels=[
            [0, 2, 2],
            [2, 12, 2],
            [0, 0, 0],
        ],
        name="border_tile_orange_corner_b",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_pink_light_blue_corner": Sprite(
        pixels=[
            [7, 7, 6],
            [7, 10, 7],
            [6, 7, 7],
        ],
        name="tile_pink_light_blue_corner",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),
    "tile_magenta_corner_b": Sprite(
        pixels=[
            [6, 6, 6],
            [7, 6, 7],
            [6, 6, 6],
        ],
        name="tile_magenta_corner_b",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "border_tile_blue4_corner": Sprite(
        pixels=[
            [3, 0, 0],
            [2, 9, 2],
            [3, 0, 0],
        ],
        name="border_tile_blue4_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_magenta_corner_c": Sprite(
        pixels=[
            [6, 6, 6],
            [6, 7, 6],
            [6, 6, 7],
        ],
        name="tile_magenta_corner_c",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "border_tile_blue5_corner": Sprite(
        pixels=[
            [3, 3, 3],
            [3, 9, 2],
            [3, 0, 2],
        ],
        name="border_tile_blue5_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),

    "border_tile_orange2_corner": Sprite(
        pixels=[
            [0, 2, 2],
            [0, 12, 0],
            [0, 2, 0],
        ],
        name="border_tile_orange2_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),

    "border_tile_orange2b_corner": Sprite(
        pixels=[
            [0, 0, 0],
            [0, 12, 2],
            [2, 0, 2],
        ],
        name="border_tile_orange2b_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "blue_2x2": Sprite(
        pixels=[
            [9, 9],
            [9, 9],
        ],
        name="blue_2x2",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "blue_2x2_b": Sprite(
        pixels=[
            [9, 9],
            [9, 9],
        ],
        name="blue_2x2_b",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),

    "border_tile_purple2_corner": Sprite(
        pixels=[
            [3, 2, 0],
            [3, 15, 2],
            [3, 2, 0],
        ],
        name="border_tile_purple2_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_yellow_corner": Sprite(
        pixels=[
            [3, 3, 2],
            [0, 11, 0],
            [3, 2, 3],
        ],
        name="border_tile_yellow_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_pink_light_blue_corner_b": Sprite(
        pixels=[
            [7, 7, 6],
            [7, 10, 7],
            [7, 7, 7],
        ],
        name="tile_pink_light_blue_corner_b",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),
    "tile_magenta_cross": Sprite(
        pixels=[
            [6, 7, 7],
            [7, 10, 7],
            [7, 7, 6],
        ],
        name="tile_magenta_cross",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),
    "tile_magenta_corner_d": Sprite(
        pixels=[
            [6, 6, 6],
            [6, 7, 6],
            [6, 6, 6],
        ],
        name="tile_magenta_corner_d",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "tile_magenta_corner_e": Sprite(
        pixels=[
            [6, 7, 6],
            [7, 7, 7],
            [6, 7, 6],
        ],
        name="tile_magenta_corner_e",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "red_orange_2x4": Sprite(
        pixels=[
            [8, 8],
            [8, 8],
            [12, 12],
            [12, 12],
        ],
        name="red_orange_2x4",
        visible=True,
        collidable=True,
        tags=["redblue_pattern"],
    ),
    "border_tile_red3_corner": Sprite(
        pixels=[
            [3, 0, 0],
            [3, 12, 0],
            [3, 0, 2],
        ],
        name="border_tile_red3_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_red4_corner": Sprite(
        pixels=[
            [0, 0, 0],
            [2, 8, 2],
            [2, 2, 2],
        ],
        name="border_tile_red4_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_purple3_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [0, 15, 3],
            [3, 3, 3],
        ],
        name="border_tile_purple3_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_green3_corner": Sprite(
        pixels=[
            [0, 2, 0],
            [2, 14, 2],
            [0, 3, 0],
        ],
        name="border_tile_green3_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_magenta_corner_f": Sprite(
        pixels=[
            [6, 7, 6],
            [7, 7, 7],
            [6, 7, 6],
        ],
        name="tile_magenta_corner_f",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "border_tile_purple4_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [0, 15, 3],
            [3, 3, 3],
        ],
        name="border_tile_purple4_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "red_2x2": Sprite(
        pixels=[
            [8, 8],
            [8, 8],
        ],
        name="red_2x2",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),

    "border_tile_red5_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [0, 8, 2],
            [2, 0, 0],
        ],
        name="border_tile_red5_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "border_tile_red_cross": Sprite(

        pixels=[
            [2, 0, 2],
            [0, 8, 0],
            [2, 0, 2],
        ],
        name="border_tile_red_cross",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "normal_tile": Sprite(
        pixels=[
            [9, 9, 9],
            [9, 9, 9],
            [9, 9, 9],
        ],
        name="normal_tile",
        visible=True,
        collidable=True,
        tags=["color_tile", "normal_tile", "clickable_tile"],
    ),
    "tile_pink_light_blue_corner_c": Sprite(
        pixels=[
            [7, 6, 7],
            [6, 10, 6],
            [7, 6, 7],
        ],
        name="tile_pink_light_blue_corner_c",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),
    "orange_red_2x4": Sprite(
        pixels=[
            [12, 12],
            [12, 12],
            [8, 8],
            [8, 8],
        ],
        name="orange_red_2x4",
        visible=True,
        collidable=True,
        tags=["redblue_pattern"],
    ),
    "border_tile_blue6_corner": Sprite(
        pixels=[
            [0, 2, 2],
            [2, 9, 2],
            [0, 0, 2],
        ],
        name="border_tile_blue6_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "blue_2x2_c": Sprite(
        pixels=[
            [9, 9],
            [9, 9],
        ],
        name="blue_2x2_c",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "blue_2x2_d": Sprite(
        pixels=[
            [9, 9],
            [9, 9],
        ],
        name="blue_2x2_d",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "border_tile_purple5_corner": Sprite(
        pixels=[
            [2, 3, 2],
            [0, 14, 0],
            [2, 0, 2],
        ],
        name="border_tile_purple5_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_misc_a": Sprite(
        pixels=[
            [0, 2, 2],
            [0, 8, 0],
            [2, 2, 0],
        ],
        name="tile_misc_a",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "border_tile_red6_corner": Sprite(
        pixels=[
            [0, 0, 2],
            [0, 14, 0],
            [3, 3, 3],
        ],
        name="border_tile_red6_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "light_blue_2x2": Sprite(
        pixels=[
            [10, 10],
            [10, 10],
        ],
        name="light_blue_2x2",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "border_tile_blue7_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [0, 9, 2],
            [0, 2, 0],
        ],
        name="border_tile_blue7_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_red7_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [2, 8, 2],
            [2, 2, 2],
        ],
        name="border_tile_red7_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "decoration_multi_dot_25h17w": Sprite(
        pixels=[
            [-1, -1, -1, -1, 15, -1, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [10, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [15, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 15, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1, 10, -1, -1, -1, 14],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1, 10],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1, -1],
        ],
        name="decoration_multi_dot_25h17w",
        visible=True,
        collidable=True,
        tags=["decoration"],
    ),
    "border_tile_blue8_corner": Sprite(
        pixels=[
            [3, 0, 2],
            [3, 9, 0],
            [3, 3, 3],
        ],
        name="border_tile_blue8_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "progress_single_1x1": Sprite(
        pixels=[
            [2],
        ],
        name="progress_single_1x1",
        visible=True,
        collidable=True,
        tags=["progress_dot"],
    ),
    "border_frame_15x15": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ],
        name="border_frame_15x15",
        visible=True,
        collidable=False,
        tags=["border_frame"],
    ),

    "border_tile_green4_corner": Sprite(
        pixels=[
            [2, 0, 3],
            [0, 14, 3],
            [2, 0, 3],
        ],
        name="border_tile_green4_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),

    "border_tile_green4b_corner": Sprite(
        pixels=[
            [3, 3, 3],
            [3, 14, 0],
            [3, 0, 2],
        ],
        name="border_tile_green4b_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_blue9_corner": Sprite(
        pixels=[
            [0, 2, 0],
            [0, 8, 2],
            [2, 0, 2],
        ],
        name="border_tile_blue9_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_blue10_corner": Sprite(
        pixels=[
            [0, 0, 2],
            [2, 8, 2],
            [2, 2, 2],
        ],
        name="border_tile_blue10_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_yellow2_corner": Sprite(
        pixels=[
            [3, 0, 3],
            [0, 11, 2],
            [3, 2, 3],
        ],
        name="border_tile_yellow2_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "blue_red_2x4": Sprite(
        pixels=[
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8],
            [8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8],
            [8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
        ],
        name="blue_red_2x4",
        visible=True,
        collidable=False,
        tags=["redblue_pattern"],
    ),
    "progress_dot_3w": Sprite(
        pixels=[
            [2, -1, 2],
        ],
        name="progress_dot_3w",
        visible=True,
        collidable=True,
        tags=["progress_dot"],
    ),
    "border_tile_red8_corner": Sprite(
        pixels=[
            [0, 2, 2],
            [0, 8, 0],
            [0, 2, 2],
        ],
        name="border_tile_red8_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),

    "border_tile_green5_corner": Sprite(
        pixels=[
            [2, 0, 3],
            [0, 14, 3],
            [3, 3, 3],
        ],
        name="border_tile_green5_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_yellow3_corner": Sprite(
        pixels=[
            [3, 0, 3],
            [0, 11, 2],
        ],
        name="border_tile_yellow3_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_frame_outer_15x15": Sprite(
        pixels=[
            [2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2],
            [2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 2, 2],
        ],
        name="border_frame_outer_15x15",
        visible=True,
        collidable=False,
        tags=["border_frame_outer", "border_frame"],
    ),
    "tile_magenta_corner_g": Sprite(
        pixels=[
            [7, 7, 7],
            [6, 7, 6],
            [6, 6, 6],
        ],
        name="tile_magenta_corner_g",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "progress_black_1x1": Sprite(
        pixels=[
            [5],
        ],
        name="progress_black_1x1",
        visible=True,
        collidable=True,
        tags=["progress_dot"],
    ),
    "border_tile_orange3_corner": Sprite(
        pixels=[
            [3, 0, 2],
            [2, 12, 0],
            [2, 2, 0],
        ],
        name="border_tile_orange3_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),

    "border_tile_yellow4_corner": Sprite(
        pixels=[
            [2, 0, 0],
            [0, 14, 0],
            [2, 0, 2],
        ],
        name="border_tile_yellow4_corner",
        visible=True,
        collidable=False,
        tags=["variation_pattern", "border_tile"],
    ),
    "progress_dot_3w_b": Sprite(
        pixels=[
            [6, -1, 6],
        ],
        name="progress_dot_3w_b",
        visible=True,
        collidable=True,
        tags=["progress_dot"],
    ),
    "progress_yellow_1x1": Sprite(
        pixels=[
            [11],
        ],
        name="progress_yellow_1x1",
        visible=True,
        collidable=True,
        tags=["progress_dot"],
    ),
    "border_tile_red9_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [0, 8, 2],
            [2, 0, 2],
        ],
        name="border_tile_red9_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "progress_white_1x1": Sprite(
        pixels=[
            [0],
        ],
        name="progress_white_1x1",
        visible=True,
        collidable=True,
        tags=["progress_dot"],
    ),
    "border_tile_red10_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 8, 0],
            [2, 2, 0],
        ],
        name="border_tile_red10_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_gray2_center": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        name="border_tile_gray2_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_red11_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 8, 2],
            [0, 2, 2],
        ],
        name="border_tile_red11_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_orange4_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 8, 2],
            [0, 2, 0],
        ],
        name="border_tile_orange4_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "yellow_2x2": Sprite(
        pixels=[
            [11, 11],
            [11, 11],
        ],
        name="yellow_2x2",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "border_tile_blue11_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [2, 12, 2],
            [2, 0, 2],
        ],
        name="border_tile_blue11_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_green6_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [2, 9, 2],
            [2, 2, 2],
        ],
        name="border_tile_green6_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_gray3_center": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        name="border_tile_gray3_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_yellow5_corner": Sprite(
        pixels=[
            [3, 0, 3],
            [2, 11, 2],
            [3, 0, 3],
        ],
        name="border_tile_yellow5_corner",
        visible=True,
        collidable=False,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_orange5_corner": Sprite(
        pixels=[
            [3, 3, 3],
            [3, 12, 0],
            [2, 2, 3],
        ],
        name="border_tile_orange5_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_blue12_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [2, 9, 2],
            [2, 2, 2],
        ],
        name="border_tile_blue12_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_blue13_corner": Sprite(
        pixels=[
            [2, 0, 0],
            [2, 9, 2],
            [2, 2, 2],
        ],
        name="border_tile_blue13_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "red_2x2_b": Sprite(
        pixels=[
            [8, 8],
            [8, 8],
        ],
        name="red_2x2_b",
        visible=True,
        collidable=True,
        tags=["correct_pattern"],
    ),
    "blue_red_pattern_11x11": Sprite(
        pixels=[
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [9, 9, 9, -1, 8, 8, 8, -1, 9, 9, 9],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 8, 8, -1, -1, -1, -1, -1, 9, 9, 9],
            [8, 8, 8, -1, -1, -1, -1, -1, 9, 9, 9],
            [8, 8, 8, -1, -1, -1, -1, -1, 9, 9, 9],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 9, 9, -1, 8, 8, 8, -1, 8, 8, 8],
            [9, 9, 9, -1, 8, 8, 8, -1, 8, 8, 8],
            [9, 9, 9, -1, 8, 8, 8, -1, 8, 8, 8],
        ],
        name="blue_red_pattern_11x11",
        visible=True,
        collidable=False,
        tags=["redblue_pattern"],
    ),
    "red_blue_pattern_11x11": Sprite(

        pixels=[
            [8, 8, 8, -1, 9, 9, 9, -1, 9, 9, 9],
            [8, 8, 8, -1, 9, 9, 9, -1, 9, 9, 9],
            [8, 8, 8, -1, 9, 9, 9, -1, 9, 9, 9],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8],
            [8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8],
            [8, 8, 8, -1, -1, -1, -1, -1, 8, 8, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [9, 9, 9, -1, 9, 9, 9, -1, 8, 8, 8],
            [9, 9, 9, -1, 9, 9, 9, -1, 8, 8, 8],
            [9, 9, 9, -1, 9, 9, 9, -1, 8, 8, 8],
        ],
        name="red_blue_pattern_11x11",
        visible=True,
        collidable=False,
        tags=["redblue_pattern"],
    ),
    "border_tile_orange6_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 12, 2],
            [2, 0, 2],
        ],
        name="border_tile_orange6_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_red12_corner": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 12, 2],
            [2, 0, 2],
        ],
        name="border_tile_red12_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_magenta_corner_h": Sprite(
        pixels=[
            [6, 6, 6],
            [7, 7, 7],
            [6, 6, 6],
        ],
        name="tile_magenta_corner_h",
        visible=True,
        collidable=True,
        tags=["variation_pattern"],
    ),
    "border_tile_yellow6_corner": Sprite(
        pixels=[
            [3, 2, 3],
            [0, 11, 0],
            [3, 2, 3],
        ],
        name="border_tile_yellow6_corner",
        visible=True,
        collidable=False,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_green7_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [0, 14, 0],
            [2, 0, 2],
        ],
        name="border_tile_green7_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "green_purple_2x4": Sprite(
        pixels=[
            [14, 14],
            [14, 14],
            [15, 15],
            [15, 15],
        ],
        name="green_purple_2x4",
        visible=True,
        collidable=True,
        tags=["redblue_pattern"],
    ),
    "border_tile_blue14_center": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 9, 2],
            [2, 2, 2],
        ],
        name="border_tile_blue14_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_green8_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [0, 14, 0],
            [2, 0, 2],
        ],
        name="border_tile_green8_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_blue15_center": Sprite(
        pixels=[
            [2, 2, 2],
            [2, 9, 2],
            [2, 2, 2],
        ],
        name="border_tile_blue15_center",
        visible=True,
        collidable=True,
        tags=["sample_tile", "border_tile"],
    ),
    "border_tile_orange7_corner": Sprite(
        pixels=[
            [3, 0, 3],
            [2, 11, 3],
            [3, 0, 3],
        ],
        name="border_tile_orange7_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_green9_corner": Sprite(
        pixels=[
            [3, 2, 3],
            [0, 14, 0],
            [3, 2, 3],
        ],
        name="border_tile_green9_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_pink_light_blue_corner_d": Sprite(
        pixels=[
            [7, 7, 6],
            [7, 10, 7],
            [7, 7, 7],
        ],
        name="tile_pink_light_blue_corner_d",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),
    "tile_light_blue_center": Sprite(
        pixels=[
            [7, 6, 7],
            [7, 10, 7],
            [7, 7, 7],
        ],
        name="tile_light_blue_center",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),
    "tile_magenta_corner_i": Sprite(
        pixels=[
            [6, 7, 7],
            [7, 10, 7],
            [7, 7, 6],
        ],
        name="tile_magenta_corner_i",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),

    "border_tile_red13_corner": Sprite(
        pixels=[
            [0, 2, 0],
            [2, 12, 2],
            [0, 0, 2],
        ],
        name="border_tile_red13_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),

    "border_tile_red13b_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [2, 8, 0],
            [0, 0, 2],
        ],
        name="border_tile_red13b_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "border_tile_green10_corner": Sprite(
        pixels=[
            [0, 2, 3],
            [2, 12, 3],
            [3, 3, 3],
        ],
        name="border_tile_green10_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
    "tile_light_blue_center_b": Sprite(
        pixels=[
            [6, 7, 7],
            [7, 10, 7],
            [7, 7, 7],
        ],
        name="tile_light_blue_center_b",
        visible=True,
        collidable=True,
        tags=["redblue_pattern", "clickable_tile", "special_tile"],
    ),
    "border_tile_red14_corner": Sprite(
        pixels=[
            [2, 0, 2],
            [2, 9, 2],
            [2, 2, 2],
        ],
        name="border_tile_red14_corner",
        visible=True,
        collidable=True,
        tags=["variation_pattern", "border_tile"],
    ),
}



levels = [

    Level(
        sprites=[
            sprites["score_bg_18h16w"].clone().set_position(0, 14),
            sprites["score_bg_16h32w"].clone(),
            sprites["normal_tile"].clone().set_position(18, 18),
            sprites["normal_tile"].clone().set_position(22, 18),
            sprites["normal_tile"].clone().set_position(26, 18),
            sprites["normal_tile"].clone().set_position(18, 22),
            sprites["normal_tile"].clone().set_position(26, 22),
            sprites["normal_tile"].clone().set_position(18, 26),
            sprites["normal_tile"].clone().set_position(22, 26),
            sprites["normal_tile"].clone().set_position(26, 26),
            sprites["border_tile_red_cross"].clone().set_position(23, 5),
            sprites["tile_misc_a"].clone().set_position(6, 22),
            sprites["border_frame_15x15"].clone().set_position(16, 16),
            sprites["border_tile_red5_corner"].clone().set_position(6, 5),
            sprites["blue_red_2x4"].clone().set_position(19, 1),
            sprites["border_frame_outer_15x15"].clone().set_position(16, 16),
            sprites["red_blue_pattern_11x11"].clone().set_position(2, 18),
            sprites["blue_red_pattern_11x11"].clone().set_position(2, 1),
            sprites["border_tile_red8_corner"].clone().set_position(22, 22),
        ],
        grid_size=(32, 32),
        data={
            "click_budget": 32,
            "color_cycle": [9, 8],
            "flip_pattern": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        },
        name="THR",
    ),

    Level(
        sprites=[
            sprites["border_tile_red13_corner"].clone().set_position(14, 19),
            sprites["normal_tile"].clone().set_position(10, 7),
            sprites["normal_tile"].clone().set_position(14, 7),
            sprites["normal_tile"].clone().set_position(18, 7),
            sprites["normal_tile"].clone().set_position(10, 11),
            sprites["normal_tile"].clone().set_position(18, 11),
            sprites["normal_tile"].clone().set_position(10, 15),
            sprites["normal_tile"].clone().set_position(14, 15),
            sprites["normal_tile"].clone().set_position(18, 15),
            sprites["normal_tile"].clone().set_position(18, 19),
            sprites["normal_tile"].clone().set_position(14, 23),
            sprites["normal_tile"].clone().set_position(10, 23),
            sprites["normal_tile"].clone().set_position(10, 19),
            sprites["normal_tile"].clone().set_position(18, 23),
            sprites["border_tile_orange2_corner"].clone().set_position(14, 11),
            sprites["blue_2x2_c"].clone().set_position(30, 0),
            sprites["orange_2x2"].clone().set_position(30, 2),
        ],
        grid_size=(32, 32),
        data={
            "click_budget": 32,
            "color_cycle": [9, 12],
            "flip_pattern": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        },
        name="hxv",
    ),

    Level(
        sprites=[
            sprites["border_tile_red_center"].clone().set_position(18, 14),
            sprites["border_tile_orange_corner"].clone().set_position(14, 22),
            sprites["red_orange_2x4"].clone().set_position(30, 0),
            sprites["normal_tile"].clone().set_position(10, 2),
            sprites["normal_tile"].clone().set_position(10, 6),
            sprites["normal_tile"].clone().set_position(10, 10),
            sprites["normal_tile"].clone().set_position(14, 10),
            sprites["normal_tile"].clone().set_position(14, 14),
            sprites["normal_tile"].clone().set_position(18, 10),
            sprites["normal_tile"].clone().set_position(18, 6),
            sprites["normal_tile"].clone().set_position(22, 10),
            sprites["normal_tile"].clone().set_position(18, 2),
            sprites["normal_tile"].clone().set_position(14, 2),
            sprites["normal_tile"].clone().set_position(6, 14),
            sprites["normal_tile"].clone().set_position(6, 10),
            sprites["normal_tile"].clone().set_position(10, 18),
            sprites["normal_tile"].clone().set_position(6, 18),
            sprites["normal_tile"].clone().set_position(14, 18),
            sprites["normal_tile"].clone().set_position(18, 18),
            sprites["normal_tile"].clone().set_position(22, 18),
            sprites["normal_tile"].clone().set_position(22, 14),
            sprites["normal_tile"].clone().set_position(10, 22),
            sprites["normal_tile"].clone().set_position(18, 22),
            sprites["normal_tile"].clone().set_position(14, 26),
            sprites["normal_tile"].clone().set_position(10, 26),
            sprites["normal_tile"].clone().set_position(18, 26),
            sprites["border_tile_orange2b_corner"].clone().set_position(14, 6),
            sprites["border_tile_red13b_corner"].clone().set_position(10, 14),
        ],
        grid_size=(32, 32),
        data={
            "click_budget": 96,
            "color_cycle": [8, 12],
            "flip_pattern": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        },
        name="Fmh",
    ),

    Level(
        sprites=[
            sprites["border_tile_blue2_center"].clone().set_position(18, 11),
            sprites["orange_2x2"].clone().set_position(30, 4),
            sprites["normal_tile"].clone().set_position(10, 7),
            sprites["normal_tile"].clone().set_position(14, 11),
            sprites["normal_tile"].clone().set_position(14, 15),
            sprites["normal_tile"].clone().set_position(22, 11),
            sprites["normal_tile"].clone().set_position(22, 15),
            sprites["normal_tile"].clone().set_position(18, 7),
            sprites["normal_tile"].clone().set_position(14, 7),
            sprites["normal_tile"].clone().set_position(6, 11),
            sprites["normal_tile"].clone().set_position(10, 15),
            sprites["normal_tile"].clone().set_position(18, 15),
            sprites["normal_tile"].clone().set_position(6, 15),
            sprites["normal_tile"].clone().set_position(6, 7),
            sprites["normal_tile"].clone().set_position(22, 7),
            sprites["normal_tile"].clone().set_position(10, 19),
            sprites["normal_tile"].clone().set_position(18, 19),
            sprites["normal_tile"].clone().set_position(10, 23),
            sprites["normal_tile"].clone().set_position(14, 23),
            sprites["normal_tile"].clone().set_position(18, 23),
            sprites["border_tile_blue11_corner"].clone().set_position(10, 11),
            sprites["red_2x2"].clone().set_position(30, 2),
            sprites["blue_2x2"].clone().set_position(30, 0),
            sprites["border_tile_orange_corner_b"].clone().set_position(14, 19),
        ],
        grid_size=(32, 32),
        data={
            "click_budget": 96,
            "color_cycle": [9, 8, 12],
            "flip_pattern": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        },
        name="oea",
    ),

    Level(
        sprites=[
            sprites["border_tile_gray_purple"].clone().set_position(19, 6),
            sprites["border_tile_green3_corner"].clone().set_position(11, 18),
            sprites["normal_tile"].clone().set_position(7, 6),
            sprites["normal_tile"].clone().set_position(15, 18),
            sprites["normal_tile"].clone().set_position(15, 6),
            sprites["normal_tile"].clone().set_position(11, 10),
            sprites["normal_tile"].clone().set_position(11, 2),
            sprites["normal_tile"].clone().set_position(23, 14),
            sprites["normal_tile"].clone().set_position(15, 10),
            sprites["normal_tile"].clone().set_position(19, 18),
            sprites["normal_tile"].clone().set_position(23, 22),
            sprites["normal_tile"].clone().set_position(15, 22),
            sprites["normal_tile"].clone().set_position(19, 26),
            sprites["normal_tile"].clone().set_position(15, 26),
            sprites["normal_tile"].clone().set_position(23, 18),
            sprites["normal_tile"].clone().set_position(19, 10),
            sprites["normal_tile"].clone().set_position(7, 10),
            sprites["normal_tile"].clone().set_position(15, 2),
            sprites["normal_tile"].clone().set_position(7, 18),
            sprites["normal_tile"].clone().set_position(7, 22),
            sprites["normal_tile"].clone().set_position(7, 26),
            sprites["normal_tile"].clone().set_position(11, 26),
            sprites["normal_tile"].clone().set_position(7, 14),
            sprites["normal_tile"].clone().set_position(3, 10),
            sprites["normal_tile"].clone().set_position(3, 18),
            sprites["normal_tile"].clone().set_position(15, 14),
            sprites["normal_tile"].clone().set_position(23, 10),
            sprites["normal_tile"].clone().set_position(27, 18),
            sprites["normal_tile"].clone().set_position(27, 10),
            sprites["border_tile_purple2_corner"].clone().set_position(3, 14),
            sprites["tile_pink_light_blue_corner_c"].clone().set_position(11, 6),
            sprites["tile_pink_light_blue_corner_c"].clone().set_position(19, 22),
            sprites["tile_pink_light_blue_corner_c"].clone().set_position(11, 14),
            sprites["border_tile_green4b_corner"].clone().set_position(7, 2),
            sprites["border_tile_green5_corner"].clone().set_position(23, 26),
            sprites["border_tile_purple5_corner"].clone().set_position(11, 22),
            sprites["border_tile_green4_corner"].clone().set_position(27, 14),
            sprites["green_purple_2x4"].clone().set_position(27, 0),
            sprites["border_tile_green7_corner"].clone().set_position(19, 14),
        ],
        grid_size=(32, 32),
        data={
            "click_budget": 128,
            "color_cycle": [14, 15],
            "flip_pattern": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        },
        name="INW",
    ),

    Level(
        sprites=[
            sprites["border_tile_green_center"].clone().set_position(22, 23),
            sprites["green_purple_2x4"].clone().set_position(30, 2) if False else sprites["orange_2x2"].clone().set_position(30, 2),
            sprites["border_tile_yellow4_corner"].clone().set_position(10, 15),
            sprites["border_tile_green_center_b"].clone().set_position(18, 11),
            sprites["border_tile_green_center_c"].clone().set_position(6, 3),
            sprites["yellow_2x2"].clone().set_position(30, 0),
            sprites["tile_light_blue_center"].clone().set_position(6, 7),
            sprites["tile_light_blue_center"].clone().set_position(10, 7),
            sprites["tile_light_blue_center"].clone().set_position(14, 7),
            sprites["tile_light_blue_center"].clone().set_position(18, 7),
            sprites["tile_light_blue_center"].clone().set_position(22, 7),
            sprites["tile_light_blue_center"].clone().set_position(6, 11),
            sprites["tile_light_blue_center"].clone().set_position(22, 11),
            sprites["tile_light_blue_center"].clone().set_position(10, 11),
            sprites["tile_light_blue_center"].clone().set_position(14, 11),
            sprites["tile_light_blue_center"].clone().set_position(18, 15),
            sprites["tile_light_blue_center"].clone().set_position(6, 15),
            sprites["tile_light_blue_center"].clone().set_position(22, 15),
            sprites["tile_light_blue_center"].clone().set_position(14, 15),
            sprites["tile_light_blue_center"].clone().set_position(18, 19),
            sprites["tile_light_blue_center"].clone().set_position(6, 19),
            sprites["tile_light_blue_center"].clone().set_position(22, 19),
            sprites["tile_light_blue_center"].clone().set_position(10, 19),
            sprites["tile_light_blue_center"].clone().set_position(14, 19),
            sprites["tile_light_blue_center"].clone().set_position(26, 19),
            sprites["tile_light_blue_center"].clone().set_position(2, 7),
            sprites["tile_light_blue_center"].clone().set_position(2, 3),
            sprites["tile_light_blue_center"].clone().set_position(26, 23),
        ],
        grid_size=(32, 32),
        data={
            "click_budget": 128,
            "color_cycle": [11, 14],
            "flip_pattern": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        },
        name="DFx",
    ),
]

BACKGROUND_COLOR = 4
PADDING_COLOR = 4




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







    "score_bg":           "Score Background",
    "border_frame_outer": "Outer Border Frame",
    "border_frame":       "Border Frame",
    "sample_tile":        "Variation Pattern",
    "variation_pattern":  "Variation Pattern",
    "correct_pattern":    "Change Rule",
    "redblue_pattern":    "Color Tile",
    "color_tile":         "Color Tile",
    "decoration":         "Decoration",
    "progress_dot":       "Progress Dot",

    "normal_tile":    "Normal Tile",
    "special_tile":   "Special Tile",
    "border_tile":    "Border Tile",
    "clickable_tile": "Clickable Tile",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "score_bg":           "",
    "border_frame_outer": "",
    "border_frame":       "",
    "sample_tile":        "",
    "variation_pattern":  "",
    "correct_pattern":    "",
    "redblue_pattern":    "",
    "color_tile":         "",
    "decoration":         "",
    "progress_dot":       "",
    "normal_tile":    "",
    "special_tile":   "",
    "border_tile":    "",
    "clickable_tile": "",
}


def _build_sprite_display_map() -> dict:
    """








    """

    PREFERRED_TAGS = (
        "score_bg", "border_frame_outer", "border_frame",
        "sample_tile", "variation_pattern", "correct_pattern",
        "redblue_pattern", "color_tile", "decoration", "progress_dot",
    )

    EXCLUDE_TAGS = {"sys_click", "clickable_tile", "border_tile", "special_tile", "normal_tile"}







    _LV1_SAMPLE_TILES = {
        "blue_red_2x4",
        "red_blue_pattern_11x11",
        "blue_red_pattern_11x11",
    }

    out: dict = {}
    for spr_name, spr in sprites.items():
        target = label_en = label_l2 = spr_name

        if spr_name in _LV1_SAMPLE_TILES:
            target   = "sample_tile"
            label_en = "Sample Tile"
            label_l2 = ""
            out[spr_name] = (target, label_en, label_l2)
            continue
        tags = list(spr.tags or [])

        chosen = None
        for pt in PREFERRED_TAGS:
            if pt in tags:
                chosen = pt
                break

        if chosen is None:
            for t in tags:
                if t in EXCLUDE_TAGS:
                    continue
                if t in TAG_DISPLAY_NAMES:
                    chosen = t
                    break
        if chosen is not None and chosen in TAG_DISPLAY_NAMES:
            label_en = TAG_DISPLAY_NAMES[chosen]
            label_l2 = TAG_DISPLAY_NAMES_L2.get(chosen, label_en)







            if chosen == "sample_tile":
                target   = "variation_pattern"
                label_en = TAG_DISPLAY_NAMES["variation_pattern"]
                label_l2 = TAG_DISPLAY_NAMES_L2.get("variation_pattern", label_en)
            elif chosen == "correct_pattern":
                target   = "change_rule"
            elif chosen == "redblue_pattern":
                if "_2x4" in spr_name or "_2x2" in spr_name:

                    target   = "change_rule"
                    label_en = TAG_DISPLAY_NAMES["correct_pattern"]
                    label_l2 = TAG_DISPLAY_NAMES_L2.get("correct_pattern", label_en)
                else:

                    target   = "color_tile"
                    label_en = TAG_DISPLAY_NAMES["color_tile"]
                    label_l2 = TAG_DISPLAY_NAMES_L2.get("color_tile", label_en)
            else:
                target = chosen
        out[spr_name] = (target, label_en, label_l2)
    return out



SPRITE_DISPLAY: dict = _build_sprite_display_map()




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
    click_budget: int | None = None
    try:
        click_budget = level.get_data("click_budget")
    except Exception:
        pass

    color_cycle: list | None = None
    try:
        color_cycle = level.get_data("color_cycle")
    except Exception:
        pass

    flip_pattern: list | None = None
    try:
        flip_pattern = level.get_data("flip_pattern")
    except Exception:
        pass

    grid = getattr(level, "grid_size", (32, 32)) or (32, 32)

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

    _analyses = [
        {
            "gimmicks":     ["Tutorial: single-cell flip pattern, 2-color cycle"],
            "gimmicks_l2":  [''                                                                  ],
            "subgoals":     ["Click tiles to cycle their color so all border tiles' constraints are satisfied"],
            "subgoals_l2":  [''                                                                                                ],
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Small grid, 2-color cycle, sparse border tiles"],
            "gimmicks_l2":  [''                                                                ],
            "subgoals":     ["Cycle tile colors to match border constraints with limited clicks"],
            "subgoals_l2":  [''                                                                                          ],
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Medium grid, 2-color cycle, multiple border tiles"],
            "gimmicks_l2":  [''                                                                ],
            "subgoals":     ["Coordinate multiple tile flips to satisfy all border tile conditions"],
            "subgoals_l2":  [''                                                                                          ],
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["3-color cycle introduces more state combinations"],
            "gimmicks_l2":  [''                                                       ],
            "subgoals":     ["Plan flip sequences across a 3-color cycle to satisfy border constraints"],
            "subgoals_l2":  [''                                                                                  ],
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Large grid with special tiles, 2-color cycle"],
            "gimmicks_l2":  [''                                                          ],
            "subgoals":     ["Manage special tile flip patterns across a large grid"],
            "subgoals_l2":  [''                                                                           ],
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Large grid, mixed tile types, 2-color cycle"],
            "gimmicks_l2":  [''                                                             ],
            "subgoals":     ["Solve with mixed normal and special tiles across large grid"],
            "subgoals_l2":  [''                                                                           ],
            "difficulty":   "hard",
        },
    ]

    analysis = _analyses[level_index] if level_index < len(_analyses) else {}

    return {
        "level_index":   level_index + 1,
        "level_name":    getattr(level, "name", f"level_{level_index+1}"),
        "grid_size":     list(grid),
        "click_budget":  click_budget,
        "color_cycle":   color_cycle,
        "flip_pattern":  flip_pattern,
        "sprites":       level_sprites_info,
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
        "game_id":          "ft09",
        "title":            "Flip Tiles 09",
        "title_l2":         ''                          ,
        "description":      "A tile-flipping puzzle where clicking a tile cycles its color (and neighboring tiles per a flip pattern). Satisfy all border tile constraints before running out of clicks.",
        "description_l2":   ''                                                                                                                                                                                                   ,
        "default_grid_size": [64, 64],
        "num_levels":        len(levels),

        "goal":             "Cycle tile colors so every border_tile's center color matches the required relationship with its neighboring tiles.",
        "goal_l2":          ''                                                                                                                                       ,
        "win_condition":    "_check_win() returns True when all border tiles pass the neighbor-color constraint check.",
        "win_condition_l2": ''                                                                                                                   ,
        "lose_condition":   "Click budget reaches 0 (no clicks remaining).",
        "lose_condition_l2":''                                                          ,

        "actions": {
            "ACTION6": {
                "key":             "click",
                "effect":          "Click on a normal_tile or special_tile to advance its color (and neighbors per flip_pattern) to the next in color_cycle.",
                "effect_l2":       ''                                                                                                                                                             ,
                "restrictions":    "Only tiles with 'clickable_tile' tag can be clicked. Each click costs 1 from click_budget.",
                "restrictions_l2": ''                                                                                                           ,
                "step_cost":       1,
            },
        },

        "sprite_roles": {
            "normal_tile":    "3x3 blue tile grid cell. Clicking it cycles the center and neighboring tiles' colors per flip_pattern.",
            "normal_tile_l2": ''                                                                                                                                          ,
            "special_tile":   "3x3 tile with a distinct magenta/pink/light_blue pattern. Clicking applies its own flip pattern based on magenta pixel positions.",
            "special_tile_l2":''                                                                                                                                                                                      ,
            "border_tile":    "3x3 constraint tile (not clickable in most cases). Its center pixel color indicates what the neighboring tiles must equal or differ from.",
            "border_tile_l2": ''                                                                                                                                  ,
            "border_frame":   "Visual frame sprite shown in level 0. Flashes when an invalid tile is clicked.",
            "border_frame_l2":''                                                                                                                      ,
            "clickable_tile": "Meta-tag on tiles that the player can click.",
            "clickable_tile_l2":''                                                                        ,
        },

        "mechanics": {
            "color_cycling":         "Each tile's center pixel cycles through color_cycle on click. The cycle wraps around.",
            "color_cycling_l2":      ''                                                                                                                                ,
            "flip_pattern":          "A 3x3 matrix (1=affected, 0=not). When a tile is clicked, all neighbors at (dx*4, dy*4) offset where flip_pattern[dy+1][dx+1]==1 also get cycled.",
            "flip_pattern_l2":       ''                                                                                                                                                                              ,
            "special_tile_flip":     "Special tiles (NTi) use their own magenta pixel pattern instead of the level flip_pattern.",
            "special_tile_flip_l2":  ''                                                                                                                             ,
            "win_check":             "For each border_tile: its center color (pixels[1][1]) must match/mismatch neighbor tiles according to pixels[row][col]==0 (match) or !=0 (mismatch).",
            "win_check_l2":          ''                                                                                                                                                                        ,
            "click_budget":          "Each level has a limited click budget (kCv). Every tile click costs 1. Budget reaching 0 triggers game over.",
            "click_budget_l2":       ''                                                                                                                                        ,
            "blink_animation":       "In level 0, clicking a non-border_tile region triggers a 4-frame blink animation on the border_frame sprite.",
            "blink_animation_l2":    ''                                                                                                                                                                       ,
        },
    },


    "levels": [
        _build_level_knowledge(i, lv)
        for i, lv in enumerate(levels)
    ],
}




class ClickBudgetBar(RenderableUserDisplay):
    """



    """

    def __init__(self, max_budget: int, game: "Ft09") -> None:
        self.max_budget = max_budget
        self.current = max_budget
        self.game = game

    def set_budget(self, budget: int) -> None:
        """"""
        self.current = max(0, min(budget, self.max_budget))

    def consume(self) -> bool:
        """"""
        if self.current > 0:
            self.current -= 1
        return self.current > 0

    def reset(self) -> None:
        """"""
        self.current = self.max_budget

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """"""
        if self.max_budget == 0:
            return frame

        ratio = self.current / self.max_budget
        filled = round(64 * ratio)
        for x in range(64):

            frame[63, x] = ORANGE if x < filled else YELLOW

        return frame




class Ft09(ARCBaseGame):
    """












    """


    def __init__(self) -> None:

        initial_budget = levels[0].get_data("click_budget") if levels else 0
        self.budget_bar = ClickBudgetBar(initial_budget or 0, self)



        self._log_state = ActionLogState(aggregator=default_aggregator, post_process=compose_default_summary)



        self._frame_sprite_log:     list[list[dict]] = []
        self._frame_nonsprite_log:  list[list[dict]] = []

        super().__init__(
            "ft09",
            levels,
            Camera(0, 0, 16, 16, 4, 4, [self.budget_bar]),
            available_actions=[6],
        )



    def _reset_budget(self) -> None:
        """"""
        budget = self.current_level.get_data("click_budget")
        if budget:
            self.budget_bar.max_budget = budget
            self.budget_bar.reset()

    def on_set_level(self, level: Level) -> None:
        """"""
        self._reset_budget()


        self.blink_counter = 0
        self.blink_target = None


        if self.level_index == 0:
            border_frames = self.current_level.get_sprites_by_tag("border_frame")
            if border_frames:
                self.blink_target = border_frames[0]


        grid_size = self.current_level.grid_size
        self.grid_width  = grid_size[0]
        self.grid_height = grid_size[1]


        self.border_sprites  = self.current_level.get_sprites_by_tag("border_tile")
        self.tile_sprites    = self.current_level.get_sprites_by_tag("clickable_tile")
        self.special_sprites = self.current_level.get_sprites_by_tag("special_tile")


        self.color_cycle = self.current_level.get_data("color_cycle")
        if self.color_cycle is None:
            self.color_cycle = [9, 8]

        self.flip_pattern = self.current_level.get_data("flip_pattern")
        if self.flip_pattern is None:
            self.flip_pattern = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


        for tile in self.tile_sprites:
            tile.color_remap(tile.pixels[0][0], self.color_cycle[0])


        for spr in self.special_sprites:
            for row in range(3):
                for col in range(3):
                    if spr.pixels[row][col] != MAGENTA:
                        spr.pixels[row][col] = self.color_cycle[0]






        _LAYOUT_ROLES = {
            "color_tile":         "",
            "variation_pattern":  "",
            "sample_tile":        "",
            "change_rule":        "",
        }
        _layout_rows: list[dict] = []
        for _spr in self.current_level.get_sprites():
            if not _spr.is_visible:
                continue
            _target, _label_en, _label_l2 = SPRITE_DISPLAY.get(
                _spr.name, (_spr.name, _spr.name, _spr.name)
            )
            if _target not in _LAYOUT_ROLES:
                continue
            _dx, _dy, _dw, _dh = self._to_display_bbox(_spr)
            _layout_rows.append({
                "role":    _target,
                "role_l2": _LAYOUT_ROLES[_target],
                "sname":   _spr.name,
                "dx":      _dx, "dy": _dy, "dw": _dw, "dh": _dh,
            })


        try:
            _ns_status = self.get_nonsprite_status() or []
        except Exception:
            _ns_status = []
        for _ns in _ns_status:
            if _ns.get("name") != "click_budget":
                continue
            _layout_rows.append({
                "role":    "remaining_clicks",
                "role_l2": ''         ,
                "sname":   "click_budget",
                "dx": int(_ns.get("x", 0)),  "dy": int(_ns.get("y", 0)),
                "dw": int(_ns.get("width", 0)), "dh": int(_ns.get("height", 0)),
            })

        if _layout_rows:
            self._set_event("initial_layout", _layout_rows)



    def step(self) -> None:
        """






        """

        self.action_log    = []
        self.action_log_l2 = []




        before: dict[int, dict] = {}
        for spr in self.current_level.get_sprites():
            center_color = None
            try:
                center_color = int(spr.pixels[1][1])
            except Exception:
                center_color = None
            before[id(spr)] = {
                "name":         spr.name,
                "visible":      spr.is_visible,
                "x":            spr.x,
                "y":            spr.y,
                "width":        spr.width,
                "height":       spr.height,
                "rotation":     spr.rotation,
                "center_color": center_color,
            }
        before_budget = self.budget_bar.current




        if self.action.id.value == 0:
            self.complete_action()
            self._append_diff_logs(before, before_budget)
            if hasattr(self, "_frame_sprite_log"):
                self._frame_sprite_log.append(self.get_sprite_status())
            if hasattr(self, "_frame_nonsprite_log"):
                self._frame_nonsprite_log.append(self.get_nonsprite_status())
            self._log_state.snapshot_step()
            return







        if self.blink_counter > 0 and self.blink_target:
            self.blink_counter -= 1
            self.blink_target.set_visible(self.blink_counter % 2 == 0)
            if self.blink_counter == 0:
                self.complete_action()
            self._append_diff_logs(before, before_budget)
            if hasattr(self, "_frame_sprite_log"):
                self._frame_sprite_log.append(self.get_sprite_status())
            if hasattr(self, "_frame_nonsprite_log"):
                self._frame_nonsprite_log.append(self.get_nonsprite_status())
            self._log_state.snapshot_step()
            return

        clicked_tile = None
        is_special   = False
        grid_pos     = None

        if self.action.id.value == 6:

            click_x = self.action.data.get("x", 0)
            click_y = self.action.data.get("y", 0)

            grid_pos = self.camera.display_to_grid(click_x, click_y)
            if grid_pos:
                gx, gy = grid_pos


                clicked_tile = self.current_level.get_sprite_at(gx, gy, "normal_tile")
                if not clicked_tile:

                    clicked_tile = self.current_level.get_sprite_at(gx, gy, "clickable_tile")
                    if clicked_tile:
                        is_special = "special_tile" in (clicked_tile._tags or [])

                if clicked_tile:
                    self.clicked_tile = clicked_tile


        if clicked_tile is None and grid_pos is not None:

            border_hit = self.current_level.get_sprite_at(grid_pos[0], grid_pos[1], "border_tile")
            if self.level_index == 0 and self.blink_target and not border_hit:
                self.blink_counter = 4




                bdx, bdy, bdw, bdh = self._to_display_bbox(self.blink_target)
                _t, _le, _l2 = self._display_for(self.blink_target.name)
                self.action_log.append(
                    f"[bright]:[{_t}]:[sname({self.blink_target.name});position({bdx}_{bdy});bbox({bdw}_{bdh});state(flash)]:{_le} at ({bdx},{bdy}) flashes to indicate the click missed a valid tile.|Object at ({bdx},{bdy}) flashes."
                )
                self.action_log_l2.append(
                    f"[bright]:[{_t}]:[sname({self.blink_target.name});position({bdx}_{bdy});bbox({bdw}_{bdh});state(flash)]:({bdx},{bdy}){_l2}|({bdx},{bdy})Object"
                )
                self._append_diff_logs(before, before_budget)
                if hasattr(self, "_frame_sprite_log"):
                    self._frame_sprite_log.append(self.get_sprite_status())
                if hasattr(self, "_frame_nonsprite_log"):
                    self._frame_nonsprite_log.append(self.get_nonsprite_status())
                self._log_state.snapshot_step()
                return
            self.complete_action()
            self._append_diff_logs(before, before_budget)
            if hasattr(self, "_frame_sprite_log"):
                self._frame_sprite_log.append(self.get_sprite_status())
            if hasattr(self, "_frame_nonsprite_log"):
                self._frame_nonsprite_log.append(self.get_nonsprite_status())
            self._log_state.snapshot_step()
            return


        NEIGHBOR_OFFSETS = [
            [(-1, -1), (0, -1), (1, -1)],
            [(-1,  0), (0,  0), (1,  0)],
            [(-1,  1), (0,  1), (1,  1)],
        ]

        if is_special and clicked_tile is not None:

            effective_pattern = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            pix = clicked_tile.pixels
            for row in range(3):
                for col in range(3):
                    if pix[row][col] == MAGENTA:
                        effective_pattern[row][col] = 1
        else:
            effective_pattern = self.flip_pattern


        if clicked_tile is not None:
            for row in range(3):
                for col in range(3):
                    if effective_pattern[row][col] == 1:
                        dx, dy = NEIGHBOR_OFFSETS[row][col]
                        target_x = self.clicked_tile.x + (dx * 4)
                        target_y = self.clicked_tile.y + (dy * 4)


                        target = self.current_level.get_sprite_at(target_x, target_y, "normal_tile")
                        if not target:
                            target = self.current_level.get_sprite_at(target_x, target_y, "clickable_tile")
                        if target:

                            current_color = target.pixels[1][1]
                            if current_color in self.color_cycle:
                                next_idx = (self.color_cycle.index(current_color) + 1) % len(self.color_cycle)
                            else:
                                next_idx = 0
                            target.color_remap(current_color, self.color_cycle[next_idx])


        if self._check_win():
            self.next_level()
            self.complete_action()

            self._append_diff_logs(before, before_budget)
            self._set_event("win", {
                "en": "[win]:[game]:[]:The level is cleared.|The level is cleared.",
                "l2": "[win]:[game]:[]:|",
            })
            if hasattr(self, "_frame_sprite_log"):
                self._frame_sprite_log.append(self.get_sprite_status())
            if hasattr(self, "_frame_nonsprite_log"):
                self._frame_nonsprite_log.append(self.get_nonsprite_status())
            self._log_state.snapshot_step()
            return


        if not self.budget_bar.consume():

            self._append_diff_logs(before, before_budget)
            self._set_event("gameover", {
                "en": "[gameover]:[game]:[]:Click budget exhausted, game over.|Game over.",
                "l2": "[gameover]:[game]:[]:0|",
            })
            if hasattr(self, "_frame_sprite_log"):
                self._frame_sprite_log.append(self.get_sprite_status())
            if hasattr(self, "_frame_nonsprite_log"):
                self._frame_nonsprite_log.append(self.get_nonsprite_status())
            self._log_state.snapshot_step()
            self.lose()

        self.complete_action()


        self._append_diff_logs(before, before_budget)


        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()

    def _display_for(self, name: str) -> tuple[str, str, str]:
        """





        """
        return SPRITE_DISPLAY.get(name, (name, name, name))

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

    def _step_bar_pixel_len(self, steps: int) -> int:
        """
        """
        max_s = self.budget_bar.max_budget
        if max_s <= 0:
            return 0
        return round(64 * steps / max_s)

    def _append_diff_logs(self, before: dict[int, dict], before_budget: int) -> None:
        """


        """
        after_sprites = {id(spr): spr for spr in self.current_level.get_sprites()}


        for spr_id, spr in after_sprites.items():
            name = spr.name
            b = before.get(spr_id)
            if b is None:
                if spr.is_visible:
                    dx, dy, dw, dh = self._to_display_bbox(spr)
                    _t, _le, _l2 = self._display_for(name)
                    self.action_log.append(f"[appear]:[{_t}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh})]:{_le} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).")
                    self.action_log_l2.append(f"[appear]:[{_t}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh})]:{_l2}({dx},{dy})|Object({dx},{dy})")
            elif spr.is_visible and not b["visible"]:
                dx, dy, dw, dh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(name)
                self.action_log.append(f"[appear]:[{_t}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh})]:{_le} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).")
                self.action_log_l2.append(f"[appear]:[{_t}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh})]:{_l2}({dx},{dy})|Object({dx},{dy})")



        _arc_scale = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in before.items():
            spr = after_sprites.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = int(b.get("width", 0))  * _arc_scale
                bdh = int(b.get("height", 0)) * _arc_scale
                _t, _le, _l2 = self._display_for(b['name'])
                self.action_log.append(f"[disappear]:[{_t}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{_le} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).")
                self.action_log_l2.append(f"[disappear]:[{_t}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{_l2}({bdx},{bdy})|Object({bdx},{bdy})")


        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            b = before.get(spr_id)
            if b is None or not b["visible"]:
                continue
            name = spr.name
            if spr.x != b["x"] or spr.y != b["y"]:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                adx, ady, adw, adh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(name)
                self.action_log.append(f"[move]:[{_t}]:[sname({name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{_le} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady}).")
                self.action_log_l2.append(f"[move]:[{_t}]:[sname({name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{_l2}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})")
            if spr.rotation != b["rotation"]:
                _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(name)
                self.action_log.append(f"[rotate]:[{_t}]:[sname({name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:{_le} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.")
                self.action_log_l2.append(f"[rotate]:[{_t}]:[sname({name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:({_rdx},{_rdy}){_l2}{b['rotation']}°{spr.rotation}°|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°")

            try:
                after_center = int(spr.pixels[1][1])
            except Exception:
                after_center = None
            before_center = b.get("center_color")
            if (
                before_center is not None and after_center is not None
                and before_center != after_center
            ):
                dx, dy, dw, dh = self._to_display_bbox(spr)


                _b_label    = COLOR_PALETTE.get(before_center, {}).get("label",    str(before_center)).lower()
                _a_label    = COLOR_PALETTE.get(after_center,  {}).get("label",    str(after_center)).lower()
                _b_label_l2 = COLOR_PALETTE.get(before_center, {}).get("label_l2", str(before_center))
                _a_label_l2 = COLOR_PALETTE.get(after_center,  {}).get("label_l2", str(after_center))


                _t, _le, _l2 = self._display_for(name)
                self.action_log.append(
                    f"[change]:[{_t}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh});color({_b_label})_({_a_label})]:{_le} at ({dx},{dy}) changes color from {_b_label} to {_a_label}.|Object at ({dx},{dy}) changes color from {_b_label} to {_a_label}."
                )
                self.action_log_l2.append(
                    f"[change]:[{_t}]:[sname({name});position({dx}_{dy});bbox({dw}_{dh});color({_b_label})_({_a_label})]:({dx},{dy}){_l2}{_b_label_l2}{_a_label_l2}|({dx},{dy})Object{_b_label_l2}{_a_label_l2}"
                )


        after_budget = self.budget_bar.current
        if after_budget != before_budget and self._step_bar_pixel_len(before_budget) != self._step_bar_pixel_len(after_budget):
            if after_budget < before_budget:
                self.action_log.append(
                    f"[change]:[click_budget]:[sname(click_budget);count({before_budget})_({after_budget})]:The orange click budget bar at the bottom shrinks from the right ({before_budget}→{after_budget} clicks remain).|The orange Object at the bottom shrinks from the right ({before_budget}→{after_budget})."
                )
                self.action_log_l2.append(
                    f"[change]:[click_budget]:[sname(click_budget);count({before_budget})_({after_budget})]:{before_budget}→{after_budget}|Object{before_budget}→{after_budget}"
                )
            else:
                self.action_log.append(
                    f"[change]:[click_budget]:[sname(click_budget);count({before_budget})_({after_budget})]:The orange click budget bar at the bottom extends to the right ({before_budget}→{after_budget} clicks).|The orange Object at the bottom extends to the right ({before_budget}→{after_budget})."
                )
                self.action_log_l2.append(
                    f"[change]:[click_budget]:[sname(click_budget);count({before_budget})_({after_budget})]:{before_budget}→{after_budget}|Object{before_budget}→{after_budget}"
                )






    def _check_win(self) -> bool:
        """







        """
        NEIGHBOR_OFFSETS = [
            [(-1, -1), (0, -1), (1, -1)],
            [(-1,  0), (0,  0), (1,  0)],
            [(-1,  1), (0,  1), (1,  1)],
        ]

        for btile in self.border_sprites:
            center_color = btile.pixels[1][1]

            for row in range(3):
                for col in range(3):
                    if row == 1 and col == 1:
                        continue

                    expect_equal = (btile.pixels[row][col] == WHITE)
                    dx, dy = NEIGHBOR_OFFSETS[row][col]
                    nx = btile.x + dx * 4
                    ny = btile.y + dy * 4


                    neighbor = self.current_level.get_sprite_at(nx, ny, "normal_tile")
                    if not neighbor:
                        neighbor = self.current_level.get_sprite_at(nx, ny, "clickable_tile")

                    if neighbor:
                        neighbor_color = neighbor.pixels[1][1]
                        color_match = (neighbor_color == center_color)
                        if expect_equal and not color_match:
                            return False
                        if not expect_equal and color_match:
                            return False

        return True



    def _get_hidden_state(self) -> np.ndarray:
        """"""
        state = np.zeros(1, dtype=np.int16)
        state[0] = self.budget_bar.current
        return state



    def _get_valid_actions(self) -> list[ActionInput]:
        """"""
        clickables = self.current_level.get_sprites_by_tag("clickable_tile")
        scale, offset_x, offset_y = self.camera._calculate_scale_and_offset()
        actions: list[ActionInput] = []

        for tile in clickables:
            if not self._is_sprite_clickable_now(tile):
                continue

            disp_x = tile._x * scale + offset_x
            disp_y = tile._y * scale + offset_y
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": disp_x, "y": disp_y}))

        return actions



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
            tags = list(spr._tags) if spr._tags else []




            _t, label, label_l2 = SPRITE_DISPLAY.get(spr.name, (spr.name, spr.name, spr.name))
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
            "name":     "click_budget",
            "label":    "Clicks Remaining",
            "label_l2": ''                     ,
            "value":    self.budget_bar.current,
            "max":      self.budget_bar.max_budget,
            "x":        0, "y":      63,
            "width":    64, "height": 1,
        })

        return result

    def get_game_knowledge(self) -> dict:
        """

        """
        return GAME_KNOWLEDGE
