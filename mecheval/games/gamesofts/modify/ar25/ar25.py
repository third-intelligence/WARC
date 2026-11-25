"""

















"""
import math
import re
from collections import deque
from typing import Optional, TypedDict

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
    "piece_blue_s": Sprite(
        pixels=[
            [9, 9, 9],
            [9, -1, 9],
            [-1, -1, 9],
            [-1, -1, 9],
        ],
        name="piece_blue_s",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_cross": Sprite(
        pixels=[
            [-1, 5, 5, 5, -1],
            [5, -1, -1, -1, 5],
            [5, -1, -1, -1, -1],
            [5, -1, -1, -1, -1],
            [-1, 5, -1, -1, -1],
        ],
        name="piece_black_cross",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_green_l": Sprite(
        pixels=[
            [14, -1],
            [14, -1],
            [14, 14],
            [-1, 14],
            [-1, 14],
        ],
        name="piece_green_l",
        visible=True,
        collidable=True,
        tags=["piece", "auto_rotate_h"],
    ),
    "piece_orange_sq": Sprite(
        pixels=[
            [12, 12, 12],
            [12, 12, 12],
            [12, 12, 12],
        ],
        name="piece_orange_sq",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_red_t": Sprite(
        pixels=[
            [-1, 8],
            [8, 8],
            [-1, 8],
        ],
        name="piece_red_t",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "deco_lattice": Sprite(
        pixels=[
            [-1, -1, -1, 12, -1, -1, -1, 12, -1, -1, -1],
            [-1, -1, 12, -1, 12, -1, 12, -1, 12, -1, -1],
            [-1, 12, 12, -1, 12, -1, 12, -1, 12, 12, -1],
            [12, -1, -1, -1, 12, -1, 12, -1, -1, -1, 12],
            [-1, 12, 12, 12, -1, -1, -1, 12, 12, 12, -1],
            [-1, -1, -1, 12, 12, -1, 12, 12, -1, -1, -1],
            [-1, 12, 12, 12, -1, -1, -1, 12, 12, 12, -1],
            [-1, -1, -1, 12, -1, -1, -1, 12, -1, -1, -1],
            [-1, 12, 12, 12, -1, -1, -1, 12, 12, 12, -1],
            [-1, -1, -1, 12, 12, -1, 12, 12, -1, -1, -1],
            [-1, 12, 12, 12, -1, -1, -1, 12, 12, 12, -1],
            [12, -1, -1, -1, 12, -1, 12, -1, -1, -1, 12],
            [-1, 12, 12, -1, 12, -1, 12, -1, 12, 12, -1],
            [-1, -1, 12, -1, 12, -1, 12, -1, 12, -1, -1],
            [-1, -1, -1, 12, -1, -1, -1, 12, -1, -1, -1],
        ],
        name="deco_lattice",
        visible=True,
        collidable=True,
        layer=-11,
    ),
    "piece_blue_rect": Sprite(
        pixels=[
            [9, 9],
            [9, 9],
            [9, 9],
        ],
        name="piece_blue_rect",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_yellow_s": Sprite(
        pixels=[
            [11, -1, -1, -1],
            [11, 11, 11, 11],
            [-1, -1, -1, 11],
        ],
        name="piece_yellow_s",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "mirror_v_fixed": Sprite(
        pixels=[
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
        ],
        name="mirror_v_fixed",
        visible=True,
        collidable=True,
        tags=["mirror_axis", "mirror_v_fixed", "mirror_v_axis"],
        layer=-5,
    ),
    "mirror_h_axis": Sprite(
        pixels=[
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ],
        name="mirror_h_axis",
        visible=True,
        collidable=True,
        tags=["mirror_axis", "sys_click", "mirror_h_axis"],
        layer=-5,
    ),
    "overlay_blank": Sprite(
        pixels=[
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        ],
        name="overlay_blank",
        visible=True,
        collidable=True,
        tags=["overlay_blank"],
    ),
    "piece_blue_l": Sprite(
        pixels=[
            [9, -1, -1],
            [9, -1, -1],
            [9, 9, 9],
        ],
        name="piece_blue_l",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_red_j": Sprite(
        pixels=[
            [8, -1],
            [8, 8],
            [8, -1],
        ],
        name="piece_red_j",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_green_t": Sprite(
        pixels=[
            [-1, 14, -1],
            [-1, 14, -1],
            [14, 14, 14],
        ],
        name="piece_green_t",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_green_v": Sprite(
        pixels=[
            [-1, -1, 14, -1, -1],
            [-1, 14, -1, 14, -1],
            [14, -1, -1, -1, 14],
        ],
        name="piece_green_v",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_fork": Sprite(
        pixels=[
            [-1, 5, -1, 5, -1],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [-1, 5, -1, 5, -1],
        ],
        name="piece_black_fork",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_hbar": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5],
        ],
        name="piece_black_hbar",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_u": Sprite(
        pixels=[
            [5, -1, -1, -1, 5],
            [5, 5, 5, 5, 5],
        ],
        name="piece_black_u",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_t4": Sprite(
        pixels=[
            [5, 5, 5, 5],
            [-1, 5, 5, -1],
        ],
        name="piece_black_t4",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_l": Sprite(
        pixels=[
            [5, -1, -1, -1],
            [5, -1, -1, -1],
            [5, -1, -1, -1],
            [5, 5, 5, 5],
        ],
        name="piece_black_l",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_s4": Sprite(
        pixels=[
            [-1, 5, -1, -1],
            [5, 5, 5, 5],
        ],
        name="piece_black_s4",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_vbar": Sprite(
        pixels=[
            [5],
            [5],
            [5],
            [5],
            [5],
            [5],
        ],
        name="piece_black_vbar",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_yellow_tri": Sprite(
        pixels=[
            [-1, -1, 11, -1, -1],
            [-1, 11, 11, 11, -1],
            [11, 11, 11, 11, 11],
        ],
        name="piece_yellow_tri",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_z": Sprite(
        pixels=[
            [5, 5, 5, -1, -1],
            [-1, -1, 5, -1, -1],
            [-1, -1, 5, -1, -1],
            [-1, -1, 5, 5, 5],
        ],
        name="piece_black_z",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_arch": Sprite(
        pixels=[
            [5, 5, 5],
            [5, -1, 5],
            [-1, 5, -1],
        ],
        name="piece_black_arch",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_yellow_corner": Sprite(
        pixels=[
            [11, 11],
            [-1, 11],
        ],
        name="piece_yellow_corner",
        visible=True,
        collidable=True,
    ),
    "deco_frame": Sprite(
        pixels=[
            [11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11],
        ],
        name="deco_frame",
        visible=True,
        collidable=True,
    ),
    "piece_orange_t": Sprite(
        pixels=[
            [-1, 12, -1],
            [12, 12, 12],
            [-1, 12, 12],
        ],
        name="piece_orange_t",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_purple_j": Sprite(
        pixels=[
            [15, -1, -1, -1],
            [15, -1, 15, 15],
            [15, 15, 15, -1],
        ],
        name="piece_purple_j",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_l3": Sprite(
        pixels=[
            [5, 5, 5],
            [-1, -1, 5],
            [-1, -1, 5],
        ],
        name="piece_black_l3",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_blue_z": Sprite(
        pixels=[
            [9, 9, 9, 9, -1],
            [-1, -1, 9, 9, -1],
            [-1, -1, -1, 9, 9],
            [-1, -1, -1, 9, -1],
        ],
        name="piece_blue_z",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_blue_l2": Sprite(
        pixels=[
            [9, -1, -1, -1, -1],
            [-1, 9, 9, 9, 9],
            [-1, 9, -1, -1, -1],
        ],
        name="piece_blue_l2",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_r": Sprite(
        pixels=[
            [5, 5, -1, -1],
            [-1, 5, -1, 5],
            [-1, 5, 5, 5],
        ],
        name="piece_black_r",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_yellow_cross": Sprite(
        pixels=[
            [-1, -1, 11],
            [-1, 11, -1],
            [-1, 11, -1],
            [11, 11, 11],
            [-1, 11, -1],
        ],
        name="piece_yellow_cross",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_z3": Sprite(
        pixels=[
            [-1, 5, 5],
            [5, 5, -1],
            [5, -1, -1],
        ],
        name="piece_black_z3",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_s5": Sprite(
        pixels=[
            [5, 5, 5, 5, -1],
            [-1, -1, -1, 5, -1],
            [-1, -1, -1, 5, 5],
            [-1, -1, -1, -1, 5],
            [-1, -1, 5, 5, 5],
        ],
        name="piece_black_s5",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_purple_z": Sprite(
        pixels=[
            [-1, -1, 15],
            [15, 15, 15],
            [15, -1, -1],
        ],
        name="piece_purple_z",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "anim_flash": Sprite(
        pixels=[
            [0, 0, 0, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, 0, -1, -1],
            [-1, -1, 0, 0, 0],
        ],
        name="anim_flash",
        visible=True,
        collidable=True,
        layer=5,
    ),
    "piece_black_tl": Sprite(
        pixels=[
            [5, 5, 5],
            [-1, -1, 5],
            [-1, -1, 5],
            [-1, -1, 5],
            [-1, -1, 5],
        ],
        name="piece_black_tl",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_vbar2": Sprite(
        pixels=[
            [5],
            [5],
            [5],
            [5],
            [5],
            [5],
        ],
        name="piece_black_vbar2",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_purple_h": Sprite(
        pixels=[
            [15, 15, 15, -1, 15],
            [-1, -1, 15, 15, 15],
        ],
        name="piece_purple_h",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_z5": Sprite(
        pixels=[
            [5, 5, -1, -1, -1],
            [5, -1, -1, -1, -1],
            [5, 5, 5, 5, 5],
        ],
        name="piece_black_z5",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "target_dot": Sprite(
        pixels=[
            [11],
        ],
        name="target_dot",
        visible=True,
        collidable=True,
        tags=["target_dot"],
        layer=-4,
    ),
    "deco_win_logo": Sprite(
        pixels=[
            [-1, -1, -1, 15, 15, 15, 15, 15, -1, -1, -1],
            [-1, 15, 15, 15, -1, 15, 15, 15, 15, 15, -1],
            [-1, 15, 15, 15, -1, 15, -1, 15, 15, 15, -1],
            [15, 15, 15, 15, 15, 15, -1, 15, 15, 15, 15],
            [-1, -1, 15, 15, -1, 15, -1, 15, 15, -1, -1],
            [-1, -1, -1, 15, -1, 15, 15, 15, -1, -1, -1],
            [-1, -1, -1, 15, 15, 15, 15, 15, -1, -1, -1],
        ],
        name="deco_win_logo",
        visible=False,
        collidable=True,
    ),
    "piece_orange_j": Sprite(
        pixels=[
            [-1, -1, 12, -1],
            [12, 12, 12, -1],
            [12, 12, 12, 12],
            [12, -1, -1, -1],
        ],
        name="piece_orange_j",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_red_auto": Sprite(
        pixels=[
            [8, 8, -1],
            [-1, 8, -1],
            [-1, 8, 8],
        ],
        name="piece_red_auto",
        visible=True,
        collidable=True,
        tags=["piece", "auto_rotate_v"],
        layer=-2,
    ),
    "piece_orange_hbar": Sprite(
        pixels=[
            [12, 12, 12, 12],
        ],
        name="piece_orange_hbar",
        visible=True,
        collidable=True,
        tags=["piece"],
    ),
    "piece_black_sq": Sprite(
        pixels=[
            [5, 5],
            [-1, 5],
        ],
        name="piece_black_sq",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_black_scurve": Sprite(
        pixels=[
            [-1, -1, 5],
            [-1, -1, 5],
            [-1, 5, 5],
            [-1, 5, -1],
            [5, 5, -1],
            [5, -1, -1],
            [5, -1, -1],
        ],
        name="piece_black_scurve",
        visible=True,
        collidable=True,
        tags=["piece", "sys_click"],
    ),
    "piece_orange_h_only": Sprite(
        pixels=[
            [-1, 12, 12, -1],
            [-1, -1, 12, -1],
            [12, 12, 12, 12],
        ],
        name="piece_orange_h_only",
        visible=True,
        collidable=True,
        tags=["piece", "reflect_h_only"],
        layer=-1,
    ),
    "mirror_v_axis": Sprite(
        pixels=[
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
            [10],
        ],
        name="mirror_v_axis",
        visible=True,
        collidable=True,
        tags=["mirror_axis", "sys_click", "mirror_v_axis"],
        layer=-5,
    ),
}
levels = [

    Level(
        sprites=[
            sprites["mirror_v_fixed"].clone().set_position(10, 0),
            sprites["piece_black_l3"].clone().set_position(6, 5),
            sprites["target_dot"].clone().set_position(19, 15),
            sprites["target_dot"].clone().set_position(17, 17),
            sprites["target_dot"].clone().set_position(17, 16),
            sprites["target_dot"].clone().set_position(17, 15),
            sprites["target_dot"].clone().set_position(18, 15),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 64,
            "mirror_axis": ["mirror_v_axis", None],
        },
    ),

    Level(
        sprites=[
            sprites["piece_black_z"].clone().set_position(15, 6),
            sprites["target_dot"].clone().set_position(5, 14),
            sprites["target_dot"].clone().set_position(1, 17),
            sprites["target_dot"].clone().set_position(4, 14),
            sprites["target_dot"].clone().set_position(3, 14),
            sprites["target_dot"].clone().set_position(3, 15),
            sprites["target_dot"].clone().set_position(3, 16),
            sprites["target_dot"].clone().set_position(3, 17),
            sprites["target_dot"].clone().set_position(2, 17),
            sprites["mirror_v_axis"].clone().set_position(12, -3),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 64,
            "mirror_axis": ["mirror_v_axis", None],
        },
    ),

    Level(
        sprites=[
            sprites["mirror_h_axis"].clone().set_position(-5, 16),
            sprites["piece_black_t4"].clone().set_position(15, 9),
            sprites["piece_black_l"].clone().set_position(4, 7),
            sprites["target_dot"].clone().set_position(14, 17),
            sprites["target_dot"].clone().set_position(14, 1),
            sprites["target_dot"].clone().set_position(11, 4),
            sprites["target_dot"].clone().set_position(11, 14),
            sprites["target_dot"].clone().set_position(11, 17),
            sprites["target_dot"].clone().set_position(11, 1),
            sprites["target_dot"].clone().set_position(3, 14),
            sprites["target_dot"].clone().set_position(4, 14),
            sprites["target_dot"].clone().set_position(5, 14),
            sprites["target_dot"].clone().set_position(6, 14),
            sprites["target_dot"].clone().set_position(4, 15),
            sprites["target_dot"].clone().set_position(5, 15),
            sprites["target_dot"].clone().set_position(3, 4),
            sprites["target_dot"].clone().set_position(4, 4),
            sprites["target_dot"].clone().set_position(5, 4),
            sprites["target_dot"].clone().set_position(6, 4),
            sprites["target_dot"].clone().set_position(5, 3),
            sprites["target_dot"].clone().set_position(4, 3),
            sprites["target_dot"].clone().set_position(11, 15),
            sprites["target_dot"].clone().set_position(11, 16),
            sprites["target_dot"].clone().set_position(12, 17),
            sprites["target_dot"].clone().set_position(13, 17),
            sprites["target_dot"].clone().set_position(11, 3),
            sprites["target_dot"].clone().set_position(11, 2),
            sprites["target_dot"].clone().set_position(12, 1),
            sprites["target_dot"].clone().set_position(13, 1),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 128,
            "mirror_axis": ["mirror_v_axis", None],
        },
    ),

    Level(
        sprites=[
            sprites["mirror_h_axis"].clone().set_position(-6, 3),
            sprites["piece_black_u"].clone().set_position(4, 6),
            sprites["piece_black_vbar2"].clone().set_position(6, 10),
            sprites["target_dot"].clone().set_position(15, 11),
            sprites["target_dot"].clone().set_position(15, 7),
            sprites["target_dot"].clone().set_position(11, 7),
            sprites["target_dot"].clone().set_position(11, 11),
            sprites["target_dot"].clone().set_position(13, 3),
            sprites["target_dot"].clone().set_position(13, 15),
            sprites["target_dot"].clone().set_position(13, 8),
            sprites["target_dot"].clone().set_position(13, 10),
            sprites["target_dot"].clone().set_position(12, 7),
            sprites["target_dot"].clone().set_position(13, 7),
            sprites["target_dot"].clone().set_position(14, 7),
            sprites["target_dot"].clone().set_position(11, 6),
            sprites["target_dot"].clone().set_position(15, 6),
            sprites["target_dot"].clone().set_position(13, 6),
            sprites["target_dot"].clone().set_position(13, 5),
            sprites["target_dot"].clone().set_position(13, 4),
            sprites["target_dot"].clone().set_position(12, 11),
            sprites["target_dot"].clone().set_position(13, 11),
            sprites["target_dot"].clone().set_position(14, 11),
            sprites["target_dot"].clone().set_position(11, 12),
            sprites["target_dot"].clone().set_position(15, 12),
            sprites["target_dot"].clone().set_position(13, 12),
            sprites["target_dot"].clone().set_position(13, 13),
            sprites["target_dot"].clone().set_position(13, 14),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 128,
            "mirror_axis": ["mirror_v_axis", None],
        },
    ),

    Level(
        sprites=[
            sprites["mirror_h_axis"].clone().set_position(0, 5),
            sprites["piece_black_s5"].clone().set_position(14, 12),
            sprites["target_dot"].clone().set_position(8, 9),
            sprites["target_dot"].clone().set_position(9, 9),
            sprites["target_dot"].clone().set_position(8, 8),
            sprites["target_dot"].clone().set_position(7, 9),
            sprites["target_dot"].clone().set_position(4, 13),
            sprites["target_dot"].clone().set_position(8, 10),
            sprites["target_dot"].clone().set_position(12, 13),
            sprites["target_dot"].clone().set_position(12, 5),
            sprites["target_dot"].clone().set_position(4, 5),
            sprites["target_dot"].clone().set_position(6, 9),
            sprites["target_dot"].clone().set_position(10, 9),
            sprites["target_dot"].clone().set_position(5, 5),
            sprites["target_dot"].clone().set_position(6, 5),
            sprites["target_dot"].clone().set_position(11, 13),
            sprites["target_dot"].clone().set_position(10, 13),
            sprites["target_dot"].clone().set_position(7, 5),
            sprites["target_dot"].clone().set_position(9, 13),
            sprites["target_dot"].clone().set_position(8, 11),
            sprites["target_dot"].clone().set_position(9, 11),
            sprites["target_dot"].clone().set_position(9, 12),
            sprites["target_dot"].clone().set_position(8, 7),
            sprites["target_dot"].clone().set_position(7, 7),
            sprites["target_dot"].clone().set_position(7, 6),
            sprites["mirror_v_axis"].clone().set_position(3, 0),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 128,
            "mirror_axis": 0,
        },
    ),

    Level(
        sprites=[
            sprites["piece_black_cross"].clone().set_position(14, 3),
            sprites["mirror_h_axis"].clone(),
            sprites["piece_black_s4"].clone().set_position(17, 8),
            sprites["target_dot"].clone().set_position(4, 3),
            sprites["target_dot"].clone().set_position(1, 6),
            sprites["target_dot"].clone().set_position(2, 7),
            sprites["target_dot"].clone().set_position(3, 7),
            sprites["target_dot"].clone().set_position(4, 7),
            sprites["target_dot"].clone().set_position(2, 9),
            sprites["target_dot"].clone().set_position(3, 9),
            sprites["target_dot"].clone().set_position(3, 12),
            sprites["target_dot"].clone().set_position(2, 13),
            sprites["target_dot"].clone().set_position(4, 15),
            sprites["target_dot"].clone().set_position(3, 15),
            sprites["target_dot"].clone().set_position(2, 15),
            sprites["target_dot"].clone().set_position(5, 9),
            sprites["target_dot"].clone().set_position(4, 9),
            sprites["target_dot"].clone().set_position(4, 13),
            sprites["target_dot"].clone().set_position(5, 13),
            sprites["target_dot"].clone().set_position(1, 16),
            sprites["target_dot"].clone().set_position(4, 19),
            sprites["target_dot"].clone().set_position(5, 18),
            sprites["target_dot"].clone().set_position(5, 17),
            sprites["target_dot"].clone().set_position(5, 16),
            sprites["target_dot"].clone().set_position(7, 16),
            sprites["target_dot"].clone().set_position(7, 17),
            sprites["target_dot"].clone().set_position(7, 18),
            sprites["target_dot"].clone().set_position(8, 19),
            sprites["target_dot"].clone().set_position(11, 16),
            sprites["target_dot"].clone().set_position(10, 15),
            sprites["target_dot"].clone().set_position(9, 15),
            sprites["target_dot"].clone().set_position(8, 15),
            sprites["target_dot"].clone().set_position(10, 13),
            sprites["target_dot"].clone().set_position(9, 12),
            sprites["target_dot"].clone().set_position(9, 9),
            sprites["target_dot"].clone().set_position(10, 9),
            sprites["target_dot"].clone().set_position(8, 7),
            sprites["target_dot"].clone().set_position(9, 7),
            sprites["target_dot"].clone().set_position(10, 7),
            sprites["target_dot"].clone().set_position(7, 13),
            sprites["target_dot"].clone().set_position(8, 13),
            sprites["target_dot"].clone().set_position(8, 9),
            sprites["target_dot"].clone().set_position(7, 9),
            sprites["target_dot"].clone().set_position(11, 6),
            sprites["target_dot"].clone().set_position(8, 3),
            sprites["target_dot"].clone().set_position(7, 4),
            sprites["target_dot"].clone().set_position(7, 5),
            sprites["target_dot"].clone().set_position(7, 6),
            sprites["target_dot"].clone().set_position(5, 6),
            sprites["target_dot"].clone().set_position(5, 5),
            sprites["target_dot"].clone().set_position(5, 4),
            sprites["target_dot"].clone().set_position(9, 13),
            sprites["target_dot"].clone().set_position(3, 13),
            sprites["target_dot"].clone().set_position(9, 10),
            sprites["target_dot"].clone().set_position(3, 10),
            sprites["mirror_v_axis"].clone().set_position(7, -1),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 320,
            "mirror_axis": 0,
        },
    ),

    Level(
        sprites=[
            sprites["mirror_h_axis"].clone().set_position(0, 5),
            sprites["piece_black_fork"].clone().set_position(5, 16),
            sprites["target_dot"].clone().set_position(9, 7),
            sprites["target_dot"].clone().set_position(7, 3),
            sprites["target_dot"].clone().set_position(8, 5),
            sprites["target_dot"].clone().set_position(7, 1),
            sprites["target_dot"].clone().set_position(9, 1),
            sprites["target_dot"].clone().set_position(11, 1),
            sprites["target_dot"].clone().set_position(13, 1),
            sprites["target_dot"].clone().set_position(15, 1),
            sprites["target_dot"].clone().set_position(17, 1),
            sprites["target_dot"].clone().set_position(17, 3),
            sprites["target_dot"].clone().set_position(16, 5),
            sprites["target_dot"].clone().set_position(15, 7),
            sprites["target_dot"].clone().set_position(16, 9),
            sprites["target_dot"].clone().set_position(8, 9),
            sprites["target_dot"].clone().set_position(7, 11),
            sprites["target_dot"].clone().set_position(17, 11),
            sprites["target_dot"].clone().set_position(17, 13),
            sprites["target_dot"].clone().set_position(7, 13),
            sprites["target_dot"].clone().set_position(9, 13),
            sprites["target_dot"].clone().set_position(11, 13),
            sprites["target_dot"].clone().set_position(13, 13),
            sprites["target_dot"].clone().set_position(15, 13),
            sprites["target_dot"].clone().set_position(11, 10),
            sprites["target_dot"].clone().set_position(13, 10),
            sprites["target_dot"].clone().set_position(11, 4),
            sprites["target_dot"].clone().set_position(13, 4),
            sprites["target_dot"].clone().set_position(9, 10),
            sprites["target_dot"].clone().set_position(15, 10),
            sprites["target_dot"].clone().set_position(9, 4),
            sprites["target_dot"].clone().set_position(15, 4),
            sprites["target_dot"].clone().set_position(11, 12),
            sprites["target_dot"].clone().set_position(11, 11),
            sprites["target_dot"].clone().set_position(9, 12),
            sprites["target_dot"].clone().set_position(9, 11),
            sprites["target_dot"].clone().set_position(13, 2),
            sprites["target_dot"].clone().set_position(13, 3),
            sprites["target_dot"].clone().set_position(15, 2),
            sprites["target_dot"].clone().set_position(15, 3),
            sprites["target_dot"].clone().set_position(10, 12),
            sprites["target_dot"].clone().set_position(10, 11),
            sprites["target_dot"].clone().set_position(14, 3),
            sprites["target_dot"].clone().set_position(14, 2),
            sprites["piece_black_scurve"].clone().set_position(17, 13),
            sprites["mirror_v_axis"].clone().set_position(3, 0),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 320,
            "mirror_axis": 0,
        },
    ),

    Level(
        sprites=[
            sprites["mirror_h_axis"].clone().set_position(0, 5),
            sprites["piece_black_tl"].clone().set_position(7, 7),
            sprites["piece_black_z5"].clone().set_position(13, 13),
            sprites["target_dot"].clone().set_position(16, 3),
            sprites["target_dot"].clone().set_position(17, 3),
            sprites["target_dot"].clone().set_position(18, 3),
            sprites["target_dot"].clone().set_position(18, 4),
            sprites["target_dot"].clone().set_position(18, 5),
            sprites["target_dot"].clone().set_position(18, 6),
            sprites["target_dot"].clone().set_position(18, 7),
            sprites["target_dot"].clone().set_position(18, 8),
            sprites["target_dot"].clone().set_position(17, 8),
            sprites["target_dot"].clone().set_position(16, 8),
            sprites["target_dot"].clone().set_position(19, 8),
            sprites["target_dot"].clone().set_position(20, 8),
            sprites["target_dot"].clone().set_position(20, 7),
            sprites["target_dot"].clone().set_position(20, 6),
            sprites["target_dot"].clone().set_position(19, 6),
            sprites["target_dot"].clone().set_position(8, 3),
            sprites["target_dot"].clone().set_position(7, 3),
            sprites["target_dot"].clone().set_position(6, 3),
            sprites["target_dot"].clone().set_position(6, 4),
            sprites["target_dot"].clone().set_position(6, 5),
            sprites["target_dot"].clone().set_position(6, 6),
            sprites["target_dot"].clone().set_position(6, 7),
            sprites["target_dot"].clone().set_position(8, 8),
            sprites["target_dot"].clone().set_position(7, 8),
            sprites["target_dot"].clone().set_position(6, 8),
            sprites["target_dot"].clone().set_position(5, 8),
            sprites["target_dot"].clone().set_position(5, 6),
            sprites["target_dot"].clone().set_position(4, 6),
            sprites["target_dot"].clone().set_position(4, 7),
            sprites["target_dot"].clone().set_position(4, 8),
            sprites["target_dot"].clone().set_position(8, 14),
            sprites["target_dot"].clone().set_position(7, 14),
            sprites["target_dot"].clone().set_position(6, 14),
            sprites["target_dot"].clone().set_position(6, 15),
            sprites["target_dot"].clone().set_position(6, 16),
            sprites["target_dot"].clone().set_position(6, 17),
            sprites["target_dot"].clone().set_position(6, 18),
            sprites["target_dot"].clone().set_position(6, 19),
            sprites["target_dot"].clone().set_position(7, 19),
            sprites["target_dot"].clone().set_position(8, 19),
            sprites["target_dot"].clone().set_position(5, 14),
            sprites["target_dot"].clone().set_position(4, 14),
            sprites["target_dot"].clone().set_position(4, 15),
            sprites["target_dot"].clone().set_position(4, 16),
            sprites["target_dot"].clone().set_position(5, 16),
            sprites["target_dot"].clone().set_position(16, 14),
            sprites["target_dot"].clone().set_position(17, 14),
            sprites["target_dot"].clone().set_position(18, 14),
            sprites["target_dot"].clone().set_position(19, 14),
            sprites["target_dot"].clone().set_position(20, 14),
            sprites["target_dot"].clone().set_position(20, 15),
            sprites["target_dot"].clone().set_position(18, 15),
            sprites["target_dot"].clone().set_position(18, 16),
            sprites["target_dot"].clone().set_position(19, 16),
            sprites["target_dot"].clone().set_position(20, 16),
            sprites["target_dot"].clone().set_position(18, 17),
            sprites["target_dot"].clone().set_position(18, 18),
            sprites["target_dot"].clone().set_position(18, 19),
            sprites["target_dot"].clone().set_position(17, 19),
            sprites["target_dot"].clone().set_position(16, 19),
            sprites["mirror_v_axis"].clone().set_position(3, 0),
        ],
        grid_size=(21, 21),
        data={
            "StepCounter": 320,
            "mirror_axis": 0,
        },
    ),
]


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

BACKGROUND_COLOR = 9
PADDING_COLOR = 5
kevchhhmha = -1
qxfymojgwa = 0
khaeijgkpw = 1
ymbzgqpfgw = 2
cokaikipqv = 3
qqcnwokqtb = 4
bwtkxvpmxp = 5
dsjxzfeaay = 6
babxulumwz = 7
iuqegctory = 8
utwrvgdxfn = 9
segaklzkcp = 10
dqailkzmtf = 11
rrdyedysxy = 12
dasrwajgpe = 13
usrptgpomh = 14
eqoiopyobo = 15
akedgxijnx = cokaikipqv
xwfrtwinpo = qxfymojgwa
liirwepfdj = BACKGROUND_COLOR
tuovamceti = bwtkxvpmxp
kddxpzgafq = qqcnwokqtb
euzrpaakrj = segaklzkcp




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
    "piece":            "Piece",
    "mirror_axis":      "Mirror Axis",
    "mirror_v_fixed":   "Fixed V-Mirror",
    "mirror_v_axis":    "V-Mirror",
    "mirror_h_axis":    "H-Mirror",
    "target_dot":       "Target Dot",
    "auto_rotate_h":    "Auto-Rotate H",
    "auto_rotate_v":    "Auto-Rotate V",
    "reflect_h_only":   "H-Reflect Only",
    "sys_click":        "Clickable",
    "overlay_blank":    "Overlay",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "piece":            "",
    "mirror_axis":      "",
    "mirror_v_fixed":   "",
    "mirror_v_axis":    "",
    "mirror_h_axis":    "",
    "target_dot":       "",
    "auto_rotate_h":    "",
    "auto_rotate_v":    "",
    "reflect_h_only":   "",
    "sys_click":        "",
    "overlay_blank":    "",
}


def _build_sprite_display_map() -> dict:
    """







    """
    out: dict = {}
    for spr_name, spr in sprites.items():
        target = label_en = label_l2 = spr_name
        for t in (spr.tags or []):
            if t == "sys_click":
                continue
            if t in TAG_DISPLAY_NAMES:
                label_en = TAG_DISPLAY_NAMES[t]
                label_l2 = TAG_DISPLAY_NAMES_L2.get(t, label_en)
                target   = label_en.lower().replace(" ", "_").replace("-", "_")
                break
        out[spr_name] = (target, label_en, label_l2)
    return out



SPRITE_DISPLAY: dict = _build_sprite_display_map()





_APPEAR_SUPPRESS = {
    "anim_flash",
}




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
    step_count: int | None = None
    try:
        step_count = level.get_data("StepCounter")
    except Exception:
        pass

    grid = getattr(level, "grid_size", (21, 21)) or (21, 21)

    target_dot_positions: list[dict] = []
    level_sprites_info: list[dict] = []

    for spr in getattr(level, "_sprites", []) or []:
        tags = list(spr._tags) if spr._tags else []
        if "target_dot" in tags:
            target_dot_positions.append({"x": spr.x, "y": spr.y})
        elif "piece" in tags or "mirror_axis" in tags:
            level_sprites_info.append({
                "name":       spr.name,
                "x":          spr.x,
                "y":          spr.y,
                "width":      int(spr.width),
                "height":     int(spr.height),
                "tags":       tags,
            })

    _analyses: list[dict] = [
        {
            "gimmicks":     ["Only 1 movable piece; vertical mirror axis is fixed"],
            "gimmicks_l2":  [''                                             ],
            "subgoals":     ["Move piece_black_l3 to align its reflections with all 5 target dots"],
            "subgoals_l2":  [''                                                                         ],
            "goal_flow":    "Move the L-shaped black piece so its reflected image covers all 5 target dots near the top-right area.",
            "goal_flow_l2": ''                                                                                            ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["1 movable piece + 1 selectable vertical mirror axis"],
            "gimmicks_l2":  [''                                                     ],
            "subgoals":     ["Select the mirror axis, position it; then move piece to cover targets via reflection"],
            "subgoals_l2":  [''                                                                                                   ],
            "goal_flow":    "Adjust the vertical mirror axis and the Z-shaped piece so the reflected image covers 8 target dots.",
            "goal_flow_l2": ''                                                                                ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Horizontal mirror axis; 2 pieces; many target dots arranged symmetrically"],
            "gimmicks_l2":  [''                                                                          ],
            "subgoals":     ["Place both pieces so their reflections across the horizontal axis cover all targets"],
            "subgoals_l2":  [''                                                          ],
            "goal_flow":    "Use the horizontal mirror to reflect two pieces onto symmetric target dot clusters.",
            "goal_flow_l2": ''                                                                               ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Horizontal mirror axis; 2 pieces; plus-shaped and vertical-bar pieces"],
            "gimmicks_l2":  [''                                                   ],
            "subgoals":     ["Position horizontal mirror, then align both pieces to cover all targets via reflections"],
            "subgoals_l2":  [''                                                                            ],
            "goal_flow":    "Move the horizontal mirror and two pieces to cover all targets through their reflections.",
            "goal_flow_l2": ''                                                                                  ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Horizontal mirror axis; 1 piece; 1 vertical mirror axis; targets form two clusters"],
            "gimmicks_l2":  [''                                                                ],
            "subgoals":     ["Adjust both mirrors and the piece so reflections cover both target clusters"],
            "subgoals_l2":  [''                                                                                   ],
            "goal_flow":    "Use both vertical and horizontal mirrors to fan out the single piece's reflection onto two separate target clusters.",
            "goal_flow_l2": ''                                                                                     ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["2 pieces + horizontal mirror + vertical mirror; larger step budget"],
            "gimmicks_l2":  [''                                                               ],
            "subgoals":     ["Select and position each piece; adjust both mirror axes to cover scattered targets"],
            "subgoals_l2":  [''                                                                                                   ],
            "goal_flow":    "Coordinate two pieces with both mirrors to cover a complex, spread-out target pattern.",
            "goal_flow_l2": ''                                                                                      ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["2 pieces + horizontal mirror + vertical mirror; fork and curve shapes"],
            "gimmicks_l2":  [''                                                ],
            "subgoals":     ["Use complex piece shapes with dual mirrors to fill all target positions"],
            "subgoals_l2":  [''                                                                       ],
            "goal_flow":    "Position the fork and S-curve pieces under both mirrors to generate reflections covering all targets.",
            "goal_flow_l2": ''                                                                                                   ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["2 pieces + horizontal mirror + vertical mirror; tl-shape and z5-shape"],
            "gimmicks_l2":  [''                               ],
            "subgoals":     ["Navigate both pieces under two mirrors to match the target dot pattern"],
            "subgoals_l2":  [''                                                                                ],
            "goal_flow":    "Align TL and Z5 pieces with both mirrors to reflect onto all scattered target dots.",
            "goal_flow_l2": ''                                                                               ,
            "difficulty":   "very_hard",
        },
    ]

    analysis = _analyses[level_index] if level_index < len(_analyses) else {}

    return {
        "level_index":          level_index + 1,
        "grid_size":            list(grid),
        "step_counter":         step_count,
        "target_dot_count":     len(target_dot_positions),
        "target_dot_positions": target_dot_positions,
        "sprites":              level_sprites_info,
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
        "game_id":          "ar25",
        "title":            "AR25",
        "title_l2":         "AR25",
        "description":      "Reflect pieces across mirror axes to cover all target dots within the step budget.",
        "description_l2":   ''                                                                                             ,
        "default_grid_size": [21, 21],
        "num_levels":        len(levels),
        "goal":             "Cover all target_dot sprites with reflected images of pieces.",
        "goal_l2":          ''                                                               ,
        "win_condition":    "_check_win() returns True → next_level() called",
        "win_condition_l2": ''                                                   ,
        "lose_condition":   "Step counter reaches 0 → self.lose() called",
        "lose_condition_l2":''                                                                    ,
        "actions": {
            "ACTION1": {
                "key":             "Up arrow",
                "effect":          "Move selected piece/mirror up by 1 grid unit",
                "effect_l2":       ''                                                     ,
                "restrictions":    "Cannot move outside grid bounds; mirror_v_axis only moves horizontally",
                "restrictions_l2": ''                                                                      ,
                "step_cost":       1,
            },
            "ACTION2": {
                "key":             "Down arrow",
                "effect":          "Move selected piece/mirror down by 1 grid unit",
                "effect_l2":       ''                                                     ,
                "restrictions":    "Cannot move outside grid bounds; mirror_v_axis only moves horizontally",
                "restrictions_l2": ''                                                                      ,
                "step_cost":       1,
            },
            "ACTION3": {
                "key":             "Left arrow",
                "effect":          "Move selected piece/mirror left by 1 grid unit",
                "effect_l2":       ''                                                     ,
                "restrictions":    "Cannot move outside grid bounds; mirror_h_axis only moves vertically",
                "restrictions_l2": ''                                                                      ,
                "step_cost":       1,
            },
            "ACTION4": {
                "key":             "Right arrow",
                "effect":          "Move selected piece/mirror right by 1 grid unit",
                "effect_l2":       ''                                                     ,
                "restrictions":    "Cannot move outside grid bounds; mirror_h_axis only moves vertically",
                "restrictions_l2": ''                                                                      ,
                "step_cost":       1,
            },
            "ACTION5": {
                "key":             "Space / Tab",
                "effect":          "Cycle selection to the next piece/mirror in the list",
                "effect_l2":       ''                                              ,
                "restrictions":    "Available only when multiple selectable sprites exist",
                "restrictions_l2": ''                                                                     ,
                "step_cost":       1,
            },
            "ACTION6": {
                "key":             "Click",
                "effect":          "Select the piece/mirror at the clicked grid coordinate",
                "effect_l2":       ''                                                             ,
                "restrictions":    "Only selects sprites without mirror_v_fixed tag",
                "restrictions_l2": ''                                                               ,
                "step_cost":       0,
            },
            "ACTION7": {
                "key":             "Z / Backspace",
                "effect":          "Undo the last move, restoring previous positions",
                "effect_l2":       ''                                                ,
                "restrictions":    "Only available if undo history exists",
                "restrictions_l2": ''                                             ,
                "step_cost":       0,
            },
        },
        "sprite_roles": {
            "piece":            "A shape that is reflected by mirror axes to create images; player moves this",
            "piece_l2":         ''                                                                             ,
            "mirror_axis":      "A mirror axis (vertical or horizontal line) that reflects pieces",
            "mirror_axis_l2":   ''                                                         ,
            "mirror_v_fixed":   "A fixed vertical mirror axis that cannot be selected or moved",
            "mirror_v_fixed_l2":''                                       ,
            "mirror_v_axis":    "A selectable vertical mirror axis; moves left/right only",
            "mirror_v_axis_l2": ''                                                     ,
            "mirror_h_axis":    "A selectable horizontal mirror axis; moves up/down only",
            "mirror_h_axis_l2": ''                                                     ,
            "target_dot":       "A 1x1 yellow dot that must be covered by a reflected piece image to win",
            "target_dot_l2":    ''                                                          ,
            "auto_rotate_h":    "A piece that automatically rotates 90° when distance to horizontal axis changes",
            "auto_rotate_h_l2": ''                                                                         ,
            "auto_rotate_v":    "A piece that automatically rotates 90° when distance to vertical axis changes",
            "auto_rotate_v_l2": ''                                                                         ,
            "reflect_h_only":   "A piece that only reflects across the horizontal axis (not vertical)",
            "reflect_h_only_l2":''                                                                           ,
        },
        "mechanics": {
            "reflection":       "Each piece is reflected across every mirror axis on the level. Reflected pixels fill 'virtual' positions on the grid.",
            "reflection_l2":    ''                                                                                                                           ,
            "win_check":        "After each move, the game checks if all target_dot positions are occupied by reflected images. If yes, level clears.",
            "win_check_l2":     ''                                                                                                                                          ,
            "step_budget":      "Each movement action (ACTION1-5) costs 1 step. When current_steps reaches 0, the player loses.",
            "step_budget_l2":   ''                                                                                                     ,
            "undo":             "ACTION7 restores the previous state (piece/mirror positions). Undo history is stored per level.",
            "undo_l2":          ''                                                                                                                 ,
            "auto_rotate":      "Pieces with auto_rotate_h or auto_rotate_v tags automatically rotate 90° when the corresponding mirror axis moves.",
            "auto_rotate_l2":   ''                                                                                                                  ,
            "multi_reflection": "Reflections can chain: a piece reflected across one axis can be further reflected across another (up to 12 bounces).",
            "multi_reflection_l2": ''                                                                                                                          ,
        },
    },
    "levels": [
        _build_level_knowledge(i, lv)
        for i, lv in enumerate(levels)
    ],
}


class StepCounter(RenderableUserDisplay):
    """"""
    def __init__(self, oecwkhnijk: "Ar25", fefqmpmfqu: int, rfftsilxlp: int = 1):
        self.fefqmpmfqu: int = fefqmpmfqu
        self.current_steps: int = fefqmpmfqu
        self.rfftsilxlp: int = max(1, rfftsilxlp)
        self.oecwkhnijk = oecwkhnijk
        self.energy_colors: list[int] = [
            dqailkzmtf,
            rrdyedysxy,
            eqoiopyobo,
            iuqegctory,
            usrptgpomh,
        ]

    def set_steps(self, pknielutwu: int) -> None:
        self.current_steps = max(0, min(pknielutwu, self.fefqmpmfqu))

    def decrement(self) -> bool:
        if self.current_steps > 0:
            self.current_steps -= 1
        return self.current_steps > 0

    def reset(self) -> None:
        self.current_steps = self.fefqmpmfqu

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.fefqmpmfqu == 0:
            return frame
        fpfjuxlmqs: int = 64
        start_y: int = 0
        start_x: int = 63
        jwuswyyoje: int = (self.fefqmpmfqu + fpfjuxlmqs - 1) // fpfjuxlmqs
        if jwuswyyoje < 1:
            jwuswyyoje = 1
        if jwuswyyoje > len(self.energy_colors):
            jwuswyyoje = len(self.energy_colors)
        gbomrqujoq: int = fpfjuxlmqs * jwuswyyoje
        uojfddscbm: int = self.fefqmpmfqu - self.current_steps
        if uojfddscbm < 0:
            uojfddscbm = 0
        if uojfddscbm > gbomrqujoq:
            uojfddscbm = gbomrqujoq
        obauionezz: int = uojfddscbm // fpfjuxlmqs
        anejthzpxk: int = uojfddscbm % fpfjuxlmqs
        if obauionezz >= jwuswyyoje:
            obauionezz = jwuswyyoje - 1
            anejthzpxk = fpfjuxlmqs
        ocrqvjqfsf = self.energy_colors[obauionezz]
        vcoyigquau: bool = obauionezz == jwuswyyoje - 1
        for i in range(anejthzpxk, fpfjuxlmqs):
            frame[start_y + i, start_x] = ocrqvjqfsf
        if not vcoyigquau and anejthzpxk > 0:
            zphaaygjdr = self.energy_colors[obauionezz + 1]
            for i in range(anejthzpxk):
                frame[start_y + i, start_x] = zphaaygjdr
        return frame


class MiniMapDisplay(RenderableUserDisplay):
    """"""

    def __init__(self, oecwkhnijk: "Ar25"):
        self.oecwkhnijk = oecwkhnijk

    def _color_priority(self, vrolnlpyvl: int) -> int:
        if vrolnlpyvl == dqailkzmtf:
            return 4
        elif vrolnlpyvl == tuovamceti:
            return 3
        elif vrolnlpyvl == euzrpaakrj:
            return 3
        elif vrolnlpyvl == xwfrtwinpo:
            return 2
        elif vrolnlpyvl == liirwepfdj:
            return 1
        return 0

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        frame_height, frame_width = frame.shape
        scale = min(64 // self.oecwkhnijk.huzumkfia, 64 // self.oecwkhnijk.aqnahsxpq)
        nqsnzgwfgp = math.ceil((64 - self.oecwkhnijk.huzumkfia * scale) / 2) + 1
        qsijnilwrj = math.ceil((64 - self.oecwkhnijk.aqnahsxpq * scale) / 2) + 1
        dfmyjxsddb = self.oecwkhnijk._compute_sprite_map()
        ngewfmuylt = np.full((self.oecwkhnijk.aqnahsxpq, self.oecwkhnijk.huzumkfia), -1, dtype=int)
        for hzmlhaicav in range(self.oecwkhnijk.aqnahsxpq):
            for hawhmrorex in range(self.oecwkhnijk.huzumkfia):
                sprite = dfmyjxsddb[hzmlhaicav, hawhmrorex]
                if sprite is None:
                    continue
                kycvqcblbi = hzmlhaicav - sprite.y
                tiqvogucvv = hawhmrorex - sprite.x
                qklrsqlyzy = 0 <= tiqvogucvv < sprite.width and 0 <= kycvqcblbi < sprite.height and (sprite.pixels[kycvqcblbi, tiqvogucvv] != kevchhhmha)
                jrxgqmfjrf = sprite == self.oecwkhnijk.llludejph
                xdjxosbrhv = "mirror_v_fixed" in sprite.tags
                folcdenxmb = None
                for uvgqszxsnw in self.oecwkhnijk.mqedygxur:
                    if uvgqszxsnw.x == hawhmrorex and uvgqszxsnw.y == hzmlhaicav:
                        folcdenxmb = uvgqszxsnw.pixels[0, 0]
                        break
                if folcdenxmb is not None:
                    kkyplokzmb = dqailkzmtf
                elif qklrsqlyzy:
                    if xdjxosbrhv:
                        kkyplokzmb = tuovamceti
                    else:
                        kkyplokzmb = xwfrtwinpo if jrxgqmfjrf else liirwepfdj
                else:
                    continue
                zgbnjdlufq = ngewfmuylt[hzmlhaicav, hawhmrorex]
                if zgbnjdlufq == -1 or self._color_priority(kkyplokzmb) > self._color_priority(zgbnjdlufq):
                    ngewfmuylt[hzmlhaicav, hawhmrorex] = kkyplokzmb
        for sprite in self.oecwkhnijk.khupblbrxc:
            jrxgqmfjrf = sprite == self.oecwkhnijk.llludejph
            xdjxosbrhv = "mirror_v_fixed" in sprite.tags
            for i in range(sprite.pixels.shape[0]):
                for ezmsggksvd in range(sprite.pixels.shape[1]):
                    if sprite.pixels[i, ezmsggksvd] > -1:
                        hawhmrorex = sprite.x + ezmsggksvd
                        hzmlhaicav = sprite.y + i
                        folcdenxmb = None
                        for uvgqszxsnw in self.oecwkhnijk.mqedygxur:
                            if uvgqszxsnw.x == hawhmrorex and uvgqszxsnw.y == hzmlhaicav:
                                folcdenxmb = uvgqszxsnw.pixels[0, 0]
                                break
                        if folcdenxmb is not None:
                            kkyplokzmb = dqailkzmtf
                        elif xdjxosbrhv:
                            kkyplokzmb = euzrpaakrj
                        else:
                            kkyplokzmb = xwfrtwinpo if jrxgqmfjrf else liirwepfdj
                        if 0 <= hawhmrorex < dfmyjxsddb.shape[1] and 0 <= hzmlhaicav < dfmyjxsddb.shape[0]:
                            idcmwgepge = dfmyjxsddb[hzmlhaicav, hawhmrorex]
                            if idcmwgepge is not None:
                                kycvqcblbi = hzmlhaicav - idcmwgepge.y
                                tiqvogucvv = hawhmrorex - idcmwgepge.x
                                qklrsqlyzy = 0 <= tiqvogucvv < idcmwgepge.width and 0 <= kycvqcblbi < idcmwgepge.height and (idcmwgepge.pixels[kycvqcblbi, tiqvogucvv] != kevchhhmha)
                                if not qklrsqlyzy and kkyplokzmb == liirwepfdj:
                                    continue
                        map_height, map_width = ngewfmuylt.shape
                        if 0 <= hawhmrorex < map_width and 0 <= hzmlhaicav < map_height:
                            zgbnjdlufq = ngewfmuylt[hzmlhaicav, hawhmrorex]
                            if zgbnjdlufq == -1 or self._color_priority(kkyplokzmb) > self._color_priority(zgbnjdlufq):
                                ngewfmuylt[hzmlhaicav, hawhmrorex] = kkyplokzmb
        for hzmlhaicav in range(self.oecwkhnijk.aqnahsxpq):
            for hawhmrorex in range(self.oecwkhnijk.huzumkfia):
                kkyplokzmb = ngewfmuylt[hzmlhaicav, hawhmrorex]
                if kkyplokzmb == -1:
                    continue
                ryzdnnmomo = hawhmrorex * scale + nqsnzgwfgp + 1
                wfwvgmoxtx = hzmlhaicav * scale + qsijnilwrj + 1
                if scale == 5:
                    if 0 <= ryzdnnmomo < frame_width and 0 <= wfwvgmoxtx < frame_height:
                        frame[wfwvgmoxtx, ryzdnnmomo] = kkyplokzmb
                elif scale == 4:
                    for dx, dy in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
                        x = ryzdnnmomo + dx
                        y = wfwvgmoxtx + dy
                        if 0 <= x < frame_width and 0 <= y < frame_height:
                            frame[y, x] = kkyplokzmb
                elif scale == 3:
                    if 0 <= ryzdnnmomo - 2 < frame_width and 0 <= wfwvgmoxtx - 2 < frame_height:
                        frame[wfwvgmoxtx - 2, ryzdnnmomo - 2] = kkyplokzmb
        return frame


class GameState(TypedDict):
    """"""
    khupblbrxc: list[tuple[str, int, int]]
    migkdsjrwk: list[tuple[int, int]]


class Ar25(ARCBaseGame):
    """"""

    def __init__(self) -> None:
        self.cxnxzbeld = MiniMapDisplay(self)
        fefqmpmfqu = 0
        self.zdrbnrjbr = StepCounter(self, fefqmpmfqu, rfftsilxlp=len(levels))
        ehifjxsph = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.cxnxzbeld, self.zdrbnrjbr],
        )
        self.usukvgwle: list[GameState] = []




        self._log_state = ActionLogState(
            aggregator   = default_aggregator,
            post_process = self._compose_summary,
        )

        self._frame_sprite_log:       list[list[dict]] = []
        self._frame_nonsprite_log:    list[list[dict]] = []

        super().__init__(
            game_id="ar25",
            levels=levels,
            camera=ehifjxsph,
            available_actions=[1, 2, 3, 4, 5, 6, 7],
        )

    def _init_step_counter(self) -> None:
        """"""
        agpgjqalgv = self.current_level.get_data("StepCounter")
        if agpgjqalgv:
            self.zdrbnrjbr.fefqmpmfqu = agpgjqalgv
            self.zdrbnrjbr.reset()

    def _calc_mirror_dist(self, sprite: Sprite, runybwgioh: Sprite) -> int:
        if "mirror_v_axis" in runybwgioh.tags:
            zrmehuuqoy = sprite.x + self._sprite_width(sprite) // 2
            return abs(zrmehuuqoy - runybwgioh.x)
        else:
            sdtbfmcgut = sprite.y + self._sprite_height(sprite) // 2
            return abs(sdtbfmcgut - runybwgioh.y)

    def _rotate_piece(self, sprite: Sprite) -> None:
        goyuaczygh = sprite.pixels
        dqapknfnfi = np.rot90(goyuaczygh)
        jevahxujmi = goyuaczygh.shape[1]
        qwjhvoblod = goyuaczygh.shape[0]
        gynarddzug = sprite.x + jevahxujmi // 2
        jbbgooakjc = sprite.y + qwjhvoblod // 2
        new_h, new_w = dqapknfnfi.shape
        yptggrfrgo = gynarddzug - new_w // 2
        wymrczshbz = jbbgooakjc - new_h // 2
        sprite.pixels = dqapknfnfi
        sprite.set_position(yptggrfrgo, wymrczshbz)
        sprite.set_position(yptggrfrgo, wymrczshbz)

    def _save_state(self) -> GameState:
        """"""
        lhduxvcvxy: list[tuple[int, int]] = [(s.x, s.y) for s in self.migkdsjrwk]
        tuzvcwvwfq: list[tuple[str, int, int]] = []
        for trspysetcl in self.khupblbrxc:
            wftxejjopq = "mirror_v_axis" if "mirror_v_axis" in trspysetcl.tags else "mirror_h_axis" if "mirror_h_axis" in trspysetcl.tags else ""
            tuzvcwvwfq.append((wftxejjopq, trspysetcl.x, trspysetcl.y))
        return {"migkdsjrwk": lhduxvcvxy, "khupblbrxc": tuzvcwvwfq}

    def _restore_state(self, state: GameState) -> None:
        """"""
        for i, (x, y) in enumerate(state["migkdsjrwk"]):
            s = self.migkdsjrwk[i]
            s.set_position(x, y)
        for wftxejjopq, x, y in state["khupblbrxc"]:
            if not wftxejjopq:
                continue
            trspysetcl = next((ax for ax in self.khupblbrxc if wftxejjopq in ax.tags), None)
            if trspysetcl:
                trspysetcl.set_position(x, y)
        self._update_overlays()

    def on_set_level(self, level: Level) -> None:
        self.ijwojcjri = 0
        self.tikqecuwc = False
        self.mewxgwwty: bool = False
        self.cnlzdmmso: int = 0
        self.uehyvizcj = sprites["anim_flash"].clone()
        self.current_level.add_sprite(self.uehyvizcj)
        self.uehyvizcj._x = 500
        self.ueryufapb = {}
        self.mllqetvjb = False
        self.llludejph: Sprite | None = None
        sngtmtozd = self.current_level.grid_size
        if sngtmtozd is not None:
            self.huzumkfia = sngtmtozd[0]
            self.aqnahsxpq = sngtmtozd[1]
        self.migkdsjrwk: list[Sprite] = self.current_level.get_sprites_by_tag("piece")
        self.mqedygxur: list[Sprite] = self.current_level.get_sprites_by_tag("target_dot")
        self.vakuvqumo = sprites["overlay_blank"].clone()
        self.vakuvqumo._layer = 0
        self.current_level.add_sprite(self.vakuvqumo)
        self.ehoxfqdzf = sprites["overlay_blank"].clone()
        self.ehoxfqdzf._layer = -1
        self.current_level.add_sprite(self.ehoxfqdzf)
        self.jgzwiwsnn = sprites["overlay_blank"].clone()
        self.jgzwiwsnn._layer = -2
        self.current_level.add_sprite(self.jgzwiwsnn)
        self.khupblbrxc = self.current_level.get_sprites_by_tag("mirror_axis")
        kefcousrdq = [trspysetcl for trspysetcl in self.khupblbrxc if "mirror_v_fixed" not in trspysetcl.tags]
        lybxqzyjle = [rhxpwsowco for rhxpwsowco in self.migkdsjrwk if "mirror_v_fixed" not in rhxpwsowco.tags]
        self.llludejph = kefcousrdq[0] if kefcousrdq else lybxqzyjle[0] if lybxqzyjle else None
        self.hnepuikbu = next(
            (trspysetcl for trspysetcl in self.khupblbrxc if "mirror_v_axis" in trspysetcl.tags),
            None,
        )
        self.hitrtbsoq = next(
            (trspysetcl for trspysetcl in self.khupblbrxc if "mirror_h_axis" in trspysetcl.tags),
            None,
        )
        for eusordagn in self.migkdsjrwk:
            if "auto_rotate_v" in eusordagn.tags and self.hnepuikbu:
                self.ueryufapb[eusordagn] = self._calc_mirror_dist(eusordagn, self.hnepuikbu)
            elif "auto_rotate_h" in eusordagn.tags and self.hitrtbsoq:
                self.ueryufapb[eusordagn] = self._calc_mirror_dist(eusordagn, self.hitrtbsoq)
        self._init_step_counter()
        self._update_overlays()
        self._check_win()
        self.usukvgwle = []
        self.xefwpvwoh: list[Sprite] = []
        for trspysetcl in self.khupblbrxc:
            self.xefwpvwoh.append(trspysetcl)
        for rhxpwsowco in self.migkdsjrwk:
            self.xefwpvwoh.append(rhxpwsowco)
        for niavljcgw in range(len(self.xefwpvwoh) - 1, -1, -1):
            if "mirror_v_fixed" in self.xefwpvwoh[niavljcgw].tags:
                del self.xefwpvwoh[niavljcgw]









        _LAYOUT_ROLE_L2 = {
            "mirror_axis": "",
            "piece":       "",
            "target_dot":  "",
        }
        _layout_rows: list[dict] = []
        for _role, _role_l2 in _LAYOUT_ROLE_L2.items():
            for _spr in self.current_level.get_sprites_by_tag(_role):
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


        try:
            _ns_status = self.get_nonsprite_status() or []
        except Exception:
            _ns_status = []
        for _ns in _ns_status:
            if _ns.get("name") != "steps":
                continue
            _layout_rows.append({
                "role":    "remaining_step",
                "role_l2": ''                     ,
                "sname":   "steps",
                "dx":      int(_ns.get("x", 0)),
                "dy":      int(_ns.get("y", 0)),
                "dw":      int(_ns.get("width", 0)),
                "dh":      int(_ns.get("height", 0)),
            })





        try:
            _refl_bboxes = self._compute_reflection_bboxes(max_depth=2)
        except Exception:
            _refl_bboxes = []
        _arc = min(64 // self.camera.width, 64 // self.camera.height)
        for _rb in _refl_bboxes:
            _rx, _ry = int(_rb.get("x", 0)), int(_rb.get("y", 0))
            _rw, _rh = int(_rb.get("width", 0)), int(_rb.get("height", 0))

            on_screen = (
                _rx + _rw > 0 and _rx < self.huzumkfia
                and _ry + _rh > 0 and _ry < self.aqnahsxpq
            )
            if not on_screen:
                continue
            _rdx, _rdy = self._to_display_xy(_rx, _ry)
            _rdw, _rdh = _rw * _arc, _rh * _arc
            _layout_rows.append({
                "role":    "reflection",
                "role_l2": ''      ,
                "sname":   _rb.get("name", "reflection"),
                "dx":      _rdx, "dy": _rdy,
                "dw":      _rdw, "dh": _rdh,
            })

        if _layout_rows:
            self._set_event("initial_layout", _layout_rows)

    def _sprite_width(self, s: Sprite) -> int:
        return s.pixels.shape[1] if s.pixels.shape else 0

    def _sprite_height(self, s: Sprite) -> int:
        return s.pixels.shape[0] if s.pixels.shape else 0

    def _sprite_color(self, s: Sprite) -> int:
        ozxyheehir = np.unique(s.pixels[s.pixels != kevchhhmha])
        if len(ozxyheehir) == 0:
            return -1
        return int(ozxyheehir[0])

    def _compute_reflection_map(self, xjxgkzyiuh: Sprite | None, cfafqujddu: bool = True) -> np.ndarray:
        vxzwenyfbf = np.full((self.aqnahsxpq, self.huzumkfia), -1, dtype=int)
        if not cfafqujddu:
            for sprite in self.current_level._sprites:
                if sprite == xjxgkzyiuh or ("piece" not in sprite.tags and "border" not in sprite.tags):
                    continue
                nqnagkqeqy, crzzncewfw = sprite.pixels.shape
                for dtbpphzyfe in range(nqnagkqeqy):
                    for rbpwwdlqqb in range(crzzncewfw):
                        pmydxcmxwv = sprite.pixels[dtbpphzyfe, rbpwwdlqqb]
                        if pmydxcmxwv != kevchhhmha:
                            hawhmrorex = sprite.x + rbpwwdlqqb
                            hzmlhaicav = sprite.y + dtbpphzyfe
                            if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                                vxzwenyfbf[hzmlhaicav, hawhmrorex] = pmydxcmxwv
            return vxzwenyfbf
        qigdoqofte = [s for s in self.current_level._sprites if s != xjxgkzyiuh and ("piece" in s.tags or "border" in s.tags)]
        for sprite in reversed(qigdoqofte):
            ritcbtdspa: deque[tuple[tuple[int, int], int, int]] = deque()
            yqbcaytwyh = set()
            sqwibixhfs = 12
            nqnagkqeqy, crzzncewfw = sprite.pixels.shape
            sdytknuhzh = "piece" in sprite.tags
            for dtbpphzyfe in range(nqnagkqeqy):
                for rbpwwdlqqb in range(crzzncewfw):
                    pmydxcmxwv = sprite.pixels[dtbpphzyfe, rbpwwdlqqb]
                    if pmydxcmxwv != kevchhhmha:
                        hawhmrorex = sprite.x + rbpwwdlqqb
                        hzmlhaicav = sprite.y + dtbpphzyfe
                        rbppudsffs = (hawhmrorex, hzmlhaicav)
                        if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                            if rbppudsffs not in yqbcaytwyh:
                                yqbcaytwyh.add(rbppudsffs)
                                ritcbtdspa.append((rbppudsffs, pmydxcmxwv, 0))
            while ritcbtdspa:
                rbppudsffs, pmydxcmxwv, depth = ritcbtdspa.popleft()
                if depth > sqwibixhfs:
                    continue
                hawhmrorex, hzmlhaicav = rbppudsffs
                for runybwgioh in self.khupblbrxc:
                    if "mirror_v_axis" in runybwgioh.tags:
                        vqnsvotjkg = 2 * runybwgioh.x - hawhmrorex
                        dhxrbswmvp = hzmlhaicav
                        xqebhtjxoa = "mirror_v_axis"
                    elif "mirror_h_axis" in runybwgioh.tags:
                        vqnsvotjkg = hawhmrorex
                        dhxrbswmvp = 2 * runybwgioh.y - hzmlhaicav
                        xqebhtjxoa = "mirror_h_axis"
                    else:
                        continue
                    if sdytknuhzh and ("reflect_horizontal_only" in sprite.tags and xqebhtjxoa != "mirror_h_axis" or ("reflect_h_only" in sprite.tags and xqebhtjxoa != "mirror_v_axis")):
                        continue
                    ikqluvhyva = (vqnsvotjkg, dhxrbswmvp)
                    if ikqluvhyva in yqbcaytwyh:
                        continue
                    yqbcaytwyh.add(ikqluvhyva)
                    ritcbtdspa.append((ikqluvhyva, pmydxcmxwv, depth + 1))
                    if 0 <= vqnsvotjkg < self.huzumkfia and 0 <= dhxrbswmvp < self.aqnahsxpq:
                        if vxzwenyfbf[dhxrbswmvp, vqnsvotjkg] == -1:
                            vxzwenyfbf[dhxrbswmvp, vqnsvotjkg] = kddxpzgafq
        for sprite in qigdoqofte:
            nqnagkqeqy, crzzncewfw = sprite.pixels.shape
            for dtbpphzyfe in range(nqnagkqeqy):
                for rbpwwdlqqb in range(crzzncewfw):
                    pmydxcmxwv = sprite.pixels[dtbpphzyfe, rbpwwdlqqb]
                    if pmydxcmxwv != kevchhhmha:
                        hawhmrorex = sprite.x + rbpwwdlqqb
                        hzmlhaicav = sprite.y + dtbpphzyfe
                        if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                            vxzwenyfbf[hzmlhaicav, hawhmrorex] = pmydxcmxwv
        return vxzwenyfbf

    def _check_pixel_collision(
        self,
        yptggrfrgo: int,
        wymrczshbz: int,
        dqapknfnfi: np.ndarray,
        vxzwenyfbf: np.ndarray,
        vovwcdkblf: int,
    ) -> bool:
        nqnagkqeqy, crzzncewfw = dqapknfnfi.shape
        for dtbpphzyfe in range(nqnagkqeqy):
            for rbpwwdlqqb in range(crzzncewfw):
                if dqapknfnfi[dtbpphzyfe, rbpwwdlqqb] != kevchhhmha:
                    hawhmrorex = yptggrfrgo + rbpwwdlqqb
                    hzmlhaicav = wymrczshbz + dtbpphzyfe
                    if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                        djewmaduhg = vxzwenyfbf[hzmlhaicav, hawhmrorex]
                        if djewmaduhg != -1 and djewmaduhg != vovwcdkblf:
                            return True
        return False

    def _check_mirror_overlap(
        self,
        yptggrfrgo: int,
        wymrczshbz: int,
        dqapknfnfi: np.ndarray,
        zrrumfochl: np.ndarray,
        sprite: Sprite,
    ) -> bool:
        from collections import deque

        khupblbrxc = self.khupblbrxc
        nqnagkqeqy, crzzncewfw = dqapknfnfi.shape
        yqbcaytwyh = set()
        ritcbtdspa: deque[tuple[tuple[int, int], int]] = deque()
        sqwibixhfs = 12
        for dtbpphzyfe in range(nqnagkqeqy):
            for rbpwwdlqqb in range(crzzncewfw):
                if dqapknfnfi[dtbpphzyfe, rbpwwdlqqb] != kevchhhmha:
                    hawhmrorex = yptggrfrgo + rbpwwdlqqb
                    hzmlhaicav = wymrczshbz + dtbpphzyfe
                    rbppudsffs = (hawhmrorex, hzmlhaicav)
                    if rbppudsffs not in yqbcaytwyh:
                        yqbcaytwyh.add(rbppudsffs)
                        ritcbtdspa.append((rbppudsffs, 0))
        while ritcbtdspa:
            rbppudsffs, depth = ritcbtdspa.popleft()
            if depth > sqwibixhfs:
                continue
            hawhmrorex, hzmlhaicav = rbppudsffs
            if depth > 0 and 0 <= hawhmrorex < self.huzumkfia and (0 <= hzmlhaicav < self.aqnahsxpq):
                if zrrumfochl[hzmlhaicav, hawhmrorex] != -1:
                    return True
            for runybwgioh in khupblbrxc:
                if "mirror_v_axis" in runybwgioh.tags:
                    vqnsvotjkg = 2 * runybwgioh.x - hawhmrorex
                    dhxrbswmvp = hzmlhaicav
                    xqebhtjxoa = "mirror_v_axis"
                elif "mirror_h_axis" in runybwgioh.tags:
                    vqnsvotjkg = hawhmrorex
                    dhxrbswmvp = 2 * runybwgioh.y - hzmlhaicav
                    xqebhtjxoa = "mirror_h_axis"
                else:
                    continue
                if "reflect_horizontal_only" in sprite.tags and xqebhtjxoa != "mirror_h_axis" or ("reflect_h_only" in sprite.tags and xqebhtjxoa != "mirror_v_axis"):
                    continue
                ikqluvhyva = (vqnsvotjkg, dhxrbswmvp)
                if ikqluvhyva in yqbcaytwyh:
                    continue
                yqbcaytwyh.add(ikqluvhyva)
                ritcbtdspa.append((ikqluvhyva, depth + 1))
        return False

    def _sprites_at(self, x: int, y: int, tag: Optional[str] = None) -> list[Sprite]:
        result = []
        sprites = self.current_level._sprites
        for sprite in sprites:
            if x >= sprite.x and y >= sprite.y and (x < sprite.x + sprite.width) and (y < sprite.y + sprite.height):
                pixels = sprite.pixels
                if pixels[y - sprite.y][x - sprite.x] == -1:
                    continue
                if tag is None or tag in sprite.tags or "mirror_axis" in sprite.tags:
                    result.append(sprite)
        return result

    def _compute_all_reflection_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def eexlmfoxay(bigbpyqxgy: list[Sprite]) -> np.ndarray:
            mqrjslqocj = np.full((self.aqnahsxpq, self.huzumkfia), -1, dtype=int)
            for sprite in reversed(bigbpyqxgy):
                ritcbtdspa: deque[tuple[tuple[int, int], int, int]] = deque()
                yqbcaytwyh = set()
                sqwibixhfs = 12
                nqnagkqeqy, crzzncewfw = sprite.pixels.shape
                for dtbpphzyfe in range(nqnagkqeqy):
                    for rbpwwdlqqb in range(crzzncewfw):
                        pmydxcmxwv = sprite.pixels[dtbpphzyfe, rbpwwdlqqb]
                        if pmydxcmxwv != kevchhhmha:
                            hawhmrorex = sprite.x + rbpwwdlqqb
                            hzmlhaicav = sprite.y + dtbpphzyfe
                            rbppudsffs = (hawhmrorex, hzmlhaicav)
                            if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                                if rbppudsffs not in yqbcaytwyh:
                                    yqbcaytwyh.add(rbppudsffs)
                                    ritcbtdspa.append((rbppudsffs, pmydxcmxwv, 0))
                while ritcbtdspa:
                    rbppudsffs, pmydxcmxwv, depth = ritcbtdspa.popleft()
                    if depth > sqwibixhfs:
                        continue
                    hawhmrorex, hzmlhaicav = rbppudsffs
                    for runybwgioh in self.khupblbrxc:
                        if "mirror_v_axis" in runybwgioh.tags:
                            vqnsvotjkg = 2 * runybwgioh.x - hawhmrorex
                            dhxrbswmvp = hzmlhaicav
                            xqebhtjxoa = "mirror_v_axis"
                        elif "mirror_h_axis" in runybwgioh.tags:
                            vqnsvotjkg = hawhmrorex
                            dhxrbswmvp = 2 * runybwgioh.y - hzmlhaicav
                            xqebhtjxoa = "mirror_h_axis"
                        else:
                            continue
                        if "reflect_horizontal_only" in sprite.tags and xqebhtjxoa != "mirror_h_axis" or ("reflect_h_only" in sprite.tags and xqebhtjxoa != "mirror_v_axis"):
                            continue
                        ikqluvhyva = (vqnsvotjkg, dhxrbswmvp)
                        if ikqluvhyva in yqbcaytwyh:
                            continue
                        yqbcaytwyh.add(ikqluvhyva)
                        ritcbtdspa.append((ikqluvhyva, pmydxcmxwv, depth + 1))
                        if 0 <= vqnsvotjkg < self.huzumkfia and 0 <= dhxrbswmvp < self.aqnahsxpq:
                            if mqrjslqocj[dhxrbswmvp, vqnsvotjkg] == -1:
                                mqrjslqocj[dhxrbswmvp, vqnsvotjkg] = kddxpzgafq
            for sprite in bigbpyqxgy:
                nqnagkqeqy, crzzncewfw = sprite.pixels.shape
                for dtbpphzyfe in range(nqnagkqeqy):
                    for rbpwwdlqqb in range(crzzncewfw):
                        pmydxcmxwv = sprite.pixels[dtbpphzyfe, rbpwwdlqqb]
                        if pmydxcmxwv != kevchhhmha:
                            hawhmrorex = sprite.x + rbpwwdlqqb
                            hzmlhaicav = sprite.y + dtbpphzyfe
                            if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                                mqrjslqocj[hzmlhaicav, hawhmrorex] = pmydxcmxwv
            return mqrjslqocj

        ohpaolwbkp = [s for s in self.migkdsjrwk if "reflect_h_only" not in s.tags and "reflect_horizontal_only" not in s.tags]
        eoizzehkwe = [s for s in self.migkdsjrwk if "reflect_h_only" in s.tags]
        qdffglinvl = [s for s in self.migkdsjrwk if "reflect_horizontal_only" in s.tags]
        oescnslrpz = eexlmfoxay(ohpaolwbkp)
        itbhbmdqlc = eexlmfoxay(eoizzehkwe)
        gjpbeiyllp = eexlmfoxay(qdffglinvl)
        return (oescnslrpz, itbhbmdqlc, gjpbeiyllp)

    def _compute_combined_map(self) -> np.ndarray:
        oescnslrpz, itbhbmdqlc, gjpbeiyllp = self._compute_all_reflection_maps()
        mqrjslqocj = np.full((self.aqnahsxpq, self.huzumkfia), -1, dtype=int)
        for rbqrinektg in [gjpbeiyllp, itbhbmdqlc, oescnslrpz]:
            ejdxlnybyw = rbqrinektg > -1
            mqrjslqocj[ejdxlnybyw] = rbqrinektg[ejdxlnybyw]
        return mqrjslqocj

    def _compute_sprite_map(self) -> np.ndarray:
        dfmyjxsddb = np.full((self.aqnahsxpq, self.huzumkfia), None, dtype=object)
        for sprite in reversed(self.migkdsjrwk):
            ritcbtdspa: deque[tuple[tuple[int, int], int]] = deque()
            yqbcaytwyh = set()
            sqwibixhfs = 12
            nqnagkqeqy, crzzncewfw = sprite.pixels.shape
            for dtbpphzyfe in range(nqnagkqeqy):
                for rbpwwdlqqb in range(crzzncewfw):
                    if sprite.pixels[dtbpphzyfe, rbpwwdlqqb] != kevchhhmha:
                        hawhmrorex = sprite.x + rbpwwdlqqb
                        hzmlhaicav = sprite.y + dtbpphzyfe
                        rbppudsffs = (hawhmrorex, hzmlhaicav)
                        if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                            if rbppudsffs not in yqbcaytwyh:
                                yqbcaytwyh.add(rbppudsffs)
                                ritcbtdspa.append((rbppudsffs, 0))
            while ritcbtdspa:
                rbppudsffs, depth = ritcbtdspa.popleft()
                if depth > sqwibixhfs:
                    continue
                hawhmrorex, hzmlhaicav = rbppudsffs
                for runybwgioh in self.khupblbrxc:
                    if "mirror_v_axis" in runybwgioh.tags:
                        vqnsvotjkg = 2 * runybwgioh.x - hawhmrorex
                        dhxrbswmvp = hzmlhaicav
                        xqebhtjxoa = "mirror_v_axis"
                    elif "mirror_h_axis" in runybwgioh.tags:
                        vqnsvotjkg = hawhmrorex
                        dhxrbswmvp = 2 * runybwgioh.y - hzmlhaicav
                        xqebhtjxoa = "mirror_h_axis"
                    else:
                        continue
                    if "reflect_horizontal_only" in sprite.tags and xqebhtjxoa != "mirror_h_axis" or ("reflect_h_only" in sprite.tags and xqebhtjxoa != "mirror_v_axis"):
                        continue
                    ikqluvhyva = (vqnsvotjkg, dhxrbswmvp)
                    if ikqluvhyva in yqbcaytwyh:
                        continue
                    yqbcaytwyh.add(ikqluvhyva)
                    ritcbtdspa.append((ikqluvhyva, depth + 1))
                    if 0 <= vqnsvotjkg < self.huzumkfia and 0 <= dhxrbswmvp < self.aqnahsxpq:
                        if dfmyjxsddb[dhxrbswmvp, vqnsvotjkg] is None:
                            dfmyjxsddb[dhxrbswmvp, vqnsvotjkg] = sprite
        for sprite in self.migkdsjrwk:
            nqnagkqeqy, crzzncewfw = sprite.pixels.shape
            for dtbpphzyfe in range(nqnagkqeqy):
                for rbpwwdlqqb in range(crzzncewfw):
                    if sprite.pixels[dtbpphzyfe, rbpwwdlqqb] != kevchhhmha:
                        hawhmrorex = sprite.x + rbpwwdlqqb
                        hzmlhaicav = sprite.y + dtbpphzyfe
                        if 0 <= hawhmrorex < self.huzumkfia and 0 <= hzmlhaicav < self.aqnahsxpq:
                            dfmyjxsddb[hzmlhaicav, hawhmrorex] = sprite
        return dfmyjxsddb

    def _update_overlays(self) -> None:
        oescnslrpz, itbhbmdqlc, gjpbeiyllp = self._compute_all_reflection_maps()
        self.vakuvqumo.pixels = oescnslrpz.copy()
        self.ehoxfqdzf.pixels = itbhbmdqlc.copy()
        self.jgzwiwsnn.pixels = gjpbeiyllp.copy()

    def _check_win(self) -> bool:
        mqrjslqocj = self._compute_combined_map()
        qzvyolqwzf = True
        for uvgqszxsnw in self.mqedygxur:
            hawhmrorex = uvgqszxsnw.x
            hzmlhaicav = uvgqszxsnw.y
            if mqrjslqocj[hzmlhaicav, hawhmrorex] < 0:
                qzvyolqwzf = False
        return qzvyolqwzf

    def step(self) -> None:
        """"""
        self.action_log    = []
        self.action_log_l2 = []



        _before_sprites = {
            id(spr): {
                "name": spr.name,
                "x": spr.x, "y": spr.y,
                "width": spr.width, "height": spr.height,
                "rotation": spr.rotation,
                "visible": spr.is_visible,
            }
            for spr in self.current_level.get_sprites()
        }
        _before_steps = self.zdrbnrjbr.current_steps

        self._before_selected = self.llludejph



        self._before_reflections_snapshot = {
            b["name"]: b for b in self._compute_reflection_bboxes(max_depth=2)
        }

        if self.action.id != GameAction.ACTION5:
            self.ijwojcjri = 0
            self.tikqecuwc = False
        if self.cnlzdmmso > 0:
            self.cnlzdmmso += 1
            if self.cnlzdmmso % 2 == 1:
                self.uehyvizcj.set_position(self.migkdsjrwk[0]._x, self.migkdsjrwk[0]._y)
            else:
                self.uehyvizcj.set_position(500, self.migkdsjrwk[0]._y)
            if self.cnlzdmmso >= 8:
                self.cnlzdmmso = -1
                self._post_step(_before_sprites, _before_steps)
                self.complete_action()
            return
        if self.mllqetvjb:
            self.next_level()
            self._post_step(_before_sprites, _before_steps)
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION7:
            if self.usukvgwle:

                state = self.usukvgwle.pop()
                self._restore_state(state)

            self._post_step(_before_sprites, _before_steps)
            self.complete_action()
            return
        if self.llludejph and self.action.id in [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
        ]:
            if "mirror_v_fixed" in self.llludejph.tags:
                self._post_step(_before_sprites, _before_steps)
                self.complete_action()
                return
            pshzrdxfu, iyaddaovv = (0, 0)
            if self.action.id == GameAction.ACTION1:
                iyaddaovv = -1
            elif self.action.id == GameAction.ACTION2:
                iyaddaovv = 1
            elif self.action.id == GameAction.ACTION3:
                pshzrdxfu = -1
            elif self.action.id == GameAction.ACTION4:
                pshzrdxfu = 1
            if "mirror_axis" in self.llludejph.tags and "mirror_v_axis" in self.llludejph.tags:
                iyaddaovv = 0
            if "mirror_axis" in self.llludejph.tags and "mirror_h_axis" in self.llludejph.tags:
                pshzrdxfu = 0
            yptggrfrgo = self.llludejph.x + pshzrdxfu
            wymrczshbz = self.llludejph.y + iyaddaovv
            crzzncewfw = self._sprite_width(self.llludejph)
            nqnagkqeqy = self._sprite_height(self.llludejph)
            _dir_name = {(0,-1):"up",(0,1):"down",(-1,0):"left",(1,0):"right"}.get((pshzrdxfu,iyaddaovv),"none")
            _dir_l2   = {(0,-1):"↑",(0,1):"↓",(-1,0):"←",(1,0):"→"}.get((pshzrdxfu,iyaddaovv),"")
            if yptggrfrgo < 0 or yptggrfrgo + crzzncewfw > self.huzumkfia:
                if not ("mirror_axis" in self.llludejph.tags and "mirror_h_axis" in self.llludejph.tags):





                    self._post_step(_before_sprites, _before_steps)
                    self.complete_action()
                    return
            if wymrczshbz < 0 or wymrczshbz + nqnagkqeqy > self.aqnahsxpq:
                if not ("mirror_axis" in self.llludejph.tags and "mirror_v_axis" in self.llludejph.tags):





                    self._post_step(_before_sprites, _before_steps)
                    self.complete_action()
                    return
            if pshzrdxfu != 0 or iyaddaovv != 0:
                self.usukvgwle.append(self._save_state())
            self.llludejph.set_position(yptggrfrgo, wymrczshbz)
            if pshzrdxfu != 0 or iyaddaovv != 0:
                for s, tgwjjzpyxc in list(self.ueryufapb.items()):
                    runybwgioh = self.hnepuikbu if "auto_rotate_v" in s.tags else self.hitrtbsoq
                    if runybwgioh:
                        vvsapeulig = self._calc_mirror_dist(s, runybwgioh)
                        if vvsapeulig != tgwjjzpyxc:
                            self._rotate_piece(s)
                            self.ueryufapb[s] = self._calc_mirror_dist(s, runybwgioh)
            self._update_overlays()
            if self._check_win():
                self.mllqetvjb = True
                self._post_step(_before_sprites, _before_steps)
                return
            if pshzrdxfu != 0 or iyaddaovv != 0:
                if not self.zdrbnrjbr.decrement():
                    self.lose()
                if self.level_index == 1 and self.cnlzdmmso == 0 and (self.zdrbnrjbr.current_steps < 50) and (not self.mewxgwwty):
                    self.cnlzdmmso = 1


                    self._post_step(_before_sprites, _before_steps)
                    return
            self._post_step(_before_sprites, _before_steps)
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION5:
            self.ijwojcjri += 1
            self.tikqecuwc = True
            uhdkxxlxxb = -1 if self.llludejph not in self.xefwpvwoh else self.xefwpvwoh.index(self.llludejph)
            dnolnhhsxs = (uhdkxxlxxb + 1) % len(self.xefwpvwoh)
            pzxckdzfza = self.xefwpvwoh[dnolnhhsxs]
            if self.llludejph != pzxckdzfza:
                self.llludejph = pzxckdzfza
                self.mewxgwwty = True


            if not self.zdrbnrjbr.decrement():
                self.lose()
            self._post_step(_before_sprites, _before_steps)
            self.complete_action()
            return
        if self.action.id == GameAction.ACTION6:
            x = self.action.data.get("x", 0)
            y = self.action.data.get("y", 0)
            qqxqwaptve = self.camera.display_to_grid(x, y)
            if qqxqwaptve:
                hawhmrorex, hzmlhaicav = qqxqwaptve
                sprites: list[Sprite] = self._sprites_at(hawhmrorex, hzmlhaicav)
                tqxwfhzaui: Sprite | None = None
                ntyhsdyfmi: list[Sprite] = [sprite for sprite in sprites if "piece" in sprite.tags and "mirror_v_fixed" not in sprite.tags]
                if ntyhsdyfmi:

                    def hpjseosbev(s: Sprite) -> int:
                        if "reflect_horizontal_only" in s.tags:
                            return 1
                        elif "reflect_h_only" in s.tags:
                            return 2
                        else:
                            return 3

                    ntyhsdyfmi.sort(key=hpjseosbev, reverse=True)
                    tqxwfhzaui = ntyhsdyfmi[0]
                else:
                    zdtrxuxrrp: list[Sprite] = [sprite for sprite in sprites if "mirror_axis" in sprite.tags and "mirror_v_fixed" not in sprite.tags]
                    if zdtrxuxrrp:
                        if len(zdtrxuxrrp) == 1:
                            tqxwfhzaui = zdtrxuxrrp[0]
                        else:
                            wpolxlughh = next((s for s in zdtrxuxrrp if "mirror_v_axis" in s.tags), None)
                            mluzjwmlbm = next((s for s in zdtrxuxrrp if "mirror_h_axis" in s.tags), None)
                            if self.llludejph is None or self.llludejph not in zdtrxuxrrp:
                                tqxwfhzaui = wpolxlughh if wpolxlughh else mluzjwmlbm
                            elif self.llludejph == wpolxlughh:
                                tqxwfhzaui = mluzjwmlbm
                            elif self.llludejph == mluzjwmlbm:
                                tqxwfhzaui = wpolxlughh
                            else:
                                tqxwfhzaui = wpolxlughh if wpolxlughh else mluzjwmlbm
                if tqxwfhzaui:
                    if self.llludejph != tqxwfhzaui:
                        self.llludejph = tqxwfhzaui
                        self.mewxgwwty = True



            if not self.llludejph:
                aydyobwonq = [trspysetcl for trspysetcl in self.khupblbrxc if "mirror_v_fixed" not in trspysetcl.tags]
                gwzaphwbxm = [rhxpwsowco for rhxpwsowco in self.migkdsjrwk if "mirror_v_fixed" not in rhxpwsowco.tags]
                self.llludejph = aydyobwonq[0] if aydyobwonq else gwzaphwbxm[0] if gwzaphwbxm else None
            self._post_step(_before_sprites, _before_steps)
            self.complete_action()
            return
        self._post_step(_before_sprites, _before_steps)
        self.complete_action()


    def _display_for(self, name: str) -> tuple[str, str, str]:
        """








        """
        result = SPRITE_DISPLAY.get(name)
        if result is not None:
            return result

        base = re.sub(r"_ref_[VH]+$", "", name)
        if base != name and base in SPRITE_DISPLAY:
            return SPRITE_DISPLAY[base]
        return (name, name, name)

    def _to_display_xy(self, x: int, y: int) -> tuple[int, int]:
        """



        """
        cam = self.camera
        arc_scale = min(64 // cam.width, 64 // cam.height)
        x_off = (64 - cam.width  * arc_scale) // 2
        y_off = (64 - cam.height * arc_scale) // 2
        return (x - cam.x) * arc_scale + x_off, (y - cam.y) * arc_scale + y_off


    def _is_mirror_axis_name(self, name: str) -> bool:
        """"""
        t, _, _ = SPRITE_DISPLAY.get(name, ("", "", ""))
        return t == "mirror_axis"

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
        if "mirror_axis" in (spr.tags or []):
            dx = max(0, dx)
            dy = max(0, dy)
        return dx, dy, dw, dh

    def _compute_chain_after(self, x: int, y: int, w: int, h: int, chain: str) -> tuple[int, int]:
        """




        """
        for axis in chain:
            if axis == "V":
                for m in self.khupblbrxc:
                    if "mirror_v_axis" in m.tags:
                        x = 2 * m.x - x - w + 1
                        break
            elif axis == "H":
                for m in self.khupblbrxc:
                    if "mirror_h_axis" in m.tags:
                        y = 2 * m.y - y - h + 1
                        break
        return x, y

    def _bar_pixel_len(self, current_steps: int) -> int:
        """

        """
        total = self.zdrbnrjbr.fefqmpmfqu
        if total == 0:
            return 0
        seg = 64
        tiers = (total + seg - 1) // seg
        if tiers < 1:
            tiers = 1
        if tiers > len(self.zdrbnrjbr.energy_colors):
            tiers = len(self.zdrbnrjbr.energy_colors)
        cap = seg * tiers
        used = total - current_steps
        if used < 0:
            used = 0
        if used > cap:
            used = cap
        return used

    def _post_step(self, before_sprites: dict, before_steps: int) -> None:
        """"""
        after_sprites = {id(spr): spr for spr in self.current_level.get_sprites()}



        for spr_id, spr in after_sprites.items():
            if spr.name in _APPEAR_SUPPRESS:
                continue
            b = before_sprites.get(spr_id)
            if b is None or (not b["visible"] and spr.is_visible):
                dx, dy, dw, dh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(spr.name)
                self.action_log.append(f"[appear]:[{_t}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{_le} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).")
                self.action_log_l2.append(f"[appear]:[{_t}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{_l2}({dx},{dy})|Object({dx},{dy})")



        _arc_scale_d = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in before_sprites.items():
            if b["name"] in _APPEAR_SUPPRESS:
                continue
            spr = after_sprites.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = int(b.get("width", 0))  * _arc_scale_d
                bdh = int(b.get("height", 0)) * _arc_scale_d

                if self._is_mirror_axis_name(b["name"]):
                    bdx, bdy = max(0, bdx), max(0, bdy)
                _t, _le, _l2 = self._display_for(b['name'])
                self.action_log.append(f"[disappear]:[{_t}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{_le} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).")
                self.action_log_l2.append(f"[disappear]:[{_t}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{_l2}({bdx},{bdy})|Object({bdx},{bdy})")


        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            b = before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                continue
            if (spr.x, spr.y) != (b["x"], b["y"]):
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                adx, ady, adw, adh = self._to_display_bbox(spr)

                if self._is_mirror_axis_name(spr.name):
                    bdx, bdy = max(0, bdx), max(0, bdy)
                _t, _le, _l2 = self._display_for(spr.name)
                self.action_log.append(f"[move]:[{_t}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{_le} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady}).")
                self.action_log_l2.append(f"[move]:[{_t}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{_l2}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})")
            if spr.rotation != b["rotation"]:
                _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(spr.name)
                self.action_log.append(f"[rotate]:[{_t}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:{_le} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.")
                self.action_log_l2.append(f"[rotate]:[{_t}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:({_rdx},{_rdy}){_l2}{b['rotation']}°{spr.rotation}°|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°")




        after_steps = self.zdrbnrjbr.current_steps
        if after_steps != before_steps and self._bar_pixel_len(before_steps) != self._bar_pixel_len(after_steps):
            if after_steps < before_steps:
                self.action_log.append(f"[change]:[steps]:[sname(steps);count({before_steps})_({after_steps})]:The yellow step bar on the right edge shrinks from the top ({before_steps}→{after_steps} steps remain).|The yellow Object on the right edge shrinks from the top ({before_steps}→{after_steps}).")
                self.action_log_l2.append(f"[change]:[steps]:[sname(steps);count({before_steps})_({after_steps})]:{before_steps}→{after_steps}|Object{before_steps}→{after_steps}")
            else:
                self.action_log.append(f"[change]:[steps]:[sname(steps);count({before_steps})_({after_steps})]:The yellow step bar on the right edge extends to the top ({before_steps}→{after_steps} steps).|The yellow Object on the right edge extends to the top ({before_steps}→{after_steps}).")
                self.action_log_l2.append(f"[change]:[steps]:[sname(steps);count({before_steps})_({after_steps})]:{before_steps}→{after_steps}|Object{before_steps}→{after_steps}")





        before_sel = getattr(self, "_before_selected", None)
        after_sel  = self.llludejph
        if before_sel is not after_sel:
            if before_sel is not None:
                _bsd = self._to_display_bbox(before_sel)
                _t, _le, _l2 = self._display_for(before_sel.name)
                self.action_log.append(
                    f"[bright]:[{_t}]:[sname({before_sel.name});position({_bsd[0]}_{_bsd[1]});bbox({_bsd[2]}_{_bsd[3]});state(off)]:{_le}'s highlight turns off.|Object's highlight turns off."
                )
                self.action_log_l2.append(
                    f"[bright]:[{_t}]:[sname({before_sel.name});position({_bsd[0]}_{_bsd[1]});bbox({_bsd[2]}_{_bsd[3]});state(off)]:{_l2}|Object"
                )
            if after_sel is not None:
                _asd = self._to_display_bbox(after_sel)
                _t, _le, _l2 = self._display_for(after_sel.name)
                self.action_log.append(
                    f"[bright]:[{_t}]:[sname({after_sel.name});position({_asd[0]}_{_asd[1]});bbox({_asd[2]}_{_asd[3]});state(selected)]:{_le}'s highlight turns on.|Object's highlight turns on."
                )
                self.action_log_l2.append(
                    f"[bright]:[{_t}]:[sname({after_sel.name});position({_asd[0]}_{_asd[1]});bbox({_asd[2]}_{_asd[3]});state(selected)]:{_l2}|Object"
                )


        if self.mllqetvjb:
            self._set_event("win", {
                "en": "[win]:[game]:[]:The level is cleared.|The level is cleared.",
                "l2": "[win]:[game]:[]:|",
            })

        if self.zdrbnrjbr.current_steps == 0 and not self.mllqetvjb:
            self._set_event("gameover", {
                "en": "[gameover]:[game]:[]:Step counter reached zero, game over.|Step counter reached zero, game over.",
                "l2": "[gameover]:[game]:[]:0|0",
            })


        if getattr(self, "uehyvizcj", None) is not None and self.uehyvizcj.x != 500:

            b = before_sprites.get(id(self.uehyvizcj))
            if b is not None and b["x"] == 500:
                _afd = self._to_display_bbox(self.uehyvizcj)
                self.action_log.append(
                    f"[bright]:[anim_flash]:[sname({self.uehyvizcj.name});position({_afd[0]}_{_afd[1]});bbox({_afd[2]}_{_afd[3]});state(flash)]:A flash highlights the target at ({_afd[0]},{_afd[1]}) to indicate a reflection match.|A flash highlights the Object at ({_afd[0]},{_afd[1]}) to indicate a reflection match."
                )
                self.action_log_l2.append(
                    f"[bright]:[anim_flash]:[sname({self.uehyvizcj.name});position({_afd[0]}_{_afd[1]});bbox({_afd[2]}_{_afd[3]});state(flash)]:({_afd[0]},{_afd[1]})|({_afd[0]},{_afd[1]})Object"
                )





        before_refs = getattr(self, "_before_reflections_snapshot", {}) or {}
        after_refs  = {b["name"]: b for b in self._compute_reflection_bboxes(max_depth=2)}

        def _is_onscreen(b: dict) -> bool:
            return (
                b["x"] + b["width"]  > 0 and b["x"] < self.huzumkfia
                and b["y"] + b["height"] > 0 and b["y"] < self.aqnahsxpq
            )


        _arc_scale = min(64 // self.camera.width, 64 // self.camera.height)

        for name, a in after_refs.items():
            if not _is_onscreen(a):
                continue
            prev = before_refs.get(name)
            if prev is None or not _is_onscreen(prev):
                adx, ady = self._to_display_xy(a["x"], a["y"])

                _adw = int(a["width"])  * _arc_scale
                _adh = int(a["height"]) * _arc_scale
                _t, _le, _l2 = self._display_for(name)
                self.action_log.append(
                    f"[appear]:[{_t}]:[sname({name});position({adx}_{ady});bbox({_adw}_{_adh})]:{_le} (reflection, depth {a['depth']}) appears at ({adx},{ady}).|Object (reflection, depth {a['depth']}) appears at ({adx},{ady})."
                )
                self.action_log_l2.append(
                    f"[appear]:[{_t}]:[sname({name});position({adx}_{ady});bbox({_adw}_{_adh})]:{_l2}{a['depth']}({adx},{ady})|Object{a['depth']}({adx},{ady})"
                )








        from collections import defaultdict

        after_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)

        for name, prev in before_refs.items():
            if not _is_onscreen(prev):
                continue
            a = after_refs.get(name)
            if not (a is None or not _is_onscreen(a)):
                continue

            source_name = prev.get("source_piece")
            source_after_spr = None
            if source_name:
                for _sid, _spr in after_sprites.items():
                    if _spr.name == source_name:
                        source_after_spr = _spr
                        break


            if a is not None:
                after_x, after_y = a["x"], a["y"]
                after_w, after_h = a["width"], a["height"]
            elif source_after_spr is not None:
                chain = name.rsplit("_ref_", 1)[-1] if "_ref_" in name else ""
                after_x, after_y = self._compute_chain_after(
                    source_after_spr.x, source_after_spr.y,
                    source_after_spr.width, source_after_spr.height,
                    chain,
                )
                after_w = source_after_spr.width
                after_h = source_after_spr.height
            else:

                after_x, after_y = prev["x"], prev["y"]
                after_w, after_h = prev["width"], prev["height"]

            after_groups[(after_x, after_y)].append({
                "name":             name,
                "depth":            prev["depth"],
                "is_pruned":        a is None,
                "after_w":          after_w,
                "after_h":          after_h,
                "source_after_spr": source_after_spr,
            })


        for (ax, ay), entries in after_groups.items():
            e0 = entries[0]
            ref_names = [e["name"] for e in entries]

            adx, ady = self._to_display_xy(ax, ay)


            adx = max(0, min(63, adx))
            ady = max(0, min(63, ady))
            aw_disp  = e0["after_w"] * _arc_scale
            ah_disp  = e0["after_h"] * _arc_scale




            sname_val = ref_names[0]
            _t, _le, _l2 = self._display_for(sname_val)
            if len(ref_names) == 1:
                en_subj   = f"{_le} (reflection, depth {e0['depth']})"
                l2_subj   = f"{_l2}{e0['depth']}"
                ssub_seg  = ""
            else:
                en_subj   = f"{len(ref_names)} reflections"
                l2_subj   = f"{len(ref_names)}"
                ssub_seg  = f"ssub({','.join(ref_names[1:])});"
            self.action_log.append(
                f"[disappear]:[{_t}]:[{ssub_seg}sname({sname_val});position({adx}_{ady});bbox({aw_disp}_{ah_disp})]:{en_subj} disappears from ({adx},{ady}).|Object disappears from ({adx},{ady})."
            )
            self.action_log_l2.append(
                f"[disappear]:[{_t}]:[{ssub_seg}sname({sname_val});position({adx}_{ady});bbox({aw_disp}_{ah_disp})]:{l2_subj}({adx},{ady})|Object({adx},{ady})"
            )


            pruned = [e for e in entries if e["is_pruned"]]
            if pruned:
                src_spr = next((e["source_after_spr"] for e in pruned if e["source_after_spr"]), None)
                if src_spr is not None:
                    overlap_list = self._log_state.get_event("overlap_list", []) or []
                    overlap_list.append({
                        "reflections": [e["name"] for e in pruned],
                        "source":      src_spr.name,
                        "sdx": adx, "sdy": ady, "sdw": aw_disp, "sdh": ah_disp,
                    })
                    self._set_event("overlap_list", overlap_list)


        for name, a in after_refs.items():
            prev = before_refs.get(name)
            if prev is None:
                continue
            if not (_is_onscreen(prev) and _is_onscreen(a)):
                continue
            if (a["x"], a["y"]) != (prev["x"], prev["y"]):
                pdx, pdy = self._to_display_xy(prev["x"], prev["y"])
                adx, ady = self._to_display_xy(a["x"], a["y"])
                _adw = int(a["width"])  * _arc_scale
                _adh = int(a["height"]) * _arc_scale
                _t, _le, _l2 = self._display_for(name)
                self.action_log.append(
                    f"[move]:[{_t}]:[sname({name});position({pdx}_{pdy})_({adx}_{ady});bbox({_adw}_{_adh})]:{_le} (reflection) moves from ({pdx},{pdy}) to ({adx},{ady}).|Object (reflection) moves from ({pdx},{pdy}) to ({adx},{ady})."
                )
                self.action_log_l2.append(
                    f"[move]:[{_t}]:[sname({name});position({pdx}_{pdy})_({adx}_{ady});bbox({_adw}_{_adh})]:{_l2}({pdx},{pdy})({adx},{ady})|Object({pdx},{pdy})({adx},{ady})"
                )


        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())

        self._log_state.snapshot_step()

    def _set_event(self, key: str, value) -> None:
        """
        """
        self._log_state.set_event(key, value)

    def _compose_summary(
        self,
        agg: list[str],
        agg_l2: list[str],
        events: dict,
    ) -> tuple[list[str], list[str]]:
        """





        """
        agg, agg_l2 = compose_default_summary(agg, agg_l2, events)

        overlap_list = events.get("overlap_list") or []
        if not overlap_list:
            return agg, agg_l2






        ref_names_to_drop: set[str] = set()
        for e in overlap_list:
            for r in e["reflections"]:
                ref_names_to_drop.add(r)


        kept_en: list[str] = []
        kept_l2: list[str] = []
        for line_en, line_l2 in zip(agg, agg_l2):


            drop = line_en.startswith("[disappear]:") and any(
                f"sname({r})" in line_en for r in ref_names_to_drop
            )
            if not drop:
                kept_en.append(line_en)
                kept_l2.append(line_l2)


        for entry in overlap_list:
            src   = entry["source"]
            refs  = entry["reflections"]
            ssub_str = ",".join(refs)
            sdx   = entry["sdx"]; sdy = entry["sdy"]
            sdw   = entry["sdw"]; sdh = entry["sdh"]


            _src_t, _src_le, _src_l2 = self._display_for(src)
            _ref_le = self._display_for(refs[0])[1]
            _ref_l2 = self._display_for(refs[0])[2]
            refs_en = f"{_ref_le} (reflection)" if len(refs) == 1 else f"{len(refs)} reflections"
            refs_l2 = f"{_ref_l2}" if len(refs) == 1 else f"{len(refs)}"



            kept_en.append(
                f"[overlap]:[{_src_t}]:[sname({src});ssub({ssub_str});position({sdx}_{sdy});bbox({sdw}_{sdh})]:"
                f"{_src_le} overlaps {refs_en} at ({sdx},{sdy})."
                f"|Object overlaps Object at ({sdx},{sdy})."
            )
            kept_l2.append(
                f"[overlap]:[{_src_t}]:[sname({src});ssub({ssub_str});position({sdx}_{sdy});bbox({sdw}_{sdh})]:"
                f"{_src_l2}{refs_l2}({sdx},{sdy})"
                f"|Object({sdx},{sdy})"
            )
        return kept_en, kept_l2




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
                "selected": spr is self.llludejph,
            })
        return result

    def _compute_reflection_bboxes(self, max_depth: int = 2) -> list[dict]:
        """














        """
        bboxes: list[dict] = []
        for piece in self.migkdsjrwk:
            piece_tags = list(piece._tags) if piece._tags else []
            skip_v = "reflect_horizontal_only" in piece_tags
            skip_h = "reflect_h_only"           in piece_tags






            seen_positions: set[tuple[int, int, int, int]] = set()
            queue: deque = deque()
            queue.append((
                piece.x, piece.y, piece.width, piece.height,
                0, "", frozenset({(piece.x, piece.y)}),
            ))

            while queue:
                x, y, w, h, depth, chain, path_visited = queue.popleft()
                if depth >= max_depth:
                    continue
                for mirror in self.khupblbrxc:
                    if "mirror_v_axis" in mirror.tags:
                        if skip_v:
                            continue
                        rx = 2 * mirror.x - x - w + 1
                        ry = y
                        axis = "V"
                    elif "mirror_h_axis" in mirror.tags:
                        if skip_h:
                            continue
                        rx = x
                        ry = 2 * mirror.y - y - h + 1
                        axis = "H"
                    else:
                        continue
                    if (rx, ry) in path_visited:

                        continue
                    new_chain  = chain + axis
                    new_depth  = depth + 1
                    pos_key    = (rx, ry, w, h)
                    if pos_key in seen_positions:

                        continue
                    seen_positions.add(pos_key)
                    bboxes.append({
                        "name":         f"{piece.name}_ref_{new_chain}",
                        "depth":        new_depth,
                        "source_piece": piece.name,
                        "chain":        new_chain,
                        "x":            rx,
                        "y":            ry,
                        "width":        w,
                        "height":       h,
                    })
                    queue.append((
                        rx, ry, w, h, new_depth, new_chain,
                        path_visited | {(rx, ry)},
                    ))
        return bboxes

    def get_nonsprite_status(self) -> list[dict]:
        """









        """
        result = []

        result.append({
            "name":     "steps",
            "label":    "Steps Remaining",
            "label_l2": ''               ,
            "value":    self.zdrbnrjbr.current_steps,
            "max":      self.zdrbnrjbr.fefqmpmfqu,
            "x":        63, "y":      0,
            "width":    1,  "height": 64,
        })


        cam = self.camera
        arc_scale = min(64 // cam.width, 64 // cam.height)
        x_off = (64 - cam.width  * arc_scale) // 2
        y_off = (64 - cam.height * arc_scale) // 2
        for bbox in self._compute_reflection_bboxes(max_depth=2):
            dx = (bbox["x"] - cam.x) * arc_scale + x_off
            dy = (bbox["y"] - cam.y) * arc_scale + y_off
            dw = bbox["width"]  * arc_scale
            dh = bbox["height"] * arc_scale
            result.append({
                "name":     bbox["name"],
                "label":    f"Reflection {bbox['depth']}x ({bbox['source_piece']})",
                "label_l2": f"''    {bbox['depth']}''         {bbox['source_piece']})",
                "value":    1,
                "max":      1,
                "x":        dx,
                "y":        dy,
                "width":    dw,
                "height":   dh,
            })
        return result

    def get_game_knowledge(self) -> dict:
        """"""
        return GAME_KNOWLEDGE

    def _get_valid_actions(self) -> list[ActionInput]:
        valid_actions: list[ActionInput] = []
        if self.llludejph is not None:
            if "mirror_v_axis" not in self.llludejph.tags:
                valid_actions.append(ActionInput(id=GameAction.from_id(1)))
                valid_actions.append(ActionInput(id=GameAction.from_id(2)))
            if "mirror_h_axis" not in self.llludejph.tags:
                valid_actions.append(ActionInput(id=GameAction.from_id(3)))
                valid_actions.append(ActionInput(id=GameAction.from_id(4)))
        if len(self.xefwpvwoh) > 1 and self.ijwojcjri < len(self.xefwpvwoh) - 1:
            valid_actions.append(ActionInput(id=GameAction.from_id(5)))
        return valid_actions
