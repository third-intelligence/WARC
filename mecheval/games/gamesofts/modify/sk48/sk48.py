"""




























































"""

import math

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from arcengine.enums import ActionInput

from gamesofts._action_log import ActionLogState


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


PROGRESS_BAR_ROW        = 53
PROGRESS_BAR_FILL_COLOR = LIGHT_GRAY
PROGRESS_BAR_BG_COLOR   = GRAY
STEP_SIZE               = 6




SKEWER_DEFAULT_BORDER   = DARK_GRAY
SKEWER_ACTIVE_BORDER    = WHITE


SEGMENT_ACTIVE_BRIGHT   = OFF_WHITE
SEGMENT_ACTIVE_DIM      = LIGHT_GRAY
SEGMENT_INACTIVE_DIM    = LIGHT_GRAY
SEGMENT_INACTIVE_DARK   = GRAY


WIN_FLASH_COLOR         = WHITE


ROTATION_TO_DIRECTION: dict[int, tuple[int, int]] = {
    0:   (1, 0),
    90:  (0, 1),
    180: (-1, 0),
    270: (0, -1),
}


sprites = {

    "skewer_magenta": Sprite(
        pixels=[
            [MAGENTA, MAGENTA, MAGENTA, MAGENTA, MAGENTA, MAGENTA],
            [MAGENTA, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, MAGENTA],
            [MAGENTA, DARK_GRAY, MAGENTA, MAGENTA, DARK_GRAY, TRANSPARENT],
            [MAGENTA, DARK_GRAY, MAGENTA, MAGENTA, DARK_GRAY, TRANSPARENT],
            [MAGENTA, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, MAGENTA],
            [MAGENTA, MAGENTA, MAGENTA, MAGENTA, MAGENTA, MAGENTA],
        ],
        name="skewer_magenta",
        visible=True,
        collidable=True,
        tags=["skewer"],
        layer=2,
    ),


    "color_block": Sprite(
        pixels=[
            [-2, -2, -2, -2, -2, -2],
            [-2, RED, RED, RED, RED, -2],
            [-2, RED, RED, RED, RED, -2],
            [-2, RED, RED, RED, RED, -2],
            [-2, RED, RED, RED, RED, -2],
            [-2, -2, -2, -2, -2, -2],
        ],
        name="color_block",
        visible=True,
        collidable=True,
        tags=["color_block"],
        layer=2,
    ),

    "floor_tile": Sprite(
        pixels=[
            [DARK_GRAY],
        ],
        name="floor_tile",
        visible=True,
        collidable=True,
        layer=-1,
    ),

    "rail_segment": Sprite(
        pixels=[
            [LIGHT_GRAY, LIGHT_GRAY],
            [LIGHT_GRAY, LIGHT_GRAY],
            [GRAY, GRAY],
            [GRAY, GRAY],
            [GRAY, GRAY],
            [GRAY, GRAY],
            [LIGHT_GRAY, LIGHT_GRAY],
            [LIGHT_GRAY, LIGHT_GRAY],
        ],
        name="rail_segment",
        visible=True,
        collidable=True,
        tags=["rail"],
        layer=-1,
    ),

    "boundary_wall": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="boundary_wall",
        visible=True,
        collidable=True,
        layer=-2,
    ),

    "platform_6x6": Sprite(
        pixels=[
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
        ],
        name="platform_6x6",
        visible=True,
        collidable=True,
        tags=["platform"],
        layer=-1,
    ),


    "match_indicator": Sprite(
        pixels=[
            [-2, -2, -2, -2],
            [-2, WHITE, WHITE, -2],
            [-2, WHITE, WHITE, -2],
            [-2, -2, -2, -2],
        ],
        name="match_indicator",
        visible=True,
        collidable=True,
        layer=2,
    ),

    "platform_7x7": Sprite(
        pixels=[
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
        ],
        name="platform_7x7",
        visible=True,
        collidable=True,
        tags=["platform"],
        layer=-1,
    ),

    "wall_block": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="wall_block",
        visible=True,
        collidable=True,
        tags=["wall_block"],
    ),


    "drop_preview": Sprite(
        pixels=[
            [-2, RED, RED, RED, RED, -2],
            [-2, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, -2],
            [-2, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, -2],
            [-2, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, -2],
            [-2, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, -2],
            [-2, RED, RED, RED, RED, -2],
        ],
        name="drop_preview",
        visible=True,
        collidable=True,
        layer=3,
    ),



    "skewer_segment": Sprite(
        pixels=[
            [-2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2],
            [GRAY, LIGHT_GRAY, LIGHT_GRAY, GRAY, LIGHT_GRAY, LIGHT_GRAY],
            [LIGHT_GRAY, LIGHT_GRAY, GRAY, LIGHT_GRAY, LIGHT_GRAY, GRAY],
            [-2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2],
        ],
        name="skewer_segment",
        visible=True,
        collidable=True,
        tags=["skewer_segment"],
    ),

    "platform_5x5": Sprite(
        pixels=[
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
            [DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY],
        ],
        name="platform_5x5",
        visible=True,
        collidable=False,
        tags=["platform"],
        layer=-1,
    ),

    "skewer_purple": Sprite(
        pixels=[
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
            [PURPLE, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, PURPLE],
            [PURPLE, DARK_GRAY, PURPLE, PURPLE, DARK_GRAY, TRANSPARENT],
            [PURPLE, DARK_GRAY, PURPLE, PURPLE, DARK_GRAY, TRANSPARENT],
            [PURPLE, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, PURPLE],
            [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
        ],
        name="skewer_purple",
        visible=True,
        collidable=True,
        tags=["skewer"],
        layer=2,
    ),


    "skewer_light_blue": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, BLACK],
            [BLACK, DARK_GRAY, LIGHT_BLUE, LIGHT_BLUE, DARK_GRAY, TRANSPARENT],
            [BLACK, DARK_GRAY, LIGHT_BLUE, LIGHT_BLUE, DARK_GRAY, TRANSPARENT],
            [BLACK, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="skewer_light_blue",
        visible=True,
        collidable=True,
        tags=["skewer"],
        layer=2,
    ),

    "floor_strip": Sprite(
        pixels=[
            [GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY],
        ],
        name="floor_strip",
        visible=True,
        collidable=True,
        layer=-1,
    ),


    "skewer_yellow": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, BLACK],
            [BLACK, DARK_GRAY, YELLOW, YELLOW, DARK_GRAY, TRANSPARENT],
            [BLACK, DARK_GRAY, YELLOW, YELLOW, DARK_GRAY, TRANSPARENT],
            [BLACK, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="skewer_yellow",
        visible=True,
        collidable=True,
        tags=["skewer"],
        layer=2,
    ),
}


levels = [

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(11, 36),
            sprites["skewer_magenta"].clone().set_position(20, 56),
            sprites["color_block"].clone().set_position(41, 30).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(41, 24).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(41, 18).color_remap(None, RED),
            sprites["color_block"].clone().set_position(26, 56),
            sprites["color_block"].clone().set_position(32, 56).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(38, 56).color_remap(None, BLUE),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(13, 32),
            sprites["rail_segment"].clone().set_position(13, 26),
            sprites["rail_segment"].clone().set_position(13, 20),
            sprites["rail_segment"].clone().set_position(13, 14),
            sprites["skewer_segment"].clone().set_position(17, 36),
            sprites["skewer_segment"].clone().set_position(20, 56),
            sprites["skewer_segment"].clone().set_position(26, 56),
            sprites["skewer_segment"].clone().set_position(32, 56),
            sprites["skewer_segment"].clone().set_position(38, 56),
            sprites["skewer_segment"].clone().set_position(11, 36),
            sprites["platform_5x5"].clone().set_position(17, 12).set_scale(6),
            sprites["floor_strip"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(5, 42),
            sprites["skewer_magenta"].clone().set_position(17, 56),
            sprites["color_block"].clone().set_position(41, 24).color_remap(None, ORANGE),
            sprites["color_block"].clone().set_position(35, 24).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(47, 24).color_remap(None, RED),
            sprites["color_block"].clone().set_position(23, 56),
            sprites["color_block"].clone().set_position(29, 56).color_remap(None, ORANGE),
            sprites["color_block"].clone().set_position(41, 56).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(35, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(29, 24).color_remap(None, GREEN),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(7, 38),
            sprites["rail_segment"].clone().set_position(7, 32),
            sprites["rail_segment"].clone().set_position(7, 26),
            sprites["rail_segment"].clone().set_position(7, 20),
            sprites["rail_segment"].clone().set_position(7, 14),
            sprites["rail_segment"].clone().set_position(7, 8),
            sprites["platform_7x7"].clone().set_position(11, 6).set_scale(6),
            sprites["skewer_segment"].clone().set_position(11, 42),
            sprites["skewer_segment"].clone().set_position(17, 56),
            sprites["skewer_segment"].clone().set_position(23, 56),
            sprites["skewer_segment"].clone().set_position(29, 56),
            sprites["skewer_segment"].clone().set_position(35, 56),
            sprites["skewer_segment"].clone().set_position(5, 42),
            sprites["skewer_segment"].clone().set_position(41, 56),
            sprites["floor_strip"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(5, 42),
            sprites["skewer_magenta"].clone().set_position(17, 56),
            sprites["color_block"].clone().set_position(29, 6).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(29, 18).color_remap(None, RED),
            sprites["color_block"].clone().set_position(29, 12).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(23, 56),
            sprites["color_block"].clone().set_position(29, 56).color_remap(None, ORANGE),
            sprites["color_block"].clone().set_position(35, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(29, 24).color_remap(None, ORANGE),
            sprites["color_block"].clone().set_position(41, 56).color_remap(None, GREEN),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(7, 38),
            sprites["rail_segment"].clone().set_position(7, 32),
            sprites["rail_segment"].clone().set_position(7, 26),
            sprites["rail_segment"].clone().set_position(7, 20),
            sprites["rail_segment"].clone().set_position(7, 14),
            sprites["rail_segment"].clone().set_position(7, 8),
            sprites["platform_7x7"].clone().set_position(11, 6).set_scale(6),
            sprites["skewer_segment"].clone().set_position(11, 42),
            sprites["skewer_segment"].clone().set_position(17, 56),
            sprites["skewer_segment"].clone().set_position(23, 56),
            sprites["skewer_segment"].clone().set_position(29, 56),
            sprites["skewer_segment"].clone().set_position(35, 56),
            sprites["skewer_segment"].clone().set_position(29, 6).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(29, 12).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(29, 18).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(29, 0).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(5, 42),
            sprites["skewer_segment"].clone().set_position(29, 24).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(41, 56),
            sprites["floor_strip"].clone().set_position(0, 53),
            sprites["skewer_yellow"].clone().set_position(29, 0).set_rotation(90),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(5, 42),
            sprites["color_block"].clone().set_position(23, 42).color_remap(None, RED),
            sprites["color_block"].clone().set_position(17, 42).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(29, 42).color_remap(None, ORANGE),
            sprites["color_block"].clone().set_position(17, 56),
            sprites["color_block"].clone().set_position(23, 56).color_remap(None, ORANGE),
            sprites["color_block"].clone().set_position(41, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(47, 56).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(11, 42).color_remap(None, BLUE),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(7, 38),
            sprites["rail_segment"].clone().set_position(7, 32),
            sprites["rail_segment"].clone().set_position(7, 26),
            sprites["rail_segment"].clone().set_position(7, 20),
            sprites["rail_segment"].clone().set_position(7, 14),
            sprites["rail_segment"].clone().set_position(7, 8),
            sprites["platform_7x7"].clone().set_position(11, 6).set_scale(6),
            sprites["skewer_segment"].clone().set_position(11, 42),
            sprites["skewer_segment"].clone().set_position(11, 56),
            sprites["skewer_segment"].clone().set_position(17, 56),
            sprites["skewer_segment"].clone().set_position(23, 56),
            sprites["skewer_segment"].clone().set_position(41, 56),
            sprites["skewer_segment"].clone().set_position(23, 6).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(23, 12).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(23, 18).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(17, 42),
            sprites["skewer_segment"].clone().set_position(23, 42),
            sprites["skewer_segment"].clone().set_position(29, 42),
            sprites["skewer_segment"].clone().set_position(23, 0).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(5, 42),
            sprites["skewer_segment"].clone().set_position(47, 56),
            sprites["skewer_segment"].clone().set_position(35, 18).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(35, 12).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(35, 6).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(35, 0).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(35, 56),
            sprites["skewer_light_blue"].clone().set_position(23, 0).set_rotation(90),
            sprites["skewer_light_blue"].clone().set_position(35, 56),
            sprites["floor_strip"].clone().set_position(0, 53),
            sprites["skewer_yellow"].clone().set_position(11, 56),
            sprites["skewer_yellow"].clone().set_position(35, 0).set_rotation(90),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(5, 24),
            sprites["skewer_magenta"].clone().set_position(20, 56),
            sprites["color_block"].clone().set_position(17, 24).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(41, 24).color_remap(None, RED),
            sprites["color_block"].clone().set_position(26, 56),
            sprites["color_block"].clone().set_position(32, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(23, 24).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(35, 24).color_remap(None, RED),
            sprites["color_block"].clone().set_position(47, 24).color_remap(None, RED),
            sprites["color_block"].clone().set_position(11, 24).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(38, 56).color_remap(None, RED),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(7, 32),
            sprites["rail_segment"].clone().set_position(7, 26),
            sprites["rail_segment"].clone().set_position(7, 20),
            sprites["rail_segment"].clone().set_position(7, 14),
            sprites["rail_segment"].clone().set_position(7, 8),
            sprites["rail_segment"].clone().set_position(7, 38),
            sprites["platform_7x7"].clone().set_position(11, 6).set_scale(6),
            sprites["wall_block"].clone().set_position(29, 24),
            sprites["skewer_segment"].clone().set_position(17, 24),
            sprites["skewer_segment"].clone().set_position(26, 56),
            sprites["skewer_segment"].clone().set_position(32, 56),
            sprites["skewer_segment"].clone().set_position(23, 24),
            sprites["skewer_segment"].clone().set_position(5, 24),
            sprites["skewer_segment"].clone().set_position(20, 56),
            sprites["skewer_segment"].clone().set_position(11, 24),
            sprites["skewer_segment"].clone().set_position(38, 56),
            sprites["floor_strip"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(5, 26),
            sprites["skewer_magenta"].clone().set_position(5, 56),
            sprites["color_block"].clone().set_position(29, 44).color_remap(None, RED),
            sprites["color_block"].clone().set_position(29, 38).color_remap(None, RED),
            sprites["color_block"].clone().set_position(11, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(17, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(23, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(47, 26).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(29, 32).color_remap(None, RED),
            sprites["color_block"].clone().set_position(41, 26).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(35, 26).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(41, 56).color_remap(None, RED),
            sprites["color_block"].clone().set_position(47, 56).color_remap(None, RED),
            sprites["color_block"].clone().set_position(53, 56).color_remap(None, RED),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(7, 40),
            sprites["rail_segment"].clone().set_position(7, 22),
            sprites["rail_segment"].clone().set_position(7, 16),
            sprites["rail_segment"].clone().set_position(7, 10),
            sprites["rail_segment"].clone().set_position(13, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(19, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(25, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(43, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(7, 34),
            sprites["rail_segment"].clone().set_position(7, 28),
            sprites["rail_segment"].clone().set_position(31, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(37, 4).set_rotation(90),
            sprites["platform_7x7"].clone().set_position(11, 8).set_scale(6),
            sprites["wall_block"].clone().set_position(29, 26),
            sprites["skewer_segment"].clone().set_position(5, 26),
            sprites["skewer_segment"].clone().set_position(11, 26),
            sprites["skewer_segment"].clone().set_position(5, 56),
            sprites["skewer_segment"].clone().set_position(11, 56),
            sprites["skewer_segment"].clone().set_position(17, 56),
            sprites["skewer_segment"].clone().set_position(23, 56),
            sprites["skewer_segment"].clone().set_position(29, 8).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(29, 2).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(35, 56),
            sprites["skewer_segment"].clone().set_position(41, 56),
            sprites["skewer_segment"].clone().set_position(47, 56),
            sprites["skewer_segment"].clone().set_position(53, 56),
            sprites["skewer_purple"].clone().set_position(29, 2).set_rotation(90),
            sprites["skewer_purple"].clone().set_position(35, 56),
            sprites["floor_strip"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(5, 26),
            sprites["skewer_magenta"].clone().set_position(5, 56),
            sprites["color_block"].clone().set_position(23, 26).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(29, 20).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(11, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(17, 56).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(23, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(23, 20).color_remap(None, RED),
            sprites["color_block"].clone().set_position(29, 26).color_remap(None, RED),
            sprites["color_block"].clone().set_position(35, 32).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(41, 56).color_remap(None, RED),
            sprites["color_block"].clone().set_position(47, 56).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(53, 56).color_remap(None, RED),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(7, 40),
            sprites["rail_segment"].clone().set_position(7, 28),
            sprites["rail_segment"].clone().set_position(7, 22),
            sprites["rail_segment"].clone().set_position(7, 16),
            sprites["rail_segment"].clone().set_position(19, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(25, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(31, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(43, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(7, 34),
            sprites["rail_segment"].clone().set_position(7, 10),
            sprites["rail_segment"].clone().set_position(37, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(13, 4).set_rotation(90),
            sprites["platform_7x7"].clone().set_position(11, 8).set_scale(6),
            sprites["skewer_segment"].clone().set_position(5, 26),
            sprites["skewer_segment"].clone().set_position(11, 26),
            sprites["skewer_segment"].clone().set_position(5, 56),
            sprites["skewer_segment"].clone().set_position(11, 56),
            sprites["skewer_segment"].clone().set_position(17, 56),
            sprites["skewer_segment"].clone().set_position(23, 56),
            sprites["skewer_segment"].clone().set_position(29, 8).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(29, 2).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(35, 56),
            sprites["skewer_segment"].clone().set_position(41, 56),
            sprites["skewer_segment"].clone().set_position(47, 56),
            sprites["skewer_segment"].clone().set_position(53, 56),
            sprites["skewer_purple"].clone().set_position(29, 2).set_rotation(90),
            sprites["skewer_purple"].clone().set_position(35, 56),
            sprites["floor_strip"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),

    Level(
        sprites=[
            sprites["skewer_magenta"].clone().set_position(5, 26),
            sprites["skewer_magenta"].clone().set_position(11, 56),
            sprites["color_block"].clone().set_position(29, 14).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(29, 38).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(17, 56).color_remap(None, BLUE),
            sprites["color_block"].clone().set_position(23, 56).color_remap(None, GREEN),
            sprites["color_block"].clone().set_position(17, 26).color_remap(None, RED),
            sprites["color_block"].clone().set_position(41, 26).color_remap(None, ORANGE),
            sprites["color_block"].clone().set_position(41, 56).color_remap(None, RED),
            sprites["color_block"].clone().set_position(47, 56).color_remap(None, ORANGE),
            sprites["floor_tile"].clone().set_position(0, 54).set_scale(64),
            sprites["rail_segment"].clone().set_position(7, 22),
            sprites["rail_segment"].clone().set_position(25, 4).set_rotation(90),
            sprites["rail_segment"].clone().set_position(7, 28),
            sprites["rail_segment"].clone().set_position(31, 4).set_rotation(90),
            sprites["platform_7x7"].clone().set_position(11, 8).set_scale(6),
            sprites["skewer_segment"].clone().set_position(5, 26),
            sprites["skewer_segment"].clone().set_position(11, 26),
            sprites["skewer_segment"].clone().set_position(11, 56),
            sprites["skewer_segment"].clone().set_position(17, 56),
            sprites["skewer_segment"].clone().set_position(23, 56),
            sprites["skewer_segment"].clone().set_position(29, 8).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(29, 2).set_rotation(90),
            sprites["skewer_segment"].clone().set_position(35, 56),
            sprites["skewer_segment"].clone().set_position(41, 56),
            sprites["skewer_segment"].clone().set_position(47, 56),
            sprites["skewer_purple"].clone().set_position(29, 2).set_rotation(90),
            sprites["skewer_purple"].clone().set_position(35, 56),
            sprites["floor_strip"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
        data={
            "grouped_pauses": False,
            "lit_extension": True,
        },
    ),
]


BACKGROUND_COLOR = BLACK
PADDING_COLOR    = GRAY


COLOR_PALETTE: dict[int, dict] = {
    -1: {"name": "TRANSPARENT", "hex": None,       "label": "Transparent", "label_l2": ''      },
     0: {"name": "WHITE",       "hex": "#FFFFFF",   "label": "White",       "label_l2": ''   },
     1: {"name": "OFF_WHITE",   "hex": "#CCCCCC",   "label": "Off White",   "label_l2": ''                  },
     2: {"name": "LIGHT_GRAY",  "hex": "#999999",   "label": "Light Gray",  "label_l2": ''                  },
     3: {"name": "GRAY",        "hex": "#666666",   "label": "Gray",        "label_l2": ''         },
     4: {"name": "DARK_GRAY",   "hex": "#333333",   "label": "Dark Gray",   "label_l2": ''                  },
     5: {"name": "BLACK",       "hex": "#000000",   "label": "Black",       "label_l2": ''   },
     6: {"name": "MAGENTA",     "hex": "#E53AA3",   "label": "Magenta",     "label_l2": ''            },
     7: {"name": "PINK",        "hex": "#FF7BCC",   "label": "Pink",        "label_l2": ''         },
     8: {"name": "RED",         "hex": "#F93C31",   "label": "Red",         "label_l2": ''   },
     9: {"name": "BLUE",        "hex": "#1E93FF",   "label": "Blue",        "label_l2": ''   },
    10: {"name": "LIGHT_BLUE",  "hex": "#88D8F1",   "label": "Light Blue",  "label_l2": ''                  },
    11: {"name": "YELLOW",      "hex": "#FFDC00",   "label": "Yellow",      "label_l2": ''   },
    12: {"name": "ORANGE",      "hex": "#FF851B",   "label": "Orange",      "label_l2": ''            },
    13: {"name": "MAROON",      "hex": "#921231",   "label": "Maroon",      "label_l2": ''            },
    14: {"name": "GREEN",       "hex": "#4FCC30",   "label": "Green",       "label_l2": ''   },
    15: {"name": "PURPLE",      "hex": "#A356D6",   "label": "Purple",      "label_l2": ''   },
}




TAG_DISPLAY_NAMES: dict[str, str] = {
    "skewer":          "Skewer",
    "skewer_segment":  "Skewer Segment",
    "color_block":     "Color Block",
    "rail":            "Rail",
    "platform":        "Platform",
    "wall_block":      "Wall Block",
    "sys_click":       "Clickable",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "skewer":          "",
    "skewer_segment":  "",
    "color_block":     "",
    "rail":            "",
    "platform":        "",
    "wall_block":      "",
    "sys_click":       "",
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
    grid = getattr(level, "grid_size", (64, 64)) or (64, 64)
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

    _analyses: list[dict] = [
        {
            "gimmicks":     ["Two skewers, three color blocks (GREEN/BLUE/RED)"],
            "gimmicks_l2":  [''                                                             ],
            "subgoals":     ["Push skewer forward so segments align with matching-color blocks"],
            "subgoals_l2":  [''                                                                              ],
            "goal_flow":    "Move the skewer forward/backward to align all colored blocks between the paired skewers.",
            "goal_flow_l2": ''                                                                                             ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Two skewers, five color blocks (ORANGE/BLUE/RED/GREEN)"],
            "gimmicks_l2":  [''                                                                 ],
            "subgoals":     ["Coordinate both skewers to match all color positions"],
            "subgoals_l2":  [''                                                         ],
            "goal_flow":    "Use keyboard or click to switch between skewers and align all color block pairs.",
            "goal_flow_l2": ''                                                                                                                  ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Rotated skewer segments, yellow skewer introduced"],
            "gimmicks_l2":  [''                                                      ],
            "subgoals":     ["Manage both vertical and horizontal skewer orientations"],
            "subgoals_l2":  [''                                             ],
            "goal_flow":    "Push horizontal and vertical skewers, aligning blocks along both axes.",
            "goal_flow_l2": ''                                                                              ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Multiple skewers, light-blue and yellow skewers, wall blocks"],
            "gimmicks_l2":  [''                                                                      ],
            "subgoals":     ["Navigate wall obstacles while aligning blocks for all skewer pairs"],
            "subgoals_l2":  [''                                                                     ],
            "goal_flow":    "Carefully push each skewer around wall_block obstacles to connect all color pairs.",
            "goal_flow_l2": ''                                                                                                         ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Mixed horizontal/vertical skewers, wall_block in center"],
            "gimmicks_l2":  [''                                                          ],
            "subgoals":     ["Route skewers around the wall block in the center"],
            "subgoals_l2":  [''                                                         ],
            "goal_flow":    "Push skewers around the central wall obstacle to align all color block pairs.",
            "goal_flow_l2": ''                                                                                                      ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Large level with purple skewer pair and wall_block"],
            "gimmicks_l2":  [''                                                         ],
            "subgoals":     ["Coordinate all skewer pairs including the diagonal purple skewer"],
            "subgoals_l2":  [''                                                                     ],
            "goal_flow":    "Manage multiple skewer pairs across the large grid, using undo when needed.",
            "goal_flow_l2": ''                                                                                                         ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Large grid, color variation from Level 6"],
            "gimmicks_l2":  [''                                                    ],
            "subgoals":     ["Solve altered color arrangement using same multi-skewer strategy"],
            "subgoals_l2":  [''                                                                  ],
            "goal_flow":    "Apply multi-skewer coordination strategy with different color targets.",
            "goal_flow_l2": ''                                                                                    ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Compact level, vertical and horizontal skewer mix"],
            "gimmicks_l2":  [''                                                   ],
            "subgoals":     ["Precisely align skewers in tight space"],
            "subgoals_l2":  [''                                                      ],
            "goal_flow":    "Use both keyboard navigation and click-to-select to efficiently solve this compact layout.",
            "goal_flow_l2": ''                                                                                                               ,
            "difficulty":   "very_hard",
        },
    ]

    analysis = _analyses[level_index] if level_index < len(_analyses) else {}
    return {
        "level_index": level_index + 1,
        "grid_size":   list(grid),
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
        "game_id":          "sk48",
        "title":            "Skewer 48",
        "title_l2":         ''              ,
        "description":      "A puzzle game where you push and retract skewers to align colored blocks between paired skewers. All pairs must match colors to win.",
        "description_l2":   ''                                                                                                                                                      ,
        "default_grid_size": [64, 64],
        "num_levels":        len(levels),

        "goal":             "Align all paired skewers so that the color block at each skewer segment position matches between the top and bottom skewer of each pair.",
        "goal_l2":          ''                                                                                                                                 ,
        "win_condition":    "All skewer pairs have matching color blocks at every segment position (gvtmoopqgy returns True).",
        "win_condition_l2": ''                                                                                                                                                            ,
        "lose_condition":   "Move budget (steps remaining) reaches 0.",
        "lose_condition_l2":''                                     ,

        "actions": {
            "ACTION1": {
                "key":             "up",
                "effect":          "Move active skewer upward (or in the reverse direction).",
                "effect_l2":       ''                                                               ,
                "restrictions":    "Blocked by walls, platforms, and grid boundaries.",
                "restrictions_l2": ''                                                                              ,
                "step_cost":       1,
            },
            "ACTION2": {
                "key":             "down",
                "effect":          "Move active skewer downward (or in the forward direction), extending a new segment.",
                "effect_l2":       ''                                                                                                         ,
                "restrictions":    "Blocked by walls, platforms, and grid boundaries.",
                "restrictions_l2": ''                                                                              ,
                "step_cost":       1,
            },
            "ACTION3": {
                "key":             "left",
                "effect":          "Move active skewer leftward.",
                "effect_l2":       ''                                                ,
                "restrictions":    "Blocked by walls, platforms, and grid boundaries.",
                "restrictions_l2": ''                                                                              ,
                "step_cost":       1,
            },
            "ACTION4": {
                "key":             "right",
                "effect":          "Move active skewer rightward, extending a new segment.",
                "effect_l2":       ''                                                                                          ,
                "restrictions":    "",
                "restrictions_l2": ''                                                                              ,
                "step_cost":       1,
            },
            "ACTION6": {
                "key":             "click",
                "effect":          "Click on another skewer to select it as the active skewer.",
                "effect_l2":       ''                                                                                       ,
                "restrictions":    "Only available skewers (not wall-based) can be selected.",
                "restrictions_l2": ''                                                               ,
                "step_cost":       0,
            },
            "ACTION7": {
                "key":             "undo",
                "effect":          "Undo the last move, restoring the previous board state.",
                "effect_l2":       ''                                                                     ,
                "restrictions":    "Cannot undo if no history is available.",
                "restrictions_l2": ''                                             ,
                "step_cost":       0,
            },
        },

        "sprite_roles": {
            "skewer":          "Main skewer sprite. Pushed by keyboard to extend/retract through color blocks.",
            "skewer_l2":       ''                                                                                                                              ,
            "skewer_segment":  "Segment of a skewer chain. Rendered with stripe pattern; bright when active.",
            "skewer_segment_l2": ''                                                                                                         ,
            "color_block":     "Colored target block. Skewer segments passing this position are checked for color match.",
            "color_block_l2":  ''                                                                                                                           ,
            "rail":            "Rail sprite that guides skewer movement along its axis.",
            "rail_l2":         ''                                                                              ,
            "platform":        "Boundary block. Skewer cannot pass through it.",
            "platform_l2":     ''                                                            ,
            "wall_block":      "Obstacle block. Skewer movement is blocked by it.",
            "wall_block_l2":   ''                                                            ,
            "match_indicator": "White square shown at a segment position when paired skewers have matching colors.",
            "match_indicator_l2": ''                                                                                             ,
            "drop_preview":    "Red-framed arrow preview shown where a new segment will drop during extension.",
            "drop_preview_l2": ''                                                                                                   ,
            "floor_tile":      "1px floor tile scaled to fill the bottom row.",
            "floor_tile_l2":   ''                                                                  ,
            "floor_strip":     "64px horizontal gray strip at the bottom of the play area.",
            "floor_strip_l2":  ''                                                                ,
        },

        "mechanics": {
            "skewer_extension":     "Moving a skewer in its forward direction adds a new segment sprite at the head position. Moving backward pops the tail segment.",
            "skewer_extension_l2":  ''                                                                                                                                                                     ,
            "color_matching":       "After each move, each segment position is checked for a color_block. Paired skewers must have identical colors at each matching segment index.",
            "color_matching_l2":    ''                                                                                                                                                                        ,
            "win_detection":        "Win is triggered when ALL pairs report full color match across all segment positions.",
            "win_detection_l2":     ''                                                                                                ,
            "undo_system":          "Each move saves a snapshot to seghobzez (history). ACTION7 pops the latest snapshot and restores the board.",
            "undo_system_l2":       ''                                                                                                                                                                   ,
            "step_budget":          "vhzjwcpmk (total) and qiercdohl (remaining). Each valid move decrements qiercdohl. Game over when 0.",
            "step_budget_l2":       ''                                                                                                                         ,
            "drop_preview_system":  "When a segment would fall onto a new rail position, a drop_preview sprite is shown briefly before the segment is placed.",
            "drop_preview_system_l2": ''                                                                                                                                       ,
        },
    },


    "levels": [
        _build_level_knowledge(i, lv)
        for i, lv in enumerate(levels)
    ],
}




class ProgressBarDisplay(RenderableUserDisplay):
    """



    """

    def __init__(self, game: "Sk48"):
        """"""
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """










        """
        width = frame.shape[0]

        ratio = self.game.qiercdohl / self.game.vhzjwcpmk
        fill_width = math.ceil(width * ratio)

        frame[PROGRESS_BAR_ROW, :] = PROGRESS_BAR_BG_COLOR
        frame[PROGRESS_BAR_ROW, 0:fill_width] = PROGRESS_BAR_FILL_COLOR
        return frame




class Sk48(ARCBaseGame):
    """




















    """

    def __init__(self) -> None:
        """"""




        self._log_state = ActionLogState()
        self.action_log:               list[str]        = []
        self.action_log_l2:            list[str]        = []
        self._frame_sprite_log:        list[list[dict]] = []
        self._frame_nonsprite_log:     list[list[dict]] = []

        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[ProgressBarDisplay(self)],
        )
        super().__init__(
            game_id="sk48",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 6, 7],
        )

    def on_set_level(self, level: Level) -> None:
        """





        """

        self._levels[self._current_level_index] = self._clean_levels[self._current_level_index].clone()

        self.vhzjwcpmk = 196
        self.qiercdohl = self.vhzjwcpmk


        self.current_level.remove_sprite(self.current_level.get_sprites_by_name("floor_strip")[0])


        all_skewers = self.current_level.get_sprites_by_tag("skewer")


        self.mwfajkguqx: dict[Sprite, list[Sprite]] = {}
        self.vjfbwggsd: dict[Sprite, list[Sprite]] = {}
        self.vbelzuaian = self.current_level.get_sprites_by_tag("color_block")

        self.xpmcmtbcv: dict[Sprite, Sprite] = {}

        for skewer in all_skewers:

            if not skewer.name.startswith("wallBase"):
                skewer.tags.append("sys_click")

            if skewer.y < PROGRESS_BAR_ROW:

                paired = next(
                    (b for b in all_skewers if b != skewer and b.pixels[2, 2] == skewer.pixels[2, 2]),
                    None,
                )
                if paired:
                    self.xpmcmtbcv[skewer] = paired
            self.mwfajkguqx[skewer] = []
            self.vjfbwggsd[skewer] = []


            dir_x, dir_y = ROTATION_TO_DIRECTION[skewer.rotation]
            is_horizontal = skewer.rotation in (0, 180)
            seg = self._find_segment_at(skewer.x, skewer.y, is_horizontal)
            while seg is not None:

                seg.set_layer(1 if seg.rotation in (0, 180) else 0)
                self.mwfajkguqx[skewer].append(seg)

                step_x, step_y = (dir_x * STEP_SIZE, dir_y * STEP_SIZE)
                seg = self._find_segment_at(seg.x + step_x, seg.y + step_y, is_horizontal)


        self.vzvypfsnt = all_skewers[0]
        self._select_skewer(self.vzvypfsnt, initial=True)


        self.jdojcthkf: dict[Sprite, list[Sprite]] = {}
        for paired_skewer in self.xpmcmtbcv.values():
            self.jdojcthkf[paired_skewer] = []
            for seg in self.mwfajkguqx[paired_skewer][1:]:

                indicator = sprites["match_indicator"].clone().set_position(seg.x + 1, seg.y + 1)
                self.current_level.add_sprite(indicator.set_visible(False))
                self.jdojcthkf[paired_skewer].append(indicator)


        self._check_win_and_update_indicators()





        _name_counts: dict[str, int] = {}
        for spr in self.current_level.get_sprites():
            _name_counts[spr.name] = _name_counts.get(spr.name, 0) + 1
        _to_rename = {n for n, c in _name_counts.items() if c >= 2}
        if _to_rename:
            _idx_per_name: dict[str, int] = {n: 0 for n in _to_rename}
            for spr in self.current_level.get_sprites():
                if spr.name in _to_rename:
                    _idx_per_name[spr.name] += 1
                    spr.set_name(f"{spr.name}_{_idx_per_name[spr.name]}")


        self.lqwkgffeb = self.current_level.get_sprites_by_tag("platform")[0]


        self.ljprkjlji: list[tuple[Sprite, tuple[int, int]]] = []
        self.hadfnehqh: Sprite | None = None
        self.lgdrixfno = -1
        self.yzyidartf: list[tuple[Sprite, int]] = []
        self.xsoixcero = False
        self.seghobzez: list[list[tuple[Sprite, int, int, int]]] = []
        self._save_undo_state()
        self.rztawzist: list[Sprite] = []
        self.pzzwlsmdt: list[Sprite] = []



        self._extend_anim_active = False

        self._extend_emit_info: tuple[Sprite, str] | None = None
        self._extend_reemit: tuple[Sprite, str] | None = None

        self._emit_initial_layout()


    def _emit_initial_layout(self) -> None:
        """














        """
        _layout_rows: list[dict] = []

        for _tag, _role, _role_l2 in (
            ("skewer",         "skewer",         ""),
            ("skewer_segment", "skewer_segment", ""),
            ("color_block",    "color_block",    ""),
            ("rail",           "rail",           ""),
            ("platform",       "platform",       ""),
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


        try:
            _ns_status = self.get_nonsprite_status() or []
        except Exception:
            _ns_status = []
        for _ns in _ns_status:
            if _ns.get("name") != "steps_remaining":
                continue
            _layout_rows.append({
                "role":    "remaining_step",
                "role_l2": ''                     ,
                "sname":   "steps_remaining",
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


        self._step_log    = []
        self._step_log_l2 = []



        _before_sprites: dict[int, dict] = {}
        for spr in self.current_level.get_sprites():
            _before_sprites[id(spr)] = {
                "name":     spr.name,
                "x":        spr.x,
                "y":        spr.y,
                "width":    spr.width,
                "height":   spr.height,
                "rotation": spr.rotation,
                "visible":  spr.is_visible,
            }

        _before_steps = self.qiercdohl


        if self.action.id._value_ not in self._available_actions:
            self.complete_action()
            self._post_step(_before_sprites, _before_steps)
            return


        if self.lgdrixfno >= 0:
            blink_on = self.lgdrixfno // 5 % 2 == 0 or self.lgdrixfno >= 25
            for target_spr, original_color in self.yzyidartf:
                target_spr.color_remap(None, WIN_FLASH_COLOR if blink_on else original_color)



            if self.lgdrixfno == 0:
                _is_transition = True
            else:
                _prev_blink = (self.lgdrixfno - 1) // 5 % 2 == 0 or (self.lgdrixfno - 1) >= 25
                _is_transition = blink_on != _prev_blink
            if _is_transition:
                for _tspr, _ in self.yzyidartf:
                    _tdx, _tdy, _tdw, _tdh = self._to_display_bbox(_tspr)
                    if blink_on:
                        self._log(f"[bright]:[{_tspr.name}]:[sname({_tspr.name});position({_tdx}_{_tdy});bbox({_tdw}_{_tdh});state(flash)]:{_tspr.name} at ({_tdx},{_tdy}) turns white.|Object at ({_tdx},{_tdy}) turns white.",
                                             f"[bright]:[{_tspr.name}]:[sname({_tspr.name});position({_tdx}_{_tdy});bbox({_tdw}_{_tdh});state(flash)]:({_tdx},{_tdy}){_tspr.name}|({_tdx},{_tdy})Object")
                    else:
                        self._log(f"[bright]:[{_tspr.name}]:[sname({_tspr.name});position({_tdx}_{_tdy});bbox({_tdw}_{_tdh});state(off)]:{_tspr.name} at ({_tdx},{_tdy}) returns to original color.|Object at ({_tdx},{_tdy}) returns to original color.",
                                             f"[bright]:[{_tspr.name}]:[sname({_tspr.name});position({_tdx}_{_tdy});bbox({_tdw}_{_tdh});state(off)]:({_tdx},{_tdy}){_tspr.name}|({_tdx},{_tdy})Object")
            self.lgdrixfno += 1
            if self.lgdrixfno >= 35:


                self.complete_action()
                self.next_level()
            self._post_step(_before_sprites, _before_steps)
            return


        if self.ljprkjlji or self.pzzwlsmdt:
            if self.ljprkjlji:

                self._animate_sprites()
                if self.hadfnehqh:
                    self.current_level.remove_sprite(self.hadfnehqh)
                    self.hadfnehqh = None
                self.ljprkjlji.clear()

                if self._extend_anim_active and self._extend_reemit:
                    self._emit_extent_log_current(*self._extend_reemit)
                    self._extend_reemit = None
                if self.pzzwlsmdt:
                    self._post_step(_before_sprites, _before_steps)
                    return
            else:

                for _block in self.pzzwlsmdt:
                    _bx, _by, _bw, _bh = self._to_display_bbox(_block)
                    self._log(
                        f"[change]:[{_block.name}]:[position({_bx}_{_by});bbox({_bw}_{_bh});shape(extended)_(normal)]:{_block.name} at ({_bx},{_by}) returns to its normal shape.|Object at ({_bx},{_by}) returns to its normal shape.",
                        f"[change]:[{_block.name}]:[position({_bx}_{_by});bbox({_bw}_{_bh});shape(extended)_(normal)]:({_bx},{_by}){_block.name}|({_bx},{_by})Object"
                    )

                for preview_spr in self.rztawzist:
                    preview_spr.set_visible(False)
                self.pzzwlsmdt.clear()

            win = self._check_win_and_update_indicators()
            if win:
                self.lgdrixfno = 0

                for color_blocks in self.vjfbwggsd.values():
                    for block in color_blocks:
                        self.yzyidartf.append((block, block.pixels[1, 1]))
                for indicators in self.jdojcthkf.values():
                    for indicator in indicators:
                        indicator.set_visible(False)
            else:
                self._save_undo_state()
                self.complete_action()
                if self.qiercdohl == 0:
                    self.lose()
            self._post_step(_before_sprites, _before_steps)
            return


        if self.action.id == GameAction.ACTION6:
            click_x, click_y = (
                self.action.data.get("x", -1),
                self.action.data.get("y", -1),
            )
            if clicked := self.current_level.get_sprite_at(click_x, click_y, "sys_click"):
                for top_skewer, bottom_skewer in self.xpmcmtbcv.items():
                    if clicked in [top_skewer, bottom_skewer] and top_skewer != self.vzvypfsnt:
                        self._select_skewer(top_skewer)
                        self.xsoixcero = True
                        break

        elif self.action.id == GameAction.ACTION7:
            self._undo()

        else:
            self.xsoixcero = False
            self._apply_move()
            if self.ljprkjlji:
                self._post_step(_before_sprites, _before_steps)
                return


        if self.qiercdohl == 0:
            self.lose()
        self.complete_action()


        self._post_step(_before_sprites, _before_steps)

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


    def _log(self, en: str, l2: str) -> None:
        """
        """
        self._step_log.append(en)
        self._step_log_l2.append(l2)

    def _set_event(self, key: str, value) -> None:
        """
        """
        self._log_state.set_event(key, value)

    def _finalize_action_log(self) -> None:
        """






        """
        import re
        _RE_POS = re.compile(
            r'(\[.+?\]):\[(.+?)\]:\[([^\]]*?)position\((-?\d+)_(-?\d+)\)_\((-?\d+)_(-?\d+)\)([^\]]*)\]:(.*)')

        seen: set[str] = set()

        extent_last: dict[tuple[str, str], tuple[str, str]] = {}

        move_chains: dict[str, list] = {}
        move_order: list[str] = []
        result_en: list[str] = []
        result_l2: list[str] = []

        all_en = [e for fr in self._frame_action_logs for e in fr]
        all_l2 = [e for fr in self._frame_action_logs_l2 for e in fr]




        _win_flash_names: set[str] = {spr.name for spr, _ in (self.yzyidartf or [])}
        _win_flash_seen: set[str] = set()
        _win_flash_entries: list[tuple[str, str]] = []

        for en, l2 in zip(all_en, all_l2):






            if en.startswith("[bright]:[game]:"):
                continue
            if (
                en.startswith("[bright]:")
                and "position(" in en
                and "state(flash)" in en
            ):
                _tname_match = re.match(r'^\[bright\]:\[([^\]]+)\]:', en)
                if _tname_match:
                    _tname = _tname_match.group(1)
                    if _tname in _win_flash_names and _tname not in _win_flash_seen:
                        _win_flash_seen.add(_tname)


                        _params_re = re.compile(
                            r'^(\[bright\]:\[[^\]]+\]:\[[^\]]+\]):.*$'
                        )
                        _en_summary = _params_re.sub(
                            rf"\1:{_tname} blinks white during the win flash animation.|The Object blinks white.",
                            en, count=1,
                        )
                        _l2_summary = _params_re.sub(
                            rf"\1:{_tname}|Object",
                            l2, count=1,
                        )
                        _win_flash_entries.append((_en_summary, _l2_summary))
                    continue

            if (
                en.startswith("[bright]:")
                and "state(off)" in en
            ):
                _tname_match = re.match(r'^\[bright\]:\[([^\]]+)\]:', en)
                if _tname_match and _tname_match.group(1) in _win_flash_names:
                    continue


            _es = re.match(r'(\[(?:extend|shrink)\]):\[(.+?)\]:', en)
            if _es:
                extent_last[(_es.group(1), _es.group(2))] = (en, l2)
                continue


            m = _RE_POS.fullmatch(en)
            if m and m.group(1) == "[move]":
                tgt = m.group(2)
                x1, y1 = int(m.group(4)), int(m.group(5))
                x2, y2 = int(m.group(6)), int(m.group(7))
                if tgt in move_chains:
                    mc = move_chains[tgt]
                    mc[2] = x2
                    mc[3] = y2
                    mc[4] = en
                    mc[5] = l2
                else:
                    move_chains[tgt] = [x1, y1, x2, y2, en, l2]
                    move_order.append(tgt)
                continue


            if en not in seen:
                seen.add(en)
                result_en.append(en)
                result_l2.append(l2)


        _RE_POS_SUB = re.compile(r'position\(-?\d+_-?\d+\)_\(-?\d+_-?\d+\)')
        for tgt in move_order:
            mc = move_chains[tgt]
            x1, y1, x2, y2, tmpl_en, tmpl_l2 = mc

            en_out = _RE_POS_SUB.sub(f"position({x1}_{y1})_({x2}_{y2})", tmpl_en, count=1)
            l2_out = _RE_POS_SUB.sub(f"position({x1}_{y1})_({x2}_{y2})", tmpl_l2, count=1)

            en_out = re.sub(r'from \(\d+,\d+\) to \(\d+,\d+\)', f'from ({x1},{y1}) to ({x2},{y2})', en_out)
            l2_out = re.sub(r'\(\d+,\d+\)\(\d+,\d+\)', f'({x1},{y1})({x2},{y2})', l2_out)
            l2_out = re.sub(r'\(\d+,\d+\)\(\d+,\d+\)', f'({x1},{y1})({x2},{y2})', l2_out)
            result_en.append(en_out)
            result_l2.append(l2_out)


        for (en, l2) in extent_last.values():
            result_en.append(en)
            result_l2.append(l2)


        for (en, l2) in _win_flash_entries:
            result_en.append(en)
            result_l2.append(l2)


        from gamesofts._action_log import compose_default_summary
        result_en, result_l2 = compose_default_summary(
            result_en, result_l2, self._log_state._events
        )

        self.action_log = result_en
        self.action_log_l2 = result_l2

    def _bar_pixel_len(self, steps: int) -> int:
        """


        """
        total = self.vhzjwcpmk
        if total == 0:
            return 0
        s = steps
        if s < 0:
            s = 0
        if s > total:
            s = total

        return math.ceil(64 * s / total)

    def _post_step(self, before_sprites: dict, before_steps: int) -> None:
        """








        """
        after_sprites: dict[int, Sprite] = {id(spr): spr for spr in self.current_level.get_sprites()}


        if self._extend_emit_info:
            self._emit_extent_log_current(*self._extend_emit_info)
            self._extend_reemit = self._extend_emit_info
            self._extend_emit_info = None



        def _is_segment_name(n: str) -> bool:
            return n == "skewer_segment" or n.startswith("skewer_segment_")
        _suppress_seg = bool(getattr(self, "_extend_anim_active", False))

        def _is_drop_preview(n: str) -> bool:
            return n == "drop_preview" or n.startswith("drop_preview_")

        def _is_match_indicator(n: str) -> bool:
            return n == "match_indicator" or n.startswith("match_indicator_")


        for spr_id, spr in after_sprites.items():
            b = before_sprites.get(spr_id)
            if (b is None and spr.is_visible) or (b is not None and not b["visible"] and spr.is_visible):
                if _suppress_seg and _is_segment_name(spr.name):
                    continue
                if _is_drop_preview(spr.name):
                    continue
                dx, dy, dw, dh = self._to_display_bbox(spr)
                if _is_match_indicator(spr.name):
                    self._log(f"[bright]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh});state(on)]:{spr.name} lights up at ({dx},{dy}).|Object lights up at ({dx},{dy}).",
                             f"[bright]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh});state(on)]:({dx},{dy}){spr.name}|({dx},{dy})Object")
                    continue
                self._log(f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).",
                         f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name}({dx},{dy})|Object({dx},{dy})")


        _arc_scale_d = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in before_sprites.items():
            spr = after_sprites.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                if _suppress_seg and _is_segment_name(b["name"]):
                    continue
                if _is_drop_preview(b["name"]):
                    continue
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = int(b.get("width", 0))  * _arc_scale_d
                bdh = int(b.get("height", 0)) * _arc_scale_d
                if _is_match_indicator(b["name"]):
                    self._log(f"[bright]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh});state(off)]:{b['name']} dims at ({bdx},{bdy}).|Object dims at ({bdx},{bdy}).",
                             f"[bright]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh});state(off)]:({bdx},{bdy}){b['name']}|({bdx},{bdy})Object")
                    continue
                self._log(f"[disappear]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).",
                         f"[disappear]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']}({bdx},{bdy})|Object({bdx},{bdy})")


        _moved: list[tuple[int, dict, Sprite]] = []
        _rotated: list[tuple[int, dict, Sprite]] = []
        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            b = before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                continue
            if (spr.x, spr.y) != (b["x"], b["y"]):
                _moved.append((spr_id, b, spr))
            if spr.rotation != b["rotation"]:
                _rotated.append((spr_id, b, spr))


        _handled_ids: set[int] = set()
        _moved_id_set = {m[0] for m in _moved}

        for skewer, segs in self.mwfajkguqx.items():
            skewer_id = id(skewer)
            if skewer_id not in _moved_id_set:
                continue
            skewer_b = before_sprites.get(skewer_id)
            if skewer_b is None:
                continue
            delta_gx = skewer.x - skewer_b["x"]
            delta_gy = skewer.y - skewer_b["y"]

            group_ids = {skewer_id}
            all_same = True
            for seg in segs:
                seg_id = id(seg)
                seg_b = before_sprites.get(seg_id)
                if seg_b is None or seg_id not in _moved_id_set:
                    all_same = False
                    break
                if (seg.x - seg_b["x"], seg.y - seg_b["y"]) != (delta_gx, delta_gy):
                    all_same = False
                    break
                group_ids.add(seg_id)

            if all_same and len(group_ids) > 1:
                _handled_ids.update(group_ids)
                all_group = [skewer] + list(segs)
                a_min_x = min(s.x for s in all_group)
                a_min_y = min(s.y for s in all_group)
                a_max_x = max(s.x + s.width for s in all_group)
                a_max_y = max(s.y + s.height for s in all_group)
                b_min_x = a_min_x - delta_gx
                b_min_y = a_min_y - delta_gy
                gw = (a_max_x - a_min_x) * _arc_scale_d
                gh = (a_max_y - a_min_y) * _arc_scale_d
                gdx_b, gdy_b = self._to_display_xy(b_min_x, b_min_y)
                gdx_a, gdy_a = self._to_display_xy(a_min_x, a_min_y)
                sn = skewer.name
                self._log(
                    f"[move]:[{sn}]:[position({gdx_b}_{gdy_b})_({gdx_a}_{gdy_a});bbox({gw}_{gh})]:{sn} group moves from ({gdx_b},{gdy_b}) to ({gdx_a},{gdy_a}).|Object group moves from ({gdx_b},{gdy_b}) to ({gdx_a},{gdy_a}).",
                    f"[move]:[{sn}]:[position({gdx_b}_{gdy_b})_({gdx_a}_{gdy_a});bbox({gw}_{gh})]:{sn}({gdx_b},{gdy_b})({gdx_a},{gdy_a})|Object({gdx_b},{gdy_b})({gdx_a},{gdy_a})"
                )


        for spr_id, b, spr in _moved:
            if spr_id in _handled_ids:
                continue
            if _suppress_seg and _is_segment_name(spr.name):
                continue
            if _is_drop_preview(spr.name):
                continue
            bdx, bdy = self._to_display_xy(b["x"], b["y"])
            adx, ady, adw, adh = self._to_display_bbox(spr)
            self._log(f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady}).",
                     f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})")


        for spr_id, b, spr in _rotated:
            _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
            self._log(f"[rotate]:[{spr.name}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:{spr.name} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.",
                     f"[rotate]:[{spr.name}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:({_rdx},{_rdy}){spr.name}{b['rotation']}°{spr.rotation}°|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°")


        if before_steps != self.qiercdohl and self._bar_pixel_len(before_steps) != self._bar_pixel_len(self.qiercdohl):
            if self.qiercdohl < before_steps:
                self._log(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({self.qiercdohl})]:The light-gray step bar near the bottom (y={PROGRESS_BAR_ROW}) shrinks from the right ({before_steps}→{self.qiercdohl} steps remain).|The light-gray Object near the bottom (y={PROGRESS_BAR_ROW}) shrinks from the right ({before_steps}→{self.qiercdohl}).",
                         f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({self.qiercdohl})]:(y={PROGRESS_BAR_ROW}){before_steps}→{self.qiercdohl}|(y={PROGRESS_BAR_ROW})Object{before_steps}→{self.qiercdohl}")
            else:
                self._log(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({self.qiercdohl})]:The light-gray step bar near the bottom (y={PROGRESS_BAR_ROW}) extends to the right ({before_steps}→{self.qiercdohl} steps).|The light-gray Object near the bottom (y={PROGRESS_BAR_ROW}) extends to the right ({before_steps}→{self.qiercdohl}).",
                         f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({self.qiercdohl})]:(y={PROGRESS_BAR_ROW}){before_steps}→{self.qiercdohl}|(y={PROGRESS_BAR_ROW})Object{before_steps}→{self.qiercdohl}")


        if getattr(self, "lgdrixfno", -1) == 0:
            self._set_event("win", {
                "en": "[win]:[game]:[]:The level is cleared.|The level is cleared.",
                "l2": "[win]:[game]:[]:|",
            })





        if self.qiercdohl == 0 and getattr(self, "lgdrixfno", -1) < 0:
            self._set_event("gameover", {
                "en": "[gameover]:[game]:[]:Step counter reached zero, game over.|The Object reached zero, game over.",
                "l2": "[gameover]:[game]:[]:0|Object0",
            })


        if self._extend_anim_active and not self.ljprkjlji and not self.pzzwlsmdt:
            self._extend_anim_active = False
            self._extend_reemit = None


        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()







        if self._action_complete:
            self._finalize_action_log()

    def _emit_extent_log_current(self, active: Sprite, verb: str) -> None:
        """



        """
        segs = self.mwfajkguqx[active]
        rot  = active.rotation
        s    = STEP_SIZE
        last = segs[-1]


        if rot == 0:
            from_x, from_y = active.x, active.y
            tip_x,  tip_y  = last.x + s, active.y
            bbox_w, bbox_h = -(tip_x - from_x), s
        elif rot == 90:
            from_x, from_y = active.x, active.y
            tip_x,  tip_y  = active.x, last.y + s
            bbox_w, bbox_h = s, -(tip_y - from_y)
        elif rot == 180:
            from_x, from_y = active.x + s, active.y
            tip_x,  tip_y  = last.x, active.y
            bbox_w, bbox_h = from_x - tip_x, s
        else:
            from_x, from_y = active.x, active.y + s
            tip_x,  tip_y  = active.x, last.y
            bbox_w, bbox_h = s, from_y - tip_y

        fdx, fdy = self._to_display_xy(from_x, from_y)
        tdx, tdy = self._to_display_xy(tip_x,  tip_y)
        cam = self.camera
        arc_scale = min(64 // cam.width, 64 // cam.height)
        bdw = bbox_w * arc_scale
        bdh = bbox_h * arc_scale

        sname = active.name
        if verb == "extend":
            en_desc = f"The {sname} skewer extends, tip at ({tdx},{tdy}).|The Object extends, tip at ({tdx},{tdy})."
            l2_desc = f"{sname}({tdx},{tdy})|Object({tdx},{tdy})"
        else:
            en_desc = f"The {sname} skewer shrinks, tip at ({tdx},{tdy}).|The Object shrinks, tip at ({tdx},{tdy})."
            l2_desc = f"{sname}({tdx},{tdy})|Object({tdx},{tdy})"
        _emit = self._log
        _emit(
            f"[{verb}]:[{sname}]:[sname({sname});position({fdx}_{fdy})_({tdx}_{tdy});bbox({bdw}_{bdh})]:{en_desc}",
            f"[{verb}]:[{sname}]:[sname({sname});position({fdx}_{fdy})_({tdx}_{tdy});bbox({bdw}_{bdh})]:{l2_desc}"
        )

    def _apply_move(self) -> None:
        """




        """
        self.qiercdohl -= 1
        move_x, move_y = self._action_to_direction(self.action.id)
        dx, dy = (move_x * STEP_SIZE, move_y * STEP_SIZE)
        active = self.vzvypfsnt
        base_dir_x, base_dir_y = ROTATION_TO_DIRECTION[active.rotation]
        sprites_to_move: set[Sprite] = set()

        if (move_x, move_y) == (base_dir_x, base_dir_y):

            if self._is_out_of_bounds(self.mwfajkguqx[active][-1], move_x, move_y):
                return
            for seg in self.mwfajkguqx[active]:
                self._collect_movable(seg, move_x, move_y, sprites_to_move)

            new_seg = sprites["skewer_segment"].clone().set_position(active.x, active.y)
            new_seg.set_layer(1 if active.rotation in (0, 180) else 0)
            new_seg.color_remap(SEGMENT_INACTIVE_DIM, SEGMENT_ACTIVE_BRIGHT)
            new_seg.color_remap(SEGMENT_INACTIVE_DARK, SEGMENT_ACTIVE_DIM)
            self.current_level.add_sprite(new_seg.set_rotation(active.rotation))
            self.mwfajkguqx[active].insert(0, new_seg)


            self._extend_emit_info = (active, "extend")
            self._extend_anim_active = True
        elif (move_x, move_y) == (-base_dir_x, -base_dir_y):

            if len(self.mwfajkguqx[active]) == 1:
                return
            self.hadfnehqh = self.mwfajkguqx[active].pop(0)
            for seg in self.mwfajkguqx[active]:
                self._collect_movable(seg, move_x, move_y, sprites_to_move)

            self._extend_emit_info = (active, "shrink")
            self._extend_anim_active = True
        elif self.current_level.get_sprite_at(active.x + 2 + dx // 2, active.y + 2 + dy // 2, "rail"):

            for seg in self.mwfajkguqx[active]:
                if not self._collect_movable(seg, move_x, move_y, sprites_to_move):
                    return
            sprites_to_move.add(active)
        else:
            return


        if self.pzzwlsmdt:
            for i, target in enumerate(self.pzzwlsmdt):
                preview = self.rztawzist[i]
                preview.set_position(target.x, target.y)
                preview.set_visible(True).color_remap(None, target.pixels[1, 1])
                preview.set_rotation(0 if move_x != 0 else 90)

            for _block in self.pzzwlsmdt:
                _bx, _by, _bw, _bh = self._to_display_bbox(_block)
                self._log(
                    f"[change]:[{_block.name}]:[position({_bx}_{_by});bbox({_bw}_{_bh});shape(normal)_(extended)]:{_block.name} at ({_bx},{_by}) appears to stretch as the skewer pushes through.|Object at ({_bx},{_by}) appears to stretch as the skewer pushes through.",
                    f"[change]:[{_block.name}]:[position({_bx}_{_by});bbox({_bw}_{_bh});shape(normal)_(extended)]:({_bx},{_by}){_block.name}|({_bx},{_by})Object"
                )


        if sprites_to_move:
            for spr in sprites_to_move:
                self.ljprkjlji.append((spr, (spr.x + dx, spr.y + dy)))
            self._animate_sprites()

    def _animate_sprites(self) -> None:
        """"""
        for spr, (target_x, target_y) in self.ljprkjlji:
            dx, dy = (target_x - spr.x, target_y - spr.y)
            spr.move(3 * np.sign(dx), 3 * np.sign(dy))

            if spr in self.pzzwlsmdt:
                index = self.pzzwlsmdt.index(spr)
                self.rztawzist[index].set_position(spr.x, spr.y)

    def _select_skewer(self, target: Sprite, initial: bool = False) -> None:
        """








        """
        def _emit_skewer_bright(skewer: Sprite, state_en: str,
                                  desc_kind_en: str, desc_kind_l2: str) -> None:
            """



            """
            _dx, _dy, _dw, _dh = self._to_display_bbox(skewer)
            if state_en == "selected":
                _sec_en = "lights up as a highlight"
                _sec_l2 = ""
            else:
                _sec_en = "turns off its highlight"
                _sec_l2 = ""
            self._log(
                f"[bright]:[{skewer.name}]:[sname({skewer.name});position({_dx}_{_dy});bbox({_dw}_{_dh});state({state_en})]:"
                f"{skewer.name} at ({_dx},{_dy}) {desc_kind_en}.|Object at ({_dx},{_dy}) {_sec_en}.",
                f"[bright]:[{skewer.name}]:[sname({skewer.name});position({_dx}_{_dy});bbox({_dw}_{_dh});state({state_en})]:"
                f"({_dx},{_dy}){skewer.name}{desc_kind_l2}|({_dx},{_dy})Object{_sec_l2}",
            )

        if not initial and self.vzvypfsnt:

            old_active = self.vzvypfsnt
            for skewer in (old_active, self.xpmcmtbcv.get(old_active)):
                if not skewer:
                    continue
                skewer.color_remap(SKEWER_ACTIVE_BORDER, SKEWER_DEFAULT_BORDER)
                for seg in self.mwfajkguqx[skewer]:
                    seg.color_remap(SEGMENT_ACTIVE_DIM, SEGMENT_INACTIVE_DARK)
                    seg.color_remap(SEGMENT_ACTIVE_BRIGHT, SEGMENT_INACTIVE_DIM)

            for sk in (old_active, self.xpmcmtbcv.get(old_active)):
                if sk:
                    _emit_skewer_bright(
                        sk, "off",
                        "is deselected and its highlight turns off",
                        "",
                    )


        self.vzvypfsnt = target
        for skewer in (self.vzvypfsnt, self.xpmcmtbcv.get(self.vzvypfsnt)):
            if not skewer:
                continue
            skewer.color_remap(SKEWER_DEFAULT_BORDER, SKEWER_ACTIVE_BORDER)
            for seg in self.mwfajkguqx[skewer]:
                seg.color_remap(SEGMENT_INACTIVE_DIM, SEGMENT_ACTIVE_BRIGHT)
                seg.color_remap(SEGMENT_INACTIVE_DARK, SEGMENT_ACTIVE_DIM)
        if not initial:

            for sk in (target, self.xpmcmtbcv.get(target)):
                if sk:
                    _emit_skewer_bright(
                        sk, "selected",
                        "is selected and highlighted",
                        "",
                    )

    def _check_win_and_update_indicators(self) -> bool:
        """








        """

        for skewer, segs in self.mwfajkguqx.items():
            self.vjfbwggsd[skewer].clear()
            for seg in segs:
                block = self._find_color_block_at(seg.x, seg.y)
                if block:
                    self.vjfbwggsd[skewer].append(block)

        all_matched = True
        for top_skewer, bottom_skewer in self.xpmcmtbcv.items():
            for i, indicator in enumerate(self.jdojcthkf[bottom_skewer]):
                if i >= len(self.vjfbwggsd[top_skewer]):

                    indicator.set_visible(False)
                    all_matched = False
                elif (
                    self.vjfbwggsd[top_skewer][i].pixels[1, 1]
                    == self.vjfbwggsd[bottom_skewer][i].pixels[1, 1]
                ):

                    indicator.set_visible(True)
                else:

                    indicator.set_visible(False)
                    all_matched = False
        return all_matched

    def _save_undo_state(self) -> None:
        """"""
        snapshot: list[tuple[Sprite, int, int, int]] = []
        for skewer, segs in self.mwfajkguqx.items():
            snapshot.append((skewer, skewer.x, skewer.y, len(segs)))
        for block in self.vbelzuaian:
            snapshot.append((block, block.x, block.y, 0))
        self.seghobzez.append(snapshot)

    def _undo(self) -> None:
        """




        """
        if len(self.seghobzez) < 2:
            return
        self.seghobzez.pop()
        snapshot = self.seghobzez[-1]


        for segs in self.mwfajkguqx.values():
            for seg in segs:
                self.current_level.remove_sprite(seg)
            segs.clear()


        for spr, x, y, seg_count in snapshot:
            spr.set_position(x, y)
            if "skewer" in spr.tags:
                dir_x, dir_y = ROTATION_TO_DIRECTION[spr.rotation]
                dx, dy = (dir_x * STEP_SIZE, dir_y * STEP_SIZE)
                for i in range(seg_count):
                    seg = sprites["skewer_segment"].clone().set_rotation(spr.rotation)
                    seg.set_position(spr.x + i * dx, spr.y + i * dy)
                    seg.set_layer(1 if spr.rotation in (0, 180) else 0)
                    self.current_level.add_sprite(seg)
                    self.mwfajkguqx[spr].append(seg)


        self._select_skewer(self.vzvypfsnt, initial=True)
        self._check_win_and_update_indicators()

    def _collect_movable(
        self,
        sprite: Sprite,
        dir_x: int,
        dir_y: int,
        result: set[Sprite],
        exclude: Sprite | None = None,
    ) -> bool:
        """
















        """
        if sprite in result:
            return True



        is_segment = "skewer_segment" in sprite.tags

        if self._is_out_of_bounds(sprite, dir_x, dir_y):
            if not (
                is_segment
                and (
                    ROTATION_TO_DIRECTION[sprite.rotation] == (-dir_x, -dir_y)
                    or self._is_out_of_bounds(sprite, 0, 0)
                )
            ):
                return False
        dx, dy = (dir_x * STEP_SIZE, dir_y * STEP_SIZE)

        if is_segment:

            for offset_x, offset_y in ((0, 0), (dx, dy)):
                block = self._find_color_block_at(sprite.x + offset_x, sprite.y + offset_y)
                if block:
                    seg_dir_x, _ = ROTATION_TO_DIRECTION[sprite.rotation]
                    if self._collect_movable(block, dir_x, dir_y, result, exclude=sprite):
                        result.add(block)
                    elif (seg_dir_x == 0) != (dir_x == 0):
                        return False
                    elif not self._find_segment_at(block.x, block.y, dir_x != 0):
                        self.pzzwlsmdt.append(block)
        else:

            is_horiz = dir_x != 0
            for offset_x, offset_y in ((0, 0), (dx, dy)):
                seg = self._find_segment_at(
                    sprite.x + offset_x,
                    sprite.y + offset_y,
                    not is_horiz,
                    exclude=exclude,
                )
                if seg:
                    seg_dir_x, _ = ROTATION_TO_DIRECTION[seg.rotation]
                    if (seg_dir_x == 0) != (dir_x == 0):
                        return False

            next_block = self._find_color_block_at(sprite.x + dx, sprite.y + dy)
            if next_block and not self._collect_movable(next_block, dir_x, dir_y, result):
                return False

            if (
                not self._find_segment_at(sprite.x, sprite.y, is_horiz)
                and self._find_segment_at(sprite.x + dx, sprite.y + dy, is_horiz)
            ):
                self.pzzwlsmdt.append(sprite)


        if len(self.pzzwlsmdt) > len(self.rztawzist):
            self.rztawzist.append(sprites["drop_preview"].clone().set_visible(False))
            self.current_level.add_sprite(self.rztawzist[-1])

        result.add(sprite)
        return True

    def _find_color_block_at(self, x: int, y: int) -> Sprite | None:
        """"""
        for block in self.vbelzuaian:
            if block.x == x and block.y == y:
                return block
        return None

    def _find_segment_at(
        self,
        x: int,
        y: int,
        is_horizontal: bool,
        exclude: Sprite | None = None,
    ) -> Sprite | None:
        """












        """
        for seg in self.current_level.get_sprites_by_tag("skewer_segment"):
            if seg == exclude:
                continue
            if seg.x == x and seg.y == y and (is_horizontal == (seg.rotation in (0, 180))):
                return seg
        return None

    def _is_out_of_bounds(self, sprite: Sprite, dir_x: int, dir_y: int) -> bool:
        """





        """
        dx, dy = (dir_x * STEP_SIZE, dir_y * STEP_SIZE)

        platform_right  = self.lqwkgffeb.x + self.lqwkgffeb.pixels.shape[1] * STEP_SIZE
        platform_bottom = self.lqwkgffeb.y + self.lqwkgffeb.pixels.shape[0] * STEP_SIZE
        x, y = (sprite.x + dx, sprite.y + dy)

        if x < self.lqwkgffeb.x or x + sprite.width > platform_right:
            return True
        if y < self.lqwkgffeb.y or y + sprite.height > platform_bottom:
            return True

        if self.current_level.get_sprite_at(x, y, "wall_block"):
            return True
        return False

    def _action_to_direction(self, action: GameAction) -> tuple[int, int]:
        """





        """
        if action == GameAction.ACTION1:
            return (0, -1)
        elif action == GameAction.ACTION2:
            return (0, 1)
        elif action == GameAction.ACTION3:
            return (-1, 0)
        elif action == GameAction.ACTION4:
            return (1, 0)
        return (0, 0)

    def _get_hidden_state(self) -> np.ndarray:
        """"""
        return np.array([self.qiercdohl, self.xsoixcero])

    def _get_valid_actions(self) -> list[ActionInput]:
        """



        """
        direction_actions = [
            ActionInput(id=action)
            for action in (
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
            )
        ]
        if self.xsoixcero:
            return direction_actions


        click_actions = []
        for top_skewer in self.xpmcmtbcv.keys():
            if top_skewer != self.vzvypfsnt and not top_skewer.name.startswith("wallBase"):
                click_actions.append(
                    ActionInput(
                        id=GameAction.ACTION6,
                        data={"x": top_skewer.x + 2, "y": top_skewer.y + 2},
                    )
                )
        return direction_actions + click_actions





    @property
    def _step_log(self) -> list[str]:
        return self._log_state._step_log
    @_step_log.setter
    def _step_log(self, value: list[str]) -> None:
        self._log_state._step_log = value

    @property
    def _step_log_l2(self) -> list[str]:
        return self._log_state._step_log_l2
    @_step_log_l2.setter
    def _step_log_l2(self, value: list[str]) -> None:
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
        self.action_log    = []
        self.action_log_l2 = []

    def get_action_log_frame(self) -> tuple[list[list[str]], list[list[str]]]:
        return self._log_state.get_frame_logs()

    def get_action_log(self) -> tuple[list[str], list[str]]:
        """






        """
        self._finalize_action_log()
        en, l2 = list(self.action_log), list(self.action_log_l2)
        reset_evt = self._log_state.get_event("reset")
        if reset_evt:
            r_en = reset_evt.get("en", "[reset]:[game]:[]:The game state is reset.") \
                if isinstance(reset_evt, dict) else "[reset]:[game]:[]:The game state is reset."
            r_l2 = reset_evt.get("l2", "[reset]:[game]:[]:") \
                if isinstance(reset_evt, dict) else "[reset]:[game]:[]:"
            en.insert(0, r_en)
            l2.insert(0, r_l2)
        return en, l2

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
            "name":     "steps_remaining",
            "label":    "Steps Remaining",
            "label_l2": ''                     ,
            "value":    self.qiercdohl,
            "max":      self.vhzjwcpmk,
            "x":        0, "y":      PROGRESS_BAR_ROW,
            "width":    64, "height": 1,
        })
        return result

    def get_game_knowledge(self) -> dict:
        """


        """
        return GAME_KNOWLEDGE
