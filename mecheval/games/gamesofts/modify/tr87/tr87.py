"""





































"""

import math
import random

import numpy as np
from arcengine import (
    ARCBaseGame,
    BlockingMode,
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

    "background": Sprite(
        pixels=[
            [GRAY, GRAY],
        ],
        name="background",
        visible=True,
        collidable=True,
        layer=-2,
    ),

    "background_strip": Sprite(
        pixels=[
            [GRAY],
            [GRAY],
        ],
        name="background_strip",
        visible=True,
        collidable=True,
        layer=-2,
    ),

    "rule_panel_blueA": Sprite(
        pixels=[
            [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE],
            [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE],
            [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE],
            [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE],
            [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE],
            [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE],
            [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE],
        ],
        name="rule_panel_blueA",
        visible=True,
        collidable=True,
        tags=["rule_panel"],
        layer=-1,
    ),

    "rule_panel_pinkB": Sprite(
        pixels=[
            [PINK, PINK, PINK, PINK, PINK, PINK, PINK],
            [PINK, PINK, PINK, PINK, PINK, PINK, PINK],
            [PINK, PINK, PINK, PINK, PINK, PINK, PINK],
            [PINK, PINK, PINK, PINK, PINK, PINK, PINK],
            [PINK, PINK, PINK, PINK, PINK, PINK, PINK],
            [PINK, PINK, PINK, PINK, PINK, PINK, PINK],
            [PINK, PINK, PINK, PINK, PINK, PINK, PINK],
        ],
        name="rule_panel_pinkB",
        visible=True,
        collidable=True,
        tags=["rule_panel"],
        layer=-1,
    ),

    "rule_panel_yellowC": Sprite(
        pixels=[
            [YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW],
            [YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW],
            [YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW],
            [YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW],
            [YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW],
            [YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW],
            [YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW],
        ],
        name="rule_panel_yellowC",
        visible=True,
        collidable=True,
        tags=["rule_panel"],
        layer=-1,
    ),

    "rule_divider": Sprite(
        pixels=[
            [GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, GRAY],
        ],
        name="rule_divider",
        visible=True,
        collidable=True,
        layer=-2,
    ),


    "symbol_setA1": Sprite(
        pixels=[
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [TRANSPARENT, BLACK, BLACK, BLACK, TRANSPARENT],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setA1",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setA2": Sprite(
        pixels=[
            [TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
        ],
        name="symbol_setA2",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setA3": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setA3",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setA4": Sprite(
        pixels=[
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
        ],
        name="symbol_setA4",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setA5": Sprite(
        pixels=[
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
        ],
        name="symbol_setA5",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setA6": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setA6",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setA7": Sprite(
        pixels=[
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, TRANSPARENT, BLACK, BLACK],
        ],
        name="symbol_setA7",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),

    "symbol_setB1": Sprite(
        pixels=[
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, TRANSPARENT],
            [BLACK, TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT],
            [BLACK, TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setB1",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setB2": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setB2",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setB3": Sprite(
        pixels=[
            [TRANSPARENT, TRANSPARENT, BLACK, BLACK, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, TRANSPARENT, TRANSPARENT],
        ],
        name="symbol_setB3",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setB4": Sprite(
        pixels=[
            [TRANSPARENT, BLACK, BLACK, BLACK, TRANSPARENT],
            [TRANSPARENT, BLACK, TRANSPARENT, BLACK, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setB4",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setB5": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
        ],
        name="symbol_setB5",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setB6": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, BLACK, TRANSPARENT],
            [BLACK, TRANSPARENT, TRANSPARENT, BLACK, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, TRANSPARENT, TRANSPARENT, BLACK],
            [TRANSPARENT, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setB6",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setB7": Sprite(
        pixels=[
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
        ],
        name="symbol_setB7",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),

    "symbol_setC1": Sprite(
        pixels=[
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, BLACK, BLACK],
        ],
        name="symbol_setC1",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setC2": Sprite(
        pixels=[
            [BLACK, BLACK, BLACK, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, BLACK, BLACK, BLACK],
        ],
        name="symbol_setC2",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setC3": Sprite(
        pixels=[
            [BLACK, TRANSPARENT, BLACK, BLACK, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, BLACK, BLACK, TRANSPARENT, BLACK],
        ],
        name="symbol_setC3",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setC4": Sprite(
        pixels=[
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [TRANSPARENT, TRANSPARENT, BLACK, TRANSPARENT, TRANSPARENT],
        ],
        name="symbol_setC4",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setC5": Sprite(
        pixels=[
            [BLACK, BLACK, TRANSPARENT, BLACK, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, TRANSPARENT, BLACK, BLACK],
        ],
        name="symbol_setC5",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setC6": Sprite(
        pixels=[
            [BLACK, TRANSPARENT, BLACK, BLACK, BLACK],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
            [BLACK, BLACK, BLACK, TRANSPARENT, BLACK],
            [TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, BLACK, TRANSPARENT, BLACK],
        ],
        name="symbol_setC6",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),
    "symbol_setC7": Sprite(
        pixels=[
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT],
            [BLACK, TRANSPARENT, TRANSPARENT, TRANSPARENT, BLACK],
            [BLACK, BLACK, TRANSPARENT, BLACK, BLACK],
            [TRANSPARENT, BLACK, TRANSPARENT, BLACK, TRANSPARENT],
        ],
        name="symbol_setC7",
        visible=True,
        collidable=True,
        tags=["symbol"],
    ),

    "rule_highlight_bg": Sprite(
        pixels=[
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
        ],
        name="rule_highlight_bg",
        visible=True,
        collidable=True,
        layer=-2,
    ),

    "selector_top1": Sprite(
        pixels=[
            [WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, TRANSPARENT, TRANSPARENT, TRANSPARENT, WHITE],
        ],
        name="selector_top1",
        visible=True,
        collidable=True,
    ),

    "selector_top2": Sprite(
        pixels=[
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, WHITE],
        ],
        name="selector_top2",
        visible=True,
        collidable=True,
    ),

    "selector_top3": Sprite(
        pixels=[
            [WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE],
            [WHITE, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, TRANSPARENT, WHITE],
        ],
        name="selector_top3",
        visible=True,
        collidable=True,
    ),
}


levels = [

    Level(
        sprites=[
            sprites["background"].clone().set_position(0, 34).set_scale(32),
            sprites["rule_panel_blueA"].clone().set_position(12, 4),
            sprites["rule_panel_blueA"].clone().set_position(12, 13),
            sprites["rule_panel_blueA"].clone().set_position(12, 22),
            sprites["rule_panel_blueA"].clone().set_position(35, 13),
            sprites["rule_panel_blueA"].clone().set_position(35, 4),
            sprites["rule_panel_blueA"].clone().set_position(35, 22),
            sprites["rule_panel_blueA"].clone().set_position(14, 40),
            sprites["rule_panel_blueA"].clone().set_position(21, 40),
            sprites["rule_panel_blueA"].clone().set_position(35, 40),
            sprites["rule_panel_blueA"].clone().set_position(28, 40),
            sprites["rule_panel_blueA"].clone().set_position(42, 40),
            sprites["rule_panel_pinkB"].clone().set_position(22, 4),
            sprites["rule_panel_pinkB"].clone().set_position(22, 13),
            sprites["rule_panel_pinkB"].clone().set_position(22, 22),
            sprites["rule_panel_pinkB"].clone().set_position(45, 13),
            sprites["rule_panel_pinkB"].clone().set_position(45, 22),
            sprites["rule_panel_pinkB"].clone().set_position(45, 4),
            sprites["rule_panel_pinkB"].clone().set_position(14, 51),
            sprites["rule_panel_pinkB"].clone().set_position(21, 51),
            sprites["rule_panel_pinkB"].clone().set_position(28, 51),
            sprites["rule_panel_pinkB"].clone().set_position(35, 51),
            sprites["rule_panel_pinkB"].clone().set_position(42, 51),
            sprites["rule_divider"].clone().set_position(15, 7),
            sprites["rule_divider"].clone().set_position(15, 16),
            sprites["rule_divider"].clone().set_position(15, 25),
            sprites["rule_divider"].clone().set_position(38, 16),
            sprites["rule_divider"].clone().set_position(38, 7),
            sprites["rule_divider"].clone().set_position(38, 25),
            sprites["symbol_setA1"].clone().set_position(13, 14),
            sprites["symbol_setA1"].clone().set_position(43, 41),
            sprites["symbol_setA2"].clone().set_position(13, 23),
            sprites["symbol_setA2"].clone().set_position(22, 41),
            sprites["symbol_setA3"].clone().set_position(13, 5),
            sprites["symbol_setA3"].clone().set_position(29, 41),
            sprites["symbol_setA4"].clone().set_position(36, 14),
            sprites["symbol_setA4"].clone().set_position(15, 41),
            sprites["symbol_setA5"].clone().set_position(36, 5),
            sprites["symbol_setA5"].clone().set_position(36, 41),
            sprites["symbol_setA7"].clone().set_position(36, 23),
            sprites["symbol_setB1"].clone().set_position(23, 14),
            sprites["symbol_setB2"].clone().set_position(23, 23),
            sprites["symbol_setB3"].clone().set_position(46, 14),
            sprites["symbol_setB5"].clone().set_position(46, 5),
            sprites["symbol_setB6"].clone().set_position(23, 5),
            sprites["symbol_setB6"].clone().set_position(15, 52),
            sprites["symbol_setB6"].clone().set_position(22, 52),
            sprites["symbol_setB6"].clone().set_position(29, 52),
            sprites["symbol_setB6"].clone().set_position(36, 52),
            sprites["symbol_setB6"].clone().set_position(43, 52),
            sprites["symbol_setB7"].clone().set_position(46, 23),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["background"].clone().set_position(0, 34).set_scale(32),
            sprites["rule_panel_pinkB"].clone().set_position(5, 13),
            sprites["rule_panel_pinkB"].clone().set_position(5, 4),
            sprites["rule_panel_pinkB"].clone().set_position(5, 22),
            sprites["rule_panel_pinkB"].clone().set_position(42, 22),
            sprites["rule_panel_pinkB"].clone().set_position(28, 4),
            sprites["rule_panel_pinkB"].clone().set_position(35, 13),
            sprites["rule_panel_pinkB"].clone().set_position(18, 40),
            sprites["rule_panel_pinkB"].clone().set_position(25, 40),
            sprites["rule_panel_pinkB"].clone().set_position(32, 40),
            sprites["rule_panel_pinkB"].clone().set_position(39, 40),
            sprites["rule_panel_yellowC"].clone().set_position(15, 13),
            sprites["rule_panel_yellowC"].clone().set_position(22, 13),
            sprites["rule_panel_yellowC"].clone().set_position(15, 4),
            sprites["rule_panel_yellowC"].clone().set_position(15, 22),
            sprites["rule_panel_yellowC"].clone().set_position(22, 22),
            sprites["rule_panel_yellowC"].clone().set_position(29, 22),
            sprites["rule_panel_yellowC"].clone().set_position(7, 51),
            sprites["rule_panel_yellowC"].clone().set_position(52, 22),
            sprites["rule_panel_yellowC"].clone().set_position(38, 4),
            sprites["rule_panel_yellowC"].clone().set_position(45, 4),
            sprites["rule_panel_yellowC"].clone().set_position(52, 4),
            sprites["rule_panel_yellowC"].clone().set_position(45, 13),
            sprites["rule_panel_yellowC"].clone().set_position(52, 13),
            sprites["rule_panel_yellowC"].clone().set_position(14, 51),
            sprites["rule_panel_yellowC"].clone().set_position(28, 51),
            sprites["rule_panel_yellowC"].clone().set_position(21, 51),
            sprites["rule_panel_yellowC"].clone().set_position(49, 51),
            sprites["rule_panel_yellowC"].clone().set_position(42, 51),
            sprites["rule_panel_yellowC"].clone().set_position(35, 51),
            sprites["rule_divider"].clone().set_position(8, 16),
            sprites["rule_divider"].clone().set_position(8, 7),
            sprites["rule_divider"].clone().set_position(8, 25),
            sprites["rule_divider"].clone().set_position(38, 16),
            sprites["rule_divider"].clone().set_position(31, 7),
            sprites["rule_divider"].clone().set_position(45, 25),
            sprites["symbol_setB1"].clone().set_position(6, 5),
            sprites["symbol_setB1"].clone().set_position(19, 41),
            sprites["symbol_setB3"].clone().set_position(6, 23),
            sprites["symbol_setB3"].clone().set_position(26, 41),
            sprites["symbol_setB4"].clone().set_position(29, 5),
            sprites["symbol_setB5"].clone().set_position(36, 14),
            sprites["symbol_setB5"].clone().set_position(33, 41),
            sprites["symbol_setB6"].clone().set_position(6, 14),
            sprites["symbol_setB7"].clone().set_position(43, 23),
            sprites["symbol_setB7"].clone().set_position(40, 41),
            sprites["symbol_setC1"].clone().set_position(16, 23),
            sprites["symbol_setC1"].clone().set_position(30, 23),
            sprites["symbol_setC2"].clone().set_position(23, 14),
            sprites["symbol_setC2"].clone().set_position(46, 14),
            sprites["symbol_setC2"].clone().set_position(53, 14),
            sprites["symbol_setC3"].clone().set_position(16, 5),
            sprites["symbol_setC3"].clone().set_position(46, 5),
            sprites["symbol_setC4"].clone().set_position(16, 14),
            sprites["symbol_setC4"].clone().set_position(39, 5),
            sprites["symbol_setC5"].clone().set_position(23, 23),
            sprites["symbol_setC5"].clone().set_position(8, 52),
            sprites["symbol_setC5"].clone().set_position(15, 52),
            sprites["symbol_setC5"].clone().set_position(29, 52),
            sprites["symbol_setC5"].clone().set_position(22, 52),
            sprites["symbol_setC5"].clone().set_position(50, 52),
            sprites["symbol_setC5"].clone().set_position(43, 52),
            sprites["symbol_setC5"].clone().set_position(36, 52),
            sprites["symbol_setC6"].clone().set_position(53, 5),
            sprites["symbol_setC7"].clone().set_position(53, 23),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["background"].clone().set_position(0, 34).set_scale(32),
            sprites["rule_panel_blueA"].clone().set_position(7, 51),
            sprites["rule_panel_blueA"].clone().set_position(14, 51),
            sprites["rule_panel_blueA"].clone().set_position(28, 51),
            sprites["rule_panel_blueA"].clone().set_position(21, 51),
            sprites["rule_panel_blueA"].clone().set_position(42, 51),
            sprites["rule_panel_blueA"].clone().set_position(35, 51),
            sprites["rule_panel_blueA"].clone().set_position(29, 22),
            sprites["rule_panel_blueA"].clone().set_position(52, 13),
            sprites["rule_panel_blueA"].clone().set_position(45, 4),
            sprites["rule_panel_blueA"].clone().set_position(15, 4),
            sprites["rule_panel_blueA"].clone().set_position(52, 4),
            sprites["rule_panel_blueA"].clone().set_position(52, 22),
            sprites["rule_panel_blueA"].clone().set_position(22, 13),
            sprites["rule_panel_blueA"].clone().set_position(15, 13),
            sprites["rule_panel_blueA"].clone().set_position(49, 51),
            sprites["rule_panel_yellowC"].clone().set_position(5, 13),
            sprites["rule_panel_yellowC"].clone().set_position(5, 4),
            sprites["rule_panel_yellowC"].clone().set_position(5, 22),
            sprites["rule_panel_yellowC"].clone().set_position(12, 22),
            sprites["rule_panel_yellowC"].clone().set_position(19, 22),
            sprites["rule_panel_yellowC"].clone().set_position(4, 40),
            sprites["rule_panel_yellowC"].clone().set_position(42, 22),
            sprites["rule_panel_yellowC"].clone().set_position(28, 4),
            sprites["rule_panel_yellowC"].clone().set_position(35, 4),
            sprites["rule_panel_yellowC"].clone().set_position(35, 13),
            sprites["rule_panel_yellowC"].clone().set_position(42, 13),
            sprites["rule_panel_yellowC"].clone().set_position(11, 40),
            sprites["rule_panel_yellowC"].clone().set_position(25, 40),
            sprites["rule_panel_yellowC"].clone().set_position(18, 40),
            sprites["rule_panel_yellowC"].clone().set_position(53, 40),
            sprites["rule_panel_yellowC"].clone().set_position(46, 40),
            sprites["rule_panel_yellowC"].clone().set_position(39, 40),
            sprites["rule_panel_yellowC"].clone().set_position(32, 40),
            sprites["rule_divider"].clone().set_position(8, 16),
            sprites["rule_divider"].clone().set_position(8, 7),
            sprites["rule_divider"].clone().set_position(22, 25),
            sprites["rule_divider"].clone().set_position(45, 16),
            sprites["rule_divider"].clone().set_position(38, 7),
            sprites["rule_divider"].clone().set_position(45, 25),
            sprites["symbol_setA1"].clone().set_position(53, 5),
            sprites["symbol_setA3"].clone().set_position(22, 52),
            sprites["symbol_setA3"].clone().set_position(8, 52),
            sprites["symbol_setA3"].clone().set_position(15, 52),
            sprites["symbol_setA3"].clone().set_position(29, 52),
            sprites["symbol_setA3"].clone().set_position(43, 52),
            sprites["symbol_setA3"].clone().set_position(36, 52),
            sprites["symbol_setA3"].clone().set_position(53, 14),
            sprites["symbol_setA3"].clone().set_position(50, 52),
            sprites["symbol_setA4"].clone().set_position(16, 5),
            sprites["symbol_setA5"].clone().set_position(53, 23),
            sprites["symbol_setA6"].clone().set_position(30, 23),
            sprites["symbol_setA6"].clone().set_position(46, 5),
            sprites["symbol_setA7"].clone().set_position(23, 14),
            sprites["symbol_setA7"].clone().set_position(16, 14),
            sprites["symbol_setC1"].clone().set_position(6, 23),
            sprites["symbol_setC1"].clone().set_position(20, 23),
            sprites["symbol_setC1"].clone().set_position(12, 41),
            sprites["symbol_setC1"].clone().set_position(26, 41),
            sprites["symbol_setC2"].clone().set_position(43, 23),
            sprites["symbol_setC2"].clone().set_position(40, 41),
            sprites["symbol_setC3"].clone().set_position(36, 5),
            sprites["symbol_setC3"].clone().set_position(29, 5),
            sprites["symbol_setC3"].clone().set_position(47, 41),
            sprites["symbol_setC3"].clone().set_position(54, 41),
            sprites["symbol_setC4"].clone().set_position(6, 14),
            sprites["symbol_setC4"].clone().set_position(33, 41),
            sprites["symbol_setC5"].clone().set_position(13, 23),
            sprites["symbol_setC5"].clone().set_position(19, 41),
            sprites["symbol_setC6"].clone().set_position(6, 5),
            sprites["symbol_setC6"].clone().set_position(5, 41),
            sprites["symbol_setC7"].clone().set_position(36, 14),
            sprites["symbol_setC7"].clone().set_position(43, 14),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["background"].clone().set_position(0, 37).set_scale(32),
            sprites["rule_panel_blueA"].clone().set_position(12, 11),
            sprites["rule_panel_blueA"].clone().set_position(12, 27),
            sprites["rule_panel_blueA"].clone().set_position(35, 3),
            sprites["rule_panel_blueA"].clone().set_position(35, 19),
            sprites["rule_panel_blueA"].clone().set_position(7, 40),
            sprites["rule_panel_blueA"].clone().set_position(14, 40),
            sprites["rule_panel_blueA"].clone().set_position(28, 40),
            sprites["rule_panel_blueA"].clone().set_position(21, 40),
            sprites["rule_panel_blueA"].clone().set_position(42, 40),
            sprites["rule_panel_blueA"].clone().set_position(35, 40),
            sprites["rule_panel_blueA"].clone().set_position(49, 40),
            sprites["rule_panel_pinkB"].clone().set_position(22, 11),
            sprites["rule_panel_pinkB"].clone().set_position(22, 27),
            sprites["rule_panel_pinkB"].clone().set_position(45, 19),
            sprites["rule_panel_pinkB"].clone().set_position(45, 3),
            sprites["rule_panel_pinkB"].clone().set_position(12, 19),
            sprites["rule_panel_pinkB"].clone().set_position(12, 3),
            sprites["rule_panel_pinkB"].clone().set_position(35, 11),
            sprites["rule_panel_pinkB"].clone().set_position(35, 27),
            sprites["rule_panel_yellowC"].clone().set_position(7, 51),
            sprites["rule_panel_yellowC"].clone().set_position(14, 51),
            sprites["rule_panel_yellowC"].clone().set_position(28, 51),
            sprites["rule_panel_yellowC"].clone().set_position(21, 51),
            sprites["rule_panel_yellowC"].clone().set_position(42, 51),
            sprites["rule_panel_yellowC"].clone().set_position(35, 51),
            sprites["rule_panel_yellowC"].clone().set_position(45, 27),
            sprites["rule_panel_yellowC"].clone().set_position(45, 11),
            sprites["rule_panel_yellowC"].clone().set_position(22, 3),
            sprites["rule_panel_yellowC"].clone().set_position(22, 19),
            sprites["rule_panel_yellowC"].clone().set_position(49, 51),
            sprites["rule_divider"].clone().set_position(15, 6),
            sprites["rule_divider"].clone().set_position(15, 14),
            sprites["rule_divider"].clone().set_position(15, 22),
            sprites["rule_divider"].clone().set_position(15, 30),
            sprites["rule_divider"].clone().set_position(38, 6),
            sprites["rule_divider"].clone().set_position(38, 14),
            sprites["rule_divider"].clone().set_position(38, 22),
            sprites["rule_divider"].clone().set_position(38, 30),
            sprites["symbol_setA1"].clone().set_position(13, 12),
            sprites["symbol_setA1"].clone().set_position(15, 41),
            sprites["symbol_setA1"].clone().set_position(36, 41),
            sprites["symbol_setA4"].clone().set_position(13, 28),
            sprites["symbol_setA4"].clone().set_position(50, 41),
            sprites["symbol_setA4"].clone().set_position(22, 41),
            sprites["symbol_setA6"].clone().set_position(36, 4),
            sprites["symbol_setA6"].clone().set_position(8, 41),
            sprites["symbol_setA6"].clone().set_position(43, 41),
            sprites["symbol_setA7"].clone().set_position(36, 20),
            sprites["symbol_setA7"].clone().set_position(29, 41),
            sprites["symbol_setB1"].clone().set_position(46, 4),
            sprites["symbol_setB1"].clone().set_position(13, 20),
            sprites["symbol_setB3"].clone().set_position(23, 12),
            sprites["symbol_setB3"].clone().set_position(36, 28),
            sprites["symbol_setB6"].clone().set_position(46, 20),
            sprites["symbol_setB6"].clone().set_position(36, 12),
            sprites["symbol_setB7"].clone().set_position(23, 28),
            sprites["symbol_setB7"].clone().set_position(13, 4),
            sprites["symbol_setC1"].clone().set_position(46, 12),
            sprites["symbol_setC2"].clone().set_position(46, 28),
            sprites["symbol_setC3"].clone().set_position(23, 20),
            sprites["symbol_setC5"].clone().set_position(8, 52),
            sprites["symbol_setC5"].clone().set_position(15, 52),
            sprites["symbol_setC5"].clone().set_position(22, 52),
            sprites["symbol_setC5"].clone().set_position(29, 52),
            sprites["symbol_setC5"].clone().set_position(36, 52),
            sprites["symbol_setC5"].clone().set_position(43, 52),
            sprites["symbol_setC5"].clone().set_position(50, 52),
            sprites["symbol_setC7"].clone().set_position(23, 4),
        ],
        grid_size=(64, 64),
        data={
            "double_translation": True,
        },
    ),

    Level(
        sprites=[
            sprites["background"].clone().set_position(0, 38).set_scale(32),
            sprites["rule_panel_blueA"].clone().set_position(8, 10),
            sprites["rule_panel_blueA"].clone().set_position(15, 22),
            sprites["rule_panel_blueA"].clone().set_position(31, 10),
            sprites["rule_panel_blueA"].clone().set_position(38, 22),
            sprites["rule_panel_blueA"].clone().set_position(14, 43),
            sprites["rule_panel_blueA"].clone().set_position(21, 43),
            sprites["rule_panel_blueA"].clone().set_position(35, 43),
            sprites["rule_panel_blueA"].clone().set_position(28, 43),
            sprites["rule_panel_blueA"].clone().set_position(42, 43),
            sprites["rule_panel_blueA"].clone().set_position(8, 22),
            sprites["rule_panel_pinkB"].clone().set_position(18, 10),
            sprites["rule_panel_pinkB"].clone().set_position(25, 22),
            sprites["rule_panel_pinkB"].clone().set_position(48, 22),
            sprites["rule_panel_pinkB"].clone().set_position(41, 10),
            sprites["rule_panel_pinkB"].clone().set_position(14, 52),
            sprites["rule_panel_pinkB"].clone().set_position(21, 52),
            sprites["rule_panel_pinkB"].clone().set_position(28, 52),
            sprites["rule_panel_pinkB"].clone().set_position(35, 52),
            sprites["rule_panel_pinkB"].clone().set_position(42, 52),
            sprites["rule_panel_pinkB"].clone().set_position(48, 10),
            sprites["rule_divider"].clone().set_position(11, 13),
            sprites["rule_divider"].clone().set_position(18, 25),
            sprites["rule_divider"].clone().set_position(34, 13),
            sprites["rule_divider"].clone().set_position(40, 25),
            sprites["symbol_setA1"].clone().set_position(22, 44),
            sprites["symbol_setA2"].clone().set_position(43, 44),
            sprites["symbol_setA3"].clone().set_position(9, 11),
            sprites["symbol_setA3"].clone().set_position(16, 23),
            sprites["symbol_setA3"].clone().set_position(39, 23),
            sprites["symbol_setA3"].clone().set_position(32, 11),
            sprites["symbol_setA3"].clone().set_position(9, 23),
            sprites["symbol_setA4"].clone().set_position(36, 44),
            sprites["symbol_setA4"].clone().set_position(29, 44),
            sprites["symbol_setA5"].clone().set_position(15, 44),
            sprites["symbol_setB1"].clone().set_position(36, 53),
            sprites["symbol_setB2"].clone().set_position(43, 53),
            sprites["symbol_setB3"].clone().set_position(15, 53),
            sprites["symbol_setB5"].clone().set_position(22, 53),
            sprites["symbol_setB5"].clone().set_position(29, 53),
            sprites["symbol_setB6"].clone().set_position(19, 11),
            sprites["symbol_setB6"].clone().set_position(26, 23),
            sprites["symbol_setB6"].clone().set_position(49, 23),
            sprites["symbol_setB6"].clone().set_position(42, 11),
            sprites["symbol_setB6"].clone().set_position(49, 11),
        ],
        grid_size=(64, 64),
        data={
            "alter_rules": True,
        },
    ),

    Level(
        sprites=[
            sprites["background"].clone().set_position(0, 41).set_scale(32),
            sprites["rule_panel_blueA"].clone().set_position(21, 44),
            sprites["rule_panel_blueA"].clone().set_position(35, 44),
            sprites["rule_panel_blueA"].clone().set_position(28, 44),
            sprites["rule_panel_blueA"].clone().set_position(9, 5),
            sprites["rule_panel_blueA"].clone().set_position(9, 17),
            sprites["rule_panel_blueA"].clone().set_position(9, 29),
            sprites["rule_panel_pinkB"].clone().set_position(19, 5),
            sprites["rule_panel_pinkB"].clone().set_position(38, 5),
            sprites["rule_panel_pinkB"].clone().set_position(26, 5),
            sprites["rule_panel_pinkB"].clone().set_position(38, 29),
            sprites["rule_panel_pinkB"].clone().set_position(19, 17),
            sprites["rule_panel_pinkB"].clone().set_position(38, 17),
            sprites["rule_panel_pinkB"].clone().set_position(26, 17),
            sprites["rule_panel_pinkB"].clone().set_position(19, 29),
            sprites["rule_panel_pinkB"].clone().set_position(26, 29),
            sprites["rule_panel_yellowC"].clone().set_position(11, 53),
            sprites["rule_panel_yellowC"].clone().set_position(25, 53),
            sprites["rule_panel_yellowC"].clone().set_position(18, 53),
            sprites["rule_panel_yellowC"].clone().set_position(32, 53),
            sprites["rule_panel_yellowC"].clone().set_position(48, 5),
            sprites["rule_panel_yellowC"].clone().set_position(48, 29),
            sprites["rule_panel_yellowC"].clone().set_position(48, 17),
            sprites["rule_panel_yellowC"].clone().set_position(39, 53),
            sprites["rule_panel_yellowC"].clone().set_position(46, 53),
            sprites["rule_divider"].clone().set_position(12, 8),
            sprites["rule_divider"].clone().set_position(41, 8),
            sprites["rule_divider"].clone().set_position(41, 32),
            sprites["rule_divider"].clone().set_position(12, 20),
            sprites["rule_divider"].clone().set_position(41, 20),
            sprites["rule_divider"].clone().set_position(12, 32),
            sprites["symbol_setA1"].clone().set_position(36, 45),
            sprites["symbol_setA1"].clone().set_position(10, 18),
            sprites["symbol_setA6"].clone().set_position(29, 45),
            sprites["symbol_setA6"].clone().set_position(10, 30),
            sprites["symbol_setA7"].clone().set_position(22, 45),
            sprites["symbol_setA7"].clone().set_position(10, 6),
            sprites["symbol_setB2"].clone().set_position(27, 6),
            sprites["symbol_setB2"].clone().set_position(20, 18),
            sprites["symbol_setB2"].clone().set_position(39, 18),
            sprites["symbol_setB6"].clone().set_position(39, 30),
            sprites["symbol_setB6"].clone().set_position(20, 30),
            sprites["symbol_setB6"].clone().set_position(27, 30),
            sprites["symbol_setB7"].clone().set_position(27, 18),
            sprites["symbol_setB7"].clone().set_position(20, 6),
            sprites["symbol_setB7"].clone().set_position(39, 6),
            sprites["symbol_setC1"].clone().set_position(40, 54),
            sprites["symbol_setC1"].clone().set_position(49, 18),
            sprites["symbol_setC1"].clone().set_position(19, 54),
            sprites["symbol_setC3"].clone().set_position(12, 54),
            sprites["symbol_setC3"].clone().set_position(49, 6),
            sprites["symbol_setC3"].clone().set_position(47, 54),
            sprites["symbol_setC5"].clone().set_position(33, 54),
            sprites["symbol_setC5"].clone().set_position(49, 30),
            sprites["symbol_setC5"].clone().set_position(26, 54),
        ],
        grid_size=(64, 64),
        data={
            "double_translation": True,
            "alter_rules": True,
            "tree_translation": True,
        },
    ),
]


BACKGROUND_COLOR = LIGHT_GRAY
PADDING_COLOR    = GRAY
SYMBOL_STEP      = 7
NUM_VARIANTS     = 7
MOVES_PER_LEVEL  = 1

VICTORY_COLOR_SEQUENCE = [BLACK, RED, GREEN, PURPLE, MAGENTA, BLUE, ORANGE, WHITE]



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
    "rule_panel": "Rule Panel",
    "symbol":     "Symbol",
    "selector":   "Selector",
    "highlight":  "Highlight",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "rule_panel": "",
    "symbol":     "",
    "selector":   "",
    "highlight":  "",
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


    _analyses = [
        {
            "gimmicks":     ["Basic rewriting rules with single-symbol patterns"],
            "gimmicks_l2":  [''                                                            ],
            "subgoals":     ["Move the selector to the output symbol and cycle variants to match the rule's right-hand side"],
            "subgoals_l2":  [''                                                                                                   ],
            "goal_flow":    "Identify which input symbols match a rule's LHS, then adjust output symbols to match the RHS.",
            "goal_flow_l2": ''                                                                                                                        ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Multiple rules with multi-symbol patterns"],
            "gimmicks_l2":  [''                                                ],
            "subgoals":     ["Match all rules simultaneously by cycling output symbols carefully"],
            "subgoals_l2":  [''                                                                                    ],
            "goal_flow":    "Apply multiple rewriting rules to transform the output sequence to match all rule RHS patterns.",
            "goal_flow_l2": ''                                                                                                            ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Larger symbol sets with 3 symbol types"],
            "gimmicks_l2":  [''                                                                      ],
            "subgoals":     ["Navigate selector across long output sequences and match all patterns"],
            "subgoals_l2":  [''                                                                                          ],
            "goal_flow":    "Use the selector to cycle each output symbol to match its corresponding rule RHS.",
            "goal_flow_l2": ''                                                                                                   ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Double translation: rules can chain"],
            "gimmicks_l2":  [''                                          ],
            "subgoals":     ["Find chained rule applications that transform the input sequence to the output"],
            "subgoals_l2":  [''                                                                     ],
            "goal_flow":    "Apply rules in sequence, using the output of one rule as input to another, to reach the target output.",
            "goal_flow_l2": ''                                                                                                                                             ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Alter rules: both LHS and RHS of rules are modifiable"],
            "gimmicks_l2":  [''                                                                     ],
            "subgoals":     ["Modify both the rule patterns and the output symbols to achieve a match"],
            "subgoals_l2":  [''                                                                                          ],
            "goal_flow":    "Cycle symbols in both rule sets and output sequences to satisfy all rewriting constraints.",
            "goal_flow_l2": ''                                                                                                            ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Double translation + alter rules + tree translation combined"],
            "gimmicks_l2":  [''                                                                  ],
            "subgoals":     ["Apply tree-structured rule applications with chaining and modifiable rules"],
            "subgoals_l2":  [''                                                                        ],
            "goal_flow":    "Solve the complex combination of tree-structured, chained, and modifiable rewriting rules.",
            "goal_flow_l2": ''                                                                                          ,
            "difficulty":   "very_hard",
        },
    ]

    analysis = _analyses[level_index] if level_index < len(_analyses) else {}

    return {
        "level_index": level_index + 1,
        "grid_size":   list(grid),
        "sprites":     level_sprites_info,
        "level_data":  {
            "double_translation": bool(getattr(level, "_data", {}) and level._data.get("double_translation", False)),
            "alter_rules":        bool(getattr(level, "_data", {}) and level._data.get("alter_rules", False)),
            "tree_translation":   bool(getattr(level, "_data", {}) and level._data.get("tree_translation", False)),
        },
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
        "game_id":          "tr87",
        "title":            "Translation Rewrite 87",
        "title_l2":         ''                    ,
        "description":      "A symbol rewriting puzzle game where you match input symbol sequences to rewriting rule left-hand sides, then adjust output symbols to match rule right-hand sides.",
        "description_l2":   ''                                                                                                                                                                        ,
        "default_grid_size": [64, 64],
        "num_levels":        len(levels),

        "goal":             "Adjust output symbols so that the input→output sequence matches all rewriting rules before running out of moves.",
        "goal_l2":          ''                                                                                                                                          ,
        "win_condition":    "check_win() returns True when the input sequence matches a rule LHS and the output sequence matches the corresponding rule RHS for every rule pair.",
        "win_condition_l2": ''                                                                                                                                                               ,
        "lose_condition":   "Moves budget reaches 0 (no moves remaining).",
        "lose_condition_l2":''                                                             ,

        "actions": {
            "ACTION1": {
                "key":             "←",
                "effect":          "Move the selector left (decrease selected output symbol index).",
                "effect_l2":       ''                                                                                                ,
                "restrictions":    "Costs 1 move from budget.",
                "restrictions_l2": ''                                  ,
                "step_cost":       1,
            },
            "ACTION2": {
                "key":             "→",
                "effect":          "Move the selector right (increase selected output symbol index).",
                "effect_l2":       ''                                                                                                ,
                "restrictions":    "Costs 1 move from budget.",
                "restrictions_l2": ''                                  ,
                "step_cost":       1,
            },
            "ACTION3": {
                "key":             "↑",
                "effect":          "Cycle the currently selected symbol to the previous variant (decrements variant index mod NUM_VARIANTS).",
                "effect_l2":       ''                                                                                                                                          ,
                "restrictions":    "Costs 1 move from budget.",
                "restrictions_l2": ''                                  ,
                "step_cost":       1,
            },
            "ACTION4": {
                "key":             "↓",
                "effect":          "Cycle the currently selected symbol to the next variant (increments variant index mod NUM_VARIANTS).",
                "effect_l2":       ''                                                                                                                                          ,
                "restrictions":    "Costs 1 move from budget.",
                "restrictions_l2": ''                                  ,
                "step_cost":       1,
            },
        },

        "sprite_roles": {
            "rule_panel":     "Colored background panel indicating whether a symbol is on the rule LHS (blue) or RHS (pink/yellow).",
            "rule_panel_l2":  ''                                                                                                          ,
            "symbol":         "A 5×5 pixel pattern representing one symbol in the rewriting alphabet. Multiple variants exist for each set (A/B/C).",
            "symbol_l2":      ''                                                                                                                                                    ,
            "selector":       "Arrow indicators above and below the currently selected output symbol (target of ACTION3/ACTION4).",
            "selector_l2":    ''                                                                                                                                    ,
            "highlight":      "White highlight background shown during win animation over matched rule panels.",
            "highlight_l2":   ''                                                                                                                  ,
        },

        "mechanics": {
            "symbol_variants":        "Each symbol position holds one of 7 variants (numbered 1-7). Cycling forward/backward wraps around.",
            "symbol_variants_l2":     ''                                                                                                                         ,
            "rule_matching":          "A rule matches when the input sequence at the current position has the same symbol names as the rule LHS sprites.",
            "rule_matching_l2":       ''                                                                                                                                 ,
            "output_matching":        "Win requires the output sequence to match the rule RHS for every matched input position.",
            "output_matching_l2":     ''                                                                                                            ,
            "selector_navigation":    "The selector highlights the active output symbol slot. ACTION1/ACTION2 move it left/right.",
            "selector_navigation_l2": ''                                                                                                                              ,
            "moves_budget":           "Each level has a limited number of moves. Every action costs 1 move. Running out causes game over.",
            "moves_budget_l2":        ''                                                                                                                   ,
            "victory_animation":      "When all rules match, a color cycling animation plays over the matched sprites, then the next level loads.",
            "victory_animation_l2":   ''                                                                                                                                             ,
            "alter_rules_mode":       "In alter_rules levels, both LHS and RHS symbol sets can be modified by the player.",
            "alter_rules_mode_l2":    ''                                                                                                        ,
            "double_translation_mode":"In double_translation levels, rule applications can chain (output of one becomes input of next).",
            "double_translation_mode_l2":''                                                                                                                     ,
            "tree_translation_mode":  "In tree_translation levels, rule applications form a tree structure where each RHS symbol may itself be a LHS.",
            "tree_translation_mode_l2":''                                                                                                                                        ,
        },
    },


    "levels": [
        _build_level_knowledge(i, lv)
        for i, lv in enumerate(levels)
    ],
}




class MovesProgressBar(RenderableUserDisplay):
    """





    """

    def __init__(self, game: "Tr87") -> None:
        self.game = game

    def render_interface(self, canvas: np.ndarray) -> np.ndarray:
        """



        """
        canvas_height = canvas.shape[0]
        ratio = self.game.upmkivwyrxz / self.game.vfpimnmtnta
        fill_width = math.ceil(canvas_height * ratio)

        canvas[canvas.shape[0] - 1, :] = DARK_GRAY
        canvas[canvas.shape[0] - 1, 0:fill_width] = OFF_WHITE
        return canvas




class Tr87(ARCBaseGame):
    """














    """



    def __init__(self) -> None:
        """"""
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[MovesProgressBar(self)],
        )








        self._log_state = ActionLogState(aggregator=default_aggregator, post_process=self._compose_summary)


        self._frame_sprite_log:      list[list[dict]] = []
        self._frame_nonsprite_log:   list[list[dict]] = []

        super().__init__(game_id="tr87", levels=levels, camera=camera, available_actions=[1, 2, 3, 4])

    def on_set_level(self, level: Level) -> None:
        """











        """

        self._levels[self._current_level_index] = self._clean_levels[self._current_level_index].clone()


        init_seeds = [4, 2, 1, 2, 12, 20]
        rng = random.Random(init_seeds[self._current_level_index])


        background_y = self.current_level.get_sprites_by_name("background")[0].y


        self.all_symbols = self.current_level.get_sprites_by_tag("symbol")
        self.all_symbols.sort(key=lambda s: (s.y, s.x))


        for sym in self.all_symbols:
            sym.set_blocking(BlockingMode.BOUNDING_BOX)
            sym.set_rotation(rng.choice([0, 90, 180, 270]))


        output_candidates = [s for s in self.all_symbols if s.y > background_y]
        self.output_symbols: list[Sprite] = []
        self.input_symbols_lower: list[Sprite] = []
        min_output_y = min(output_candidates, key=lambda s: s.y).y
        for sym in output_candidates:
            (self.output_symbols if sym.y == min_output_y else self.input_symbols_lower).append(sym)


        self.rule_pairs: list[tuple[list[Sprite], list[Sprite]]] = []
        for divider in sorted(
            self.current_level.get_sprites_by_name("rule_divider"),
            key=lambda s: (s.y, s.x),
        ):

            lhs_head = self.current_level.get_sprite_at(divider.x, divider.y, "symbol")
            assert lhs_head
            lhs = [lhs_head]

            while prev := self._get_adjacent_symbol(lhs[0], -1):
                lhs.insert(0, prev)
            rhs_head = self.current_level.get_sprite_at(divider.x + divider.width, divider.y, "symbol")
            assert rhs_head
            rhs = [rhs_head]

            while nxt := self._get_adjacent_symbol(rhs[-1], 1):
                rhs.append(nxt)
            self.rule_pairs.append((lhs, rhs))


        variant_seeds = [7, 7, 32, 18, 23, 11]
        rng.seed(variant_seeds[self._current_level_index])

        if self.current_level.get_data("alter_rules"):

            for lhs, rhs in self.rule_pairs:
                for symbol_list in (lhs, rhs):
                    steps = rng.randint(0, 6)
                    for i, sym in enumerate(symbol_list):
                        for _ in range(steps):
                            sym = self._cycle_symbol_variant(sym, 1)
                        symbol_list[i] = sym
        else:

            for i, rhs_sym in enumerate(self.input_symbols_lower):
                for _ in range(rng.randint(0, 6)):
                    rhs_sym = self._cycle_symbol_variant(rhs_sym, 1)
                self.input_symbols_lower[i] = rhs_sym


        for marker in self.current_level.get_sprites_by_tag("marker"):
            marker.set_visible(False)


        self.selector_index = 0
        self.selector_parts = [
            sprites["selector_top1"].clone(),
            sprites["selector_top1"].clone().set_rotation(180),
        ]
        for part in self.selector_parts:
            self.current_level.add_sprite(part)
        self._update_selector_position()


        self.vfpimnmtnta = 128 if self._current_level_index <= 4 else 256
        self.upmkivwyrxz = self.vfpimnmtnta


        self.victory_step = -1
        self.matched_lhs_rhs: list[tuple[list[Sprite], list[Sprite]]] = []
        self.highlight_sprites: list[list[Sprite]] = []
        self.pending_highlights: list[Sprite] = []

        self._emit_initial_layout()


    def _emit_initial_layout(self) -> None:
        """












        """
        _layout_rows: list[dict] = []

        for _spr in self.current_level.get_sprites():
            if not _spr.is_visible:
                continue
            _tags = list(_spr.tags) if _spr.tags else []
            if "symbol" in _tags:
                _role, _role_l2 = "symbol", ""
            elif _spr.name.startswith("rule_panel"):
                _role, _role_l2 = "rule_panel", ""
            elif _spr.name.startswith("selector"):
                _role, _role_l2 = "selector", ""
            else:
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
            if _ns.get("name") != "moves_budget":
                continue
            _layout_rows.append({
                "role":    "remaining_step",
                "role_l2": ''                  ,
                "sname":   "moves_budget",
                "dx":      int(_ns.get("x", 0)),
                "dy":      int(_ns.get("y", 0)),
                "dw":      int(_ns.get("width", 0)),
                "dh":      int(_ns.get("height", 0)),
            })

        if _layout_rows:
            self._set_event("initial_layout", _layout_rows)

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

    def _bar_pixel_len(self, remaining: int) -> int:
        """


        """
        total = self.vfpimnmtnta
        if total == 0:
            return 0
        r = remaining
        if r < 0:
            r = 0
        if r > total:
            r = total

        return math.ceil(64 * r / total)

    def step(self) -> None:
        """










        """

        self.action_log    = []
        self.action_log_l2 = []


        _before_sprites = {
            id(spr): {
                "name":     spr.name,
                "x":        spr.x,
                "y":        spr.y,
                "width":    spr.width,
                "height":   spr.height,
                "rotation": spr.rotation,
                "visible":  spr.is_visible,
            }
            for spr in self.current_level.get_sprites()
        }
        _before_moves = self.upmkivwyrxz


        if self.victory_step >= 0:

            color_step_size = len(VICTORY_COLOR_SEQUENCE) - 1
            rule_index = self.victory_step // color_step_size
            lhs_sprites, rhs_sprites = self.matched_lhs_rhs[rule_index]
            color_from = VICTORY_COLOR_SEQUENCE[self.victory_step % color_step_size]
            color_to   = VICTORY_COLOR_SEQUENCE[self.victory_step % color_step_size + 1]


            _from_lbl    = COLOR_PALETTE.get(int(color_from), {}).get("label",    str(color_from)).lower()
            _to_lbl      = COLOR_PALETTE.get(int(color_to),   {}).get("label",    str(color_to)).lower()
            _from_lbl_l2 = COLOR_PALETTE.get(int(color_from), {}).get("label_l2", str(color_from))
            _to_lbl_l2   = COLOR_PALETTE.get(int(color_to),   {}).get("label_l2", str(color_to))


            for sym in lhs_sprites + rhs_sprites:
                _before_px = sym.pixels.tobytes()
                sym.color_remap(color_from, color_to)
                if not sym.is_visible:
                    continue
                if sym.pixels.tobytes() == _before_px:
                    continue
                _sdx, _sdy, _sdw, _sdh = self._to_display_bbox(sym)
                self.action_log.append(
                    f"[change]:[{sym.name}]:[sname({sym.name});color({_from_lbl})_({_to_lbl});position({_sdx}_{_sdy});bbox({_sdw}_{_sdh})]:{sym.name} at ({_sdx},{_sdy}) changes color from {_from_lbl} to {_to_lbl}.|Object at ({_sdx},{_sdy}) changes color from {_from_lbl} to {_to_lbl}."
                )
                self.action_log_l2.append(
                    f"[change]:[{sym.name}]:[sname({sym.name});color({_from_lbl_l2})_({_to_lbl_l2});position({_sdx}_{_sdy});bbox({_sdw}_{_sdh})]:({_sdx},{_sdy}){sym.name}{_from_lbl_l2}{_to_lbl_l2}|({_sdx},{_sdy})Object{_from_lbl_l2}{_to_lbl_l2}"
                )

            if self.victory_step % color_step_size == 1:



                if self.pending_highlights:
                    _prev_idx = rule_index - 1
                    for _hl in self.pending_highlights:
                        _hdx, _hdy, _hdw, _hdh = self._to_display_bbox(_hl)



                        self.action_log.append(
                            f"[bright]:[rule_highlight]:[state(off);index({_prev_idx});sname({_hl.name});position({_hdx}_{_hdy});bbox({_hdw}_{_hdh})]:The rule highlight at ({_hdx},{_hdy}) for matched rule {_prev_idx} turns off.|The Object at ({_hdx},{_hdy}) turns off."
                        )
                        self.action_log_l2.append(
                            f"[bright]:[rule_highlight]:[state(off);index({_prev_idx});sname({_hl.name});position({_hdx}_{_hdy});bbox({_hdw}_{_hdh})]:({_hdx},{_hdy}){_prev_idx}|({_hdx},{_hdy})Object"
                        )
                for hlsp in self.pending_highlights:
                    self.current_level.remove_sprite(hlsp)
                self.pending_highlights = []
                for sym in self.highlight_sprites[rule_index]:
                    hl = sprites["rule_highlight_bg"].clone().set_position(sym.x - 2, sym.y - 2)
                    self.current_level.add_sprite(hl)
                    self.pending_highlights.append(hl)






                _rule_hl_info = []
                for _hl in self.pending_highlights:
                    _hdx, _hdy, _hdw, _hdh = self._to_display_bbox(_hl)
                    self.action_log.append(
                        f"[bright]:[rule_highlight]:[state(on);index({rule_index});sname({_hl.name});position({_hdx}_{_hdy});bbox({_hdw}_{_hdh})]:The rule highlight at ({_hdx},{_hdy}) for matched rule {rule_index} lights up.|The Object at ({_hdx},{_hdy}) lights up."
                    )
                    self.action_log_l2.append(
                        f"[bright]:[rule_highlight]:[state(on);index({rule_index});sname({_hl.name});position({_hdx}_{_hdy});bbox({_hdw}_{_hdh})]:({_hdx},{_hdy}){rule_index}|({_hdx},{_hdy})Object"
                    )
                    _rule_hl_info.append({
                        "name": _hl.name, "dx": _hdx, "dy": _hdy, "dw": _hdw, "dh": _hdh,
                    })


                _matches = self._log_state.get_event("rule_matches", []) or []
                _matches.append({"index": rule_index, "highlights": _rule_hl_info})
                self._set_event("rule_matches", _matches)
            self.victory_step += 1

            if self.victory_step == len(self.matched_lhs_rhs) * color_step_size:


                if self.pending_highlights:
                    for _hl in self.pending_highlights:
                        _hdx, _hdy, _hdw, _hdh = self._to_display_bbox(_hl)
                        self.action_log.append(
                            f"[bright]:[rule_highlight]:[state(off);index({rule_index});sname({_hl.name});position({_hdx}_{_hdy});bbox({_hdw}_{_hdh})]:The rule highlight at ({_hdx},{_hdy}) for matched rule {rule_index} turns off.|The Object at ({_hdx},{_hdy}) turns off."
                        )
                        self.action_log_l2.append(
                            f"[bright]:[rule_highlight]:[state(off);index({rule_index});sname({_hl.name});position({_hdx}_{_hdy});bbox({_hdw}_{_hdh})]:({_hdx},{_hdy}){rule_index}|({_hdx},{_hdy})Object"
                        )
                self.next_level()
                self._post_step(_before_sprites, _before_moves)
                self.complete_action()
                return
            self._post_step(_before_sprites, _before_moves)
            return


        elif self.action.id in [GameAction.ACTION3, GameAction.ACTION4]:
            direction = -1 if self.action.id == GameAction.ACTION3 else 1
            if self.current_level.get_data("alter_rules"):


                rule_sides = [sym_list for rule in self.rule_pairs for sym_list in rule]
                self.selector_index = (self.selector_index + direction) % len(rule_sides)
            else:

                self.selector_index = (self.selector_index + direction) % len(self.input_symbols_lower)
            self.upmkivwyrxz -= 1
            self._update_selector_position()


        elif self.action.id in [GameAction.ACTION1, GameAction.ACTION2]:
            direction = -1 if self.action.id == GameAction.ACTION1 else 1
            if self.current_level.get_data("alter_rules"):

                rule_sides = [sym_list for rule in self.rule_pairs for sym_list in rule]
                target_side = rule_sides[self.selector_index]
                for i in range(len(target_side)):
                    target_side[i] = self._cycle_symbol_variant(target_side[i], direction)
            else:

                target = self.input_symbols_lower[self.selector_index]
                self.input_symbols_lower[self.selector_index] = self._cycle_symbol_variant(target, direction)
            self.upmkivwyrxz -= 1

            if self._check_win():

                for part in self.selector_parts:
                    part.set_visible(False)
                self.victory_step = 0
                self._post_step(_before_sprites, _before_moves)
                return


        if self.upmkivwyrxz == 0:
            self.lose()
        self.complete_action()


        self._post_step(_before_sprites, _before_moves)



    _BRIGHT_SUPPRESS: frozenset = frozenset({"rule_highlight_bg"})

    def _post_step(self, before_sprites: dict, before_moves: int) -> None:
        """



        """
        after_sprites = {
            id(spr): spr for spr in self.current_level.get_sprites()
        }







        _SHAPE_PREFIX = "symbol_set"
        _shape_change_old_ids: set[int] = set()
        _shape_change_new_ids: set[int] = set()
        _gone_shape: list[tuple[int, dict, str]] = []
        for spr_id, b in before_sprites.items():
            if not b["name"].startswith(_SHAPE_PREFIX) or not b["visible"]:
                continue
            spr = after_sprites.get(spr_id)
            if spr is None or not spr.is_visible:
                _gone_shape.append((spr_id, b, b["name"][len(_SHAPE_PREFIX):]))
        _appear_shape: list[tuple[int, "Sprite", str]] = []
        for spr_id, spr in after_sprites.items():
            if not spr.name.startswith(_SHAPE_PREFIX) or not spr.is_visible:
                continue
            b = before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                _appear_shape.append((spr_id, spr, spr.name[len(_SHAPE_PREFIX):]))
        for g_id, g_b, g_suffix in _gone_shape:
            for a_id, a_spr, a_suffix in _appear_shape:
                if a_id in _shape_change_new_ids:
                    continue
                if (g_b["x"], g_b["y"]) == (a_spr.x, a_spr.y) and g_suffix != a_suffix:
                    adx, ady, adw, adh = self._to_display_bbox(a_spr)
                    self.action_log.append(
                        f"[change]:[{a_spr.name}]:[sname({a_spr.name});shape({g_suffix})_({a_suffix});position({adx}_{ady});bbox({adw}_{adh})]:Symbol at ({adx},{ady}) changes shape from {g_suffix} to {a_suffix}.|Object at ({adx},{ady}) changes shape from {g_suffix} to {a_suffix}."
                    )
                    self.action_log_l2.append(
                        f"[change]:[{a_spr.name}]:[sname({a_spr.name});shape({g_suffix})_({a_suffix});position({adx}_{ady});bbox({adw}_{adh})]:({adx},{ady}){g_suffix}{a_suffix}|({adx},{ady})Object{g_suffix}{a_suffix}"
                    )
                    _shape_change_old_ids.add(g_id)
                    _shape_change_new_ids.add(a_id)
                    break


        for spr_id, spr in after_sprites.items():
            if spr.name in self._BRIGHT_SUPPRESS:
                continue
            if spr_id in _shape_change_new_ids:
                continue
            b = before_sprites.get(spr_id)
            if b is None or (not b["visible"] and spr.is_visible):
                if not spr.is_visible:
                    continue
                dx, dy, dw, dh = self._to_display_bbox(spr)
                self.action_log.append(
                    f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name} appears at ({dx},{dy}).|Object appears at ({dx},{dy})."
                )
                self.action_log_l2.append(
                    f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name}({dx},{dy})|Object({dx},{dy})"
                )



        _arc_scale_d = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in before_sprites.items():
            if b["name"] in self._BRIGHT_SUPPRESS:
                continue
            if spr_id in _shape_change_old_ids:
                continue
            spr = after_sprites.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = int(b.get("width", 0))  * _arc_scale_d
                bdh = int(b.get("height", 0)) * _arc_scale_d
                self.action_log.append(
                    f"[disappear]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy})."
                )
                self.action_log_l2.append(
                    f"[disappear]:[{b['name']}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']}({bdx},{bdy})|Object({bdx},{bdy})"
                )





        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            if spr.name in self._BRIGHT_SUPPRESS:
                continue
            b = before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                continue
            if (spr.x, spr.y) != (b["x"], b["y"]):
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                adx, ady, adw, adh = self._to_display_bbox(spr)
                self.action_log.append(
                    f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady})."
                )
                self.action_log_l2.append(
                    f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})"
                )
            if spr.rotation != b["rotation"]:
                _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
                self.action_log.append(
                    f"[rotate]:[{spr.name}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:{spr.name} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°."
                )
                self.action_log_l2.append(
                    f"[rotate]:[{spr.name}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:({_rdx},{_rdy}){spr.name}{b['rotation']}°{spr.rotation}°|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°"
                )


        after_moves = self.upmkivwyrxz
        if after_moves != before_moves and self._bar_pixel_len(before_moves) != self._bar_pixel_len(after_moves):
            if after_moves < before_moves:
                self.action_log.append(
                    f"[change]:[moves]:[sname(moves_budget);count({before_moves})_({after_moves})]:The off-white moves progress bar at the bottom shrinks from the right ({before_moves}→{after_moves} moves remain).|The off-white Object at the bottom shrinks from the right ({before_moves}→{after_moves})."
                )
                self.action_log_l2.append(
                    f"[change]:[moves]:[sname(moves_budget);count({before_moves})_({after_moves})]:{before_moves}→{after_moves}|Object{before_moves}→{after_moves}"
                )
            else:
                self.action_log.append(
                    f"[change]:[moves]:[sname(moves_budget);count({before_moves})_({after_moves})]:The off-white moves progress bar at the bottom extends to the right ({before_moves}→{after_moves} moves).|The off-white Object at the bottom extends to the right ({before_moves}→{after_moves})."
                )
                self.action_log_l2.append(
                    f"[change]:[moves]:[sname(moves_budget);count({before_moves})_({after_moves})]:{before_moves}→{after_moves}|Object{before_moves}→{after_moves}"
                )



        state = getattr(self, "_state", None)
        state_name = getattr(state, "name", None) or str(state) if state is not None else ""
        if state_name == "WIN" or state_name.endswith(".WIN"):
            self._set_event("win", {
                "en": "[win]:[game]:[]:The level is cleared.|The level is cleared.",
                "l2": "[win]:[game]:[]:|",
            })
        elif state_name == "GAME_OVER" or state_name.endswith(".GAME_OVER"):
            self._set_event("gameover", {
                "en": "[gameover]:[game]:[]:No moves remaining, game over.|The Object is exhausted, game over.",
                "l2": "[gameover]:[game]:[]:0|Object",
            })


        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()

    def _cycle_symbol_variant(self, sym: Sprite, direction: int) -> Sprite:
        """













        """

        new_variant_name = sym.name[:-1] + str(
            (int(sym.name[-1]) + direction - 1) % NUM_VARIANTS + 1
        )

        new_sym = (
            sprites[new_variant_name]
            .clone()
            .set_position(sym.x, sym.y)
            .set_rotation(sym.rotation)
        )
        self.current_level.remove_sprite(sym)
        self.current_level.add_sprite(new_sym.set_blocking(BlockingMode.BOUNDING_BOX))
        return new_sym

    def _update_selector_position(self) -> None:
        """



        """
        target_sym: Sprite = Sprite([[]])
        if self.current_level.get_data("alter_rules"):


            rule_sides = [sym_list for rule in self.rule_pairs for sym_list in rule]
            target_side = rule_sides[self.selector_index]
            target_sym = target_side[0]
            for part in self.selector_parts:
                part.pixels = sprites[f"selector_top{len(target_side)}"].pixels
        else:
            target_sym = self.input_symbols_lower[self.selector_index]
        y_offset = 4
        self.selector_parts[0].set_position(target_sym.x, target_sym.y - y_offset)
        self.selector_parts[1].set_position(
            target_sym.x, target_sym.y + target_sym.height - 2 + y_offset
        )

    def _check_win(self) -> bool:
        """








        """
        def _try_double_translation(
            lhs: list[Sprite], rhs: list[Sprite]
        ) -> tuple[list[Sprite], list[Sprite], bool]:
            """





            """
            marker = self.current_level.get_sprite_at(lhs[0].x, lhs[0].y, "marker")
            if not marker:
                return (lhs, rhs, False)
            if marker.name.endswith("2"):
                return ([], [], True)

            marker2_name = marker.name.replace("1", "2")
            marker2 = self.current_level.get_sprites_by_name(marker2_name)[0]
            sym_at_marker2 = self.current_level.get_sprite_at(marker2.x, marker2.y, "symbol")

            matching_rule = next(
                (rule for rule in self.rule_pairs if sym_at_marker2 == rule[0][0]), None
            )
            if not matching_rule:
                return (lhs, rhs, False)
            return (lhs + matching_rule[0], rhs + matching_rule[1], False)

        input_pos = 0
        output_pos = 0
        self.matched_lhs_rhs = []
        self.highlight_sprites = []

        while input_pos < len(self.output_symbols):
            for lhs, rhs in self.rule_pairs:
                if not self._sequence_matches(self.output_symbols, input_pos, lhs):
                    continue

                extra_highlights: list[Sprite] = []
                if self.current_level.get_data("tree_translation"):

                    resolved_rhs: list[Sprite] = []
                    failed = False
                    for rhs_sym in rhs:
                        for sub_lhs, sub_rhs in self.rule_pairs:
                            if sub_lhs[0].name == rhs_sym.name:
                                resolved_rhs += sub_rhs
                                extra_highlights += [rhs_sym] + sub_lhs
                                break
                        else:
                            failed = True
                            break
                    if failed:
                        continue
                    rhs = resolved_rhs
                elif self.current_level.get_data("double_translation"):
                    lhs, rhs, skip = _try_double_translation(lhs, rhs)
                    if skip:
                        continue

                    for sub_lhs, sub_rhs in self.rule_pairs:
                        sub_lhs, sub_rhs, skip = _try_double_translation(sub_lhs, sub_rhs)
                        if skip:
                            continue
                        if all(a.name == b.name for a, b in zip(rhs, sub_lhs)):
                            extra_highlights = list(rhs) + list(sub_lhs)
                            rhs = sub_rhs
                            break
                    else:
                        continue


                if not self._sequence_matches(self.input_symbols_lower, output_pos, rhs):
                    return False


                matched_l = self.output_symbols[input_pos: input_pos + len(lhs)]
                matched_r = self.input_symbols_lower[output_pos: output_pos + len(rhs)]
                self.matched_lhs_rhs.append((matched_l, matched_r))
                self.highlight_sprites.append(lhs + rhs + extra_highlights)
                input_pos  += len(lhs)
                output_pos += len(rhs)
                break
            else:
                return False

        return True

    def _sequence_matches(
        self, symbol_list: list[Sprite], start: int, pattern: list[Sprite]
    ) -> bool:
        """










        """
        if start + len(pattern) > len(symbol_list):
            return False
        return all(
            symbol_list[start + i].name == pattern[i].name
            for i in range(len(pattern))
        )

    def _get_adjacent_symbol(self, sym: Sprite, direction: int) -> Sprite | None:
        """










        """
        neighbor = None
        for s in self.all_symbols:
            if s.x == sym.x + direction * SYMBOL_STEP and s.y == sym.y:
                neighbor = s
                break
        return neighbor

    def _get_hidden_state(self) -> np.ndarray:
        """"""
        return np.array([self.upmkivwyrxz])

    def _set_event(self, key: str, value) -> None:
        """
        """
        self._log_state.set_event(key, value)

    def _compose_summary(
        self,
        agg_en: list[str],
        agg_l2: list[str],
        events: dict,
    ) -> tuple[list[str], list[str]]:
        """






        """

        kept_en: list[str] = []
        kept_l2: list[str] = []
        for line_en, line_l2 in zip(agg_en, agg_l2):
            if line_en.startswith("[bright]:[rule_highlight]:"):
                continue
            kept_en.append(line_en)
            kept_l2.append(line_l2)


        rule_matches = events.get("rule_matches") or []
        for entry in rule_matches:
            idx = entry["index"]
            for h in entry.get("highlights", []):
                kept_en.append(
                    f"[bright]:[rule_highlight]:[state(flash);index({idx});sname({h['name']});"
                    f"position({h['dx']}_{h['dy']});bbox({h['dw']}_{h['dh']})]:"
                    f"The rule highlight at ({h['dx']},{h['dy']}) flashes over matched rule {idx} to confirm the rule match.|The Object at ({h['dx']},{h['dy']}) flashes."
                )
                kept_l2.append(
                    f"[bright]:[rule_highlight]:[state(flash);index({idx});sname({h['name']});"
                    f"position({h['dx']}_{h['dy']});bbox({h['dw']}_{h['dh']})]:"
                    f"({h['dx']},{h['dy']}){idx}|({h['dx']},{h['dy']})Object"
                )


        return compose_default_summary(kept_en, kept_l2, events)


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
            "name":     "moves_budget",
            "label":    "Moves Remaining",
            "label_l2": ''            ,
            "value":    self.upmkivwyrxz,
            "max":      self.vfpimnmtnta,
            "x":        0, "y":      0,
            "width":    64, "height": 1,
        })
        return result

    def get_game_knowledge(self) -> dict:
        """


        """
        return GAME_KNOWLEDGE
