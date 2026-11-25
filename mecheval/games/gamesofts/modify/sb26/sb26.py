"""












































"""

import math
import re

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

from gamesofts._action_log import ActionLogState, default_aggregator, compose_default_summary











TRANSPARENT = -1
BLACK       =  0
BLUE        =  1
RED         =  2
GREEN       =  3
YELLOW      =  4
GRAY        =  5
FUCHSIA     =  6
ORANGE      =  7
AZURE       =  8
MAROON      =  9
PURPLE      = 10
SCARLET     = 11
LIME        = 12
BROWN       = 13
TEAL        = 14
NAVY        = 15


TAG_DISPLAY_NAMES = {
    "color_tile":   "Color Tile",
    "slot_button":  "Slot Button",
    "frame":        "Frame",
    "sys_click":    "Clickable",
}

TAG_DISPLAY_NAMES_L2 = {
    "color_tile":   "",
    "slot_button":  "",
    "frame":        "",
    "sys_click":    "",
}



COLOR_PALETTE = {
    TRANSPARENT: "transparent",
    0:  "white",
    1:  "off_white",
    2:  "light_gray",
    3:  "gray",
    4:  "dark_gray",
    5:  "black",
    6:  "magenta",
    7:  "pink",
    8:  "red",
    9:  "blue",
    10: "light_blue",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}

COLOR_PALETTE_L2 = {
    TRANSPARENT: "",
    0:  "",
    1:  "",
    2:  "",
    3:  "",
    4:  "",
    5:  "",
    6:  "",
    7:  "",
    8:  "",
    9:  "",
    10: "",
    11: "",
    12: "",
    13: "",
    14: "",
    15: "",
}


GAME_KNOWLEDGE = {
    "title": "SB26 - Subroutine Box Puzzle",
    "description": (
        "Color tile sorting puzzle with nested frame references. "
        "Place color_tiles into slots to match the target_box sequence. "
        "Nested tiles (nested_tile) recurse into sub-frames."
    ),
    "goal": "Match all target_box colors by placing color_tiles in the correct slot order.",
    "actions": {
        "ACTION5": "Confirm / Skip (costs 1 energy)",
        "ACTION6": "Click a color_tile or slot to select/move",
        "ACTION7": "Undo last move",
    },
    "energy": {
        "description": "Displayed as a bar at y=53. Starts at 64. Decreases on each swap or skip.",
        "initial": 64,
        "game_over_condition": "energy reaches 0",
    },
    "sprites": {
        "color_tile":     "A 6x6 single-color tile the player rearranges.",
        "target_box":     "A 6x6 colored box showing the required color at each position.",
        "slot_button":    "A 6x6 slot where a color_tile can be placed.",
        "nested_tile":    "A 6x6 tile that references another frame (recursive).",
        "frame_Nslot":    "A frame holding N slots (N = 1..7).",
        "frame_cursor":   "Cursor frame highlighting the active frame.",
        "slot_cursor_spr":"Cursor highlighting the active slot.",
        "cursor_box":     "Highlight on the currently selected color_tile.",
        "bg_tile":        "8x8 background grid tile.",
        "bracket_box":    "Blue bracket indicating a nested frame reference.",
        "overlay_medium": "Overlay shown when wrong answer (medium size).",
        "overlay_wide":   "Overlay shown when wrong answer (wide size).",
    },
    "levels": [
        {"level": 1, "num_frames": 1, "num_targets": 4, "has_nested": False,
         "description": "4-color sequence, single frame"},
        {"level": 2, "num_frames": 2, "num_targets": 7, "has_nested": True,
         "description": "7-color sequence, two frames with nested reference"},
        {"level": 3, "num_frames": 2, "num_targets": 7, "has_nested": True,
         "description": "7-color sequence with two nested tiles"},
        {"level": 4, "num_frames": 2, "num_targets": 7, "has_nested": True,
         "description": "7-color sequence, mixed frame sizes"},
        {"level": 5, "num_frames": 2, "num_targets": 8, "has_nested": True,
         "description": "8-color sequence, wider board"},
        {"level": 6, "num_frames": 4, "num_targets": 8, "has_nested": True,
         "description": "8-color sequence, four frames"},
        {"level": 7, "num_frames": 3, "num_targets": 7, "has_nested": True,
         "description": "7-color sequence, three-level nesting"},
        {"level": 8, "num_frames": 2, "num_targets": 6, "has_nested": False,
         "description": "6-color sequence with two rows of targets"},
    ],
    "tips": [
        "Read target_box colors left-to-right, top row first.",
        "nested_tile (with inner ring) means the answer sequence recurses into a sub-frame.",
        "Use ACTION7 (undo) to recover from wrong placements before they cost energy.",
        "bracket_box (blue) marks which sub-frame you are currently inside.",
    ],
}

sprites = {
    "selected_highlight": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="selected_highlight",
        visible=True,
        collidable=True,
        layer=1,
    ),
    "frame_1slot": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="frame_1slot",
        visible=True,
        collidable=True,
        tags=["frame"],
    ),
    "frame_2slot": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="frame_2slot",
        visible=True,
        collidable=True,
        tags=["frame"],
    ),
    "overlay_wide": Sprite(
        pixels=[
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="overlay_wide",
        visible=True,
        collidable=True,
        layer=12,
    ),
    "color_tile": Sprite(
        pixels=[
            [-2, -2, -2, -2, -2, -2],
            [-2, 8, 8, 8, 8, -2],
            [-2, 8, 8, 8, 8, -2],
            [-2, 8, 8, 8, 8, -2],
            [-2, 8, 8, 8, 8, -2],
            [-2, -2, -2, -2, -2, -2],
        ],
        name="color_tile",
        visible=True,
        collidable=True,
        tags=["color_tile", "sys_click"],
        layer=1,
    ),
    "frame_cursor": Sprite(
        pixels=[
            [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        ],
        name="frame_cursor",
        visible=True,
        collidable=True,
        layer=10,
    ),
    "frame_5slot": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="frame_5slot",
        visible=True,
        collidable=True,
        tags=["frame"],
    ),
    "overlay_medium": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="overlay_medium",
        visible=True,
        collidable=True,
        layer=12,
    ),
    "frame_3slot": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="frame_3slot",
        visible=True,
        collidable=True,
        tags=["frame"],
    ),
    "divider_bar": Sprite(
        pixels=[
            [14, 14],
            [14, 14],
            [14, 14],
            [14, 14],
            [14, 14],
            [14, 14],
            [14, 14],
            [14, 14],
        ],
        name="divider_bar",
        visible=True,
        collidable=False,
        layer=1,
    ),
    "frame_4slot": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="frame_4slot",
        visible=True,
        collidable=True,
        tags=["frame"],
    ),
    "target_box": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8],
            [8, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, 8],
            [8, -1, -1, -1, -1, 8],
            [8, 8, 8, 8, 8, 8],
        ],
        name="target_box",
        visible=True,
        collidable=True,
        layer=1,
    ),
    "slot_button": Sprite(
        pixels=[
            [-2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2],
            [-2, -2, 2, 2, -2, -2],
            [-2, -2, 2, 2, -2, -2],
            [-2, -2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2, -2],
        ],
        name="slot_button",
        visible=True,
        collidable=True,
        tags=["slot_button", "sys_click"],
    ),
    "bracket_box": Sprite(
        pixels=[
            [3, 3, 3, 3, 3, 3],
            [3, -1, -1, -1, -1, 3],
            [3, -1, -1, -1, -1, 3],
            [3, -1, -1, -1, -1, 3],
            [3, -1, -1, -1, -1, 3],
            [3, 3, 3, 3, 3, 3],
        ],
        name="bracket_box",
        visible=True,
        collidable=True,
    ),
    "bg_tile": Sprite(
        pixels=[
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5, 5, 5, 5],
        ],
        name="bg_tile",
        visible=True,
        collidable=True,
    ),
    "nested_tile": Sprite(
        pixels=[
            [-2, -2, -2, -2, -2, -2],
            [-2, 8, 8, 8, 8, -2],
            [-2, 8, -2, -2, 8, -2],
            [-2, 8, -2, -2, 8, -2],
            [-2, 8, 8, 8, 8, -2],
            [-2, -2, -2, -2, -2, -2],
        ],
        name="nested_tile",
        visible=True,
        collidable=True,
        tags=["color_tile", "sys_click"],
        layer=1,
    ),
    "frame_6slot": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="frame_6slot",
        visible=True,
        collidable=True,
        tags=["frame"],
    ),
    "cursor_box": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        name="cursor_box",
        visible=True,
        collidable=True,
    ),
    "divider_line": Sprite(
        pixels=[
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ],
        name="divider_line",
        visible=True,
        collidable=True,
    ),
    "frame_7slot": Sprite(
        pixels=[
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ],
        name="frame_7slot",
        visible=True,
        collidable=True,
        tags=["frame"],
    ),
}
levels = [

    Level(
        sprites=[
            sprites["color_tile"].clone().set_position(17, 56).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(25, 56).color_remap(None, 15),
            sprites["color_tile"].clone().set_position(33, 56).color_remap(None, 9),
            sprites["color_tile"].clone().set_position(41, 56).color_remap(None, 11),
            sprites["frame_4slot"].clone().set_position(18, 25),
            sprites["target_box"].clone().set_position(18, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(25, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(32, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(39, 1).color_remap(None, 15),
            sprites["slot_button"].clone().set_position(20, 27),
            sprites["slot_button"].clone().set_position(26, 27),
            sprites["slot_button"].clone().set_position(32, 27),
            sprites["slot_button"].clone().set_position(38, 27),
            sprites["bg_tile"].clone().set_position(17, 0),
            sprites["bg_tile"].clone().set_position(24, 0),
            sprites["bg_tile"].clone().set_position(31, 0),
            sprites["bg_tile"].clone().set_position(38, 0),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["color_tile"].clone().set_position(8, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(15, 56).color_remap(None, 15),
            sprites["color_tile"].clone().set_position(22, 56).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(29, 56).color_remap(None, 12),
            sprites["color_tile"].clone().set_position(36, 56).color_remap(None, 6),
            sprites["color_tile"].clone().set_position(43, 56).color_remap(None, 9),
            sprites["color_tile"].clone().set_position(50, 56).color_remap(None, 11),
            sprites["divider_bar"].clone().set_position(34, 24),
            sprites["frame_4slot"].clone().set_position(18, 18),
            sprites["frame_4slot"].clone().set_position(18, 32).color_remap(None, 14),
            sprites["target_box"].clone().set_position(8, 1).color_remap(None, 12),
            sprites["target_box"].clone().set_position(15, 1).color_remap(None, 15),
            sprites["target_box"].clone().set_position(22, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(29, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(36, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(43, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(50, 1).color_remap(None, 6),
            sprites["slot_button"].clone().set_position(20, 20),
            sprites["slot_button"].clone().set_position(26, 20),
            sprites["slot_button"].clone().set_position(38, 20),
            sprites["slot_button"].clone().set_position(20, 34),
            sprites["slot_button"].clone().set_position(26, 34),
            sprites["slot_button"].clone().set_position(38, 34),
            sprites["slot_button"].clone().set_position(32, 34),
            sprites["bg_tile"].clone().set_position(7, 0),
            sprites["bg_tile"].clone().set_position(14, 0),
            sprites["bg_tile"].clone().set_position(21, 0),
            sprites["bg_tile"].clone().set_position(28, 0),
            sprites["bg_tile"].clone().set_position(35, 0),
            sprites["bg_tile"].clone().set_position(42, 0),
            sprites["bg_tile"].clone().set_position(49, 0),
            sprites["nested_tile"].clone().set_position(32, 20).color_remap(None, 14),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["frame_2slot"].clone().set_position(15, 31).color_remap(None, 14),
            sprites["frame_2slot"].clone().set_position(33, 31).color_remap(None, 9),
            sprites["color_tile"].clone().set_position(50, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(22, 56).color_remap(None, 15),
            sprites["color_tile"].clone().set_position(15, 56).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(8, 56).color_remap(None, 12),
            sprites["color_tile"].clone().set_position(43, 56).color_remap(None, 6),
            sprites["color_tile"].clone().set_position(36, 56).color_remap(None, 9),
            sprites["color_tile"].clone().set_position(29, 56).color_remap(None, 11),
            sprites["frame_5slot"].clone().set_position(15, 19),
            sprites["target_box"].clone().set_position(15, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(22, 1).color_remap(None, 15),
            sprites["target_box"].clone().set_position(29, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(36, 1).color_remap(None, 6),
            sprites["target_box"].clone().set_position(43, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(8, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(50, 1).color_remap(None, 12),
            sprites["slot_button"].clone().set_position(29, 21),
            sprites["slot_button"].clone().set_position(17, 33),
            sprites["slot_button"].clone().set_position(23, 33),
            sprites["slot_button"].clone().set_position(35, 33),
            sprites["slot_button"].clone().set_position(41, 33),
            sprites["slot_button"].clone().set_position(17, 21),
            sprites["slot_button"].clone().set_position(41, 21),
            sprites["bg_tile"].clone().set_position(14, 0),
            sprites["bg_tile"].clone().set_position(21, 0),
            sprites["bg_tile"].clone().set_position(28, 0),
            sprites["bg_tile"].clone().set_position(35, 0),
            sprites["bg_tile"].clone().set_position(42, 0),
            sprites["bg_tile"].clone().set_position(7, 0),
            sprites["bg_tile"].clone().set_position(49, 0),
            sprites["nested_tile"].clone().set_position(23, 21).color_remap(None, 14),
            sprites["nested_tile"].clone().set_position(35, 21).color_remap(None, 9),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["color_tile"].clone().set_position(29, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(15, 56).color_remap(None, 6),
            sprites["color_tile"].clone().set_position(36, 56).color_remap(None, 15),
            sprites["color_tile"].clone().set_position(22, 56).color_remap(None, 12),
            sprites["color_tile"].clone().set_position(8, 56).color_remap(None, 11),
            sprites["color_tile"].clone().set_position(23, 34).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(43, 56).color_remap(None, 9),
            sprites["frame_5slot"].clone().set_position(15, 18),
            sprites["frame_3slot"].clone().set_position(21, 32).color_remap(None, 14),
            sprites["target_box"].clone().set_position(8, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(15, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(22, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(29, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(36, 1).color_remap(None, 6),
            sprites["target_box"].clone().set_position(43, 1).color_remap(None, 12),
            sprites["target_box"].clone().set_position(50, 1).color_remap(None, 15),
            sprites["slot_button"].clone().set_position(17, 20),
            sprites["slot_button"].clone().set_position(23, 20),
            sprites["slot_button"].clone().set_position(35, 20),
            sprites["slot_button"].clone().set_position(41, 20),
            sprites["slot_button"].clone().set_position(29, 20),
            sprites["slot_button"].clone().set_position(29, 34),
            sprites["slot_button"].clone().set_position(35, 34),
            sprites["bg_tile"].clone().set_position(7, 0),
            sprites["bg_tile"].clone().set_position(14, 0),
            sprites["bg_tile"].clone().set_position(21, 0),
            sprites["bg_tile"].clone().set_position(28, 0),
            sprites["bg_tile"].clone().set_position(35, 0),
            sprites["bg_tile"].clone().set_position(42, 0),
            sprites["bg_tile"].clone().set_position(49, 0),
            sprites["nested_tile"].clone().set_position(50, 56).color_remap(None, 14),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["color_tile"].clone().set_position(39, 56).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(25, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(32, 56).color_remap(None, 11),
            sprites["color_tile"].clone().set_position(18, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(11, 56).color_remap(None, 6),
            sprites["color_tile"].clone().set_position(4, 56).color_remap(None, 15),
            sprites["frame_5slot"].clone().set_position(15, 18),
            sprites["frame_3slot"].clone().set_position(21, 32).color_remap(None, 9),
            sprites["target_box"].clone().set_position(1, 1).color_remap(None, 6),
            sprites["target_box"].clone().set_position(8, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(50, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(15, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(22, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(29, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(36, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(43, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(57, 1).color_remap(None, 15),
            sprites["slot_button"].clone().set_position(17, 20),
            sprites["slot_button"].clone().set_position(23, 20),
            sprites["slot_button"].clone().set_position(29, 20),
            sprites["slot_button"].clone().set_position(29, 34),
            sprites["slot_button"].clone().set_position(23, 34),
            sprites["slot_button"].clone().set_position(35, 20),
            sprites["slot_button"].clone().set_position(35, 34),
            sprites["slot_button"].clone().set_position(41, 20),
            sprites["bg_tile"].clone(),
            sprites["bg_tile"].clone().set_position(7, 0),
            sprites["bg_tile"].clone().set_position(49, 0),
            sprites["bg_tile"].clone().set_position(14, 0),
            sprites["bg_tile"].clone().set_position(21, 0),
            sprites["bg_tile"].clone().set_position(28, 0),
            sprites["bg_tile"].clone().set_position(35, 0),
            sprites["bg_tile"].clone().set_position(42, 0),
            sprites["bg_tile"].clone().set_position(56, 0),
            sprites["nested_tile"].clone().set_position(46, 56).color_remap(None, 9),
            sprites["nested_tile"].clone().set_position(53, 56).color_remap(None, 9),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["color_tile"].clone().set_position(36, 56).color_remap(None, 15),
            sprites["color_tile"].clone().set_position(29, 56).color_remap(None, 6),
            sprites["color_tile"].clone().set_position(22, 56).color_remap(None, 11),
            sprites["color_tile"].clone().set_position(36, 20).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(10, 34).color_remap(None, 9),
            sprites["color_tile"].clone().set_position(36, 34).color_remap(None, 12),
            sprites["color_tile"].clone().set_position(15, 56).color_remap(None, 15),
            sprites["color_tile"].clone().set_position(8, 56).color_remap(None, 6),
            sprites["color_tile"].clone().set_position(1, 56).color_remap(None, 11),
            sprites["frame_3slot"].clone().set_position(8, 18),
            sprites["frame_3slot"].clone().set_position(34, 18).color_remap(None, 14),
            sprites["frame_3slot"].clone().set_position(8, 32).color_remap(None, 9),
            sprites["frame_3slot"].clone().set_position(34, 32).color_remap(None, 12),
            sprites["target_box"].clone().set_position(1, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(8, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(15, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(22, 1).color_remap(None, 12),
            sprites["target_box"].clone().set_position(29, 1).color_remap(None, 15),
            sprites["target_box"].clone().set_position(36, 1).color_remap(None, 15),
            sprites["target_box"].clone().set_position(43, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(50, 1).color_remap(None, 6),
            sprites["target_box"].clone().set_position(57, 1).color_remap(None, 6),
            sprites["slot_button"].clone().set_position(10, 20),
            sprites["slot_button"].clone().set_position(16, 20),
            sprites["slot_button"].clone().set_position(22, 20),
            sprites["slot_button"].clone().set_position(48, 20),
            sprites["slot_button"].clone().set_position(22, 34),
            sprites["slot_button"].clone().set_position(48, 34),
            sprites["slot_button"].clone().set_position(42, 20),
            sprites["slot_button"].clone().set_position(42, 34),
            sprites["slot_button"].clone().set_position(16, 34),
            sprites["bg_tile"].clone(),
            sprites["bg_tile"].clone().set_position(7, 0),
            sprites["bg_tile"].clone().set_position(14, 0),
            sprites["bg_tile"].clone().set_position(21, 0),
            sprites["bg_tile"].clone().set_position(28, 0),
            sprites["bg_tile"].clone().set_position(35, 0),
            sprites["bg_tile"].clone().set_position(42, 0),
            sprites["bg_tile"].clone().set_position(49, 0),
            sprites["bg_tile"].clone().set_position(56, 0),
            sprites["nested_tile"].clone().set_position(43, 56).color_remap(None, 14),
            sprites["nested_tile"].clone().set_position(50, 56).color_remap(None, 9),
            sprites["nested_tile"].clone().set_position(57, 56).color_remap(None, 12),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["color_tile"].clone().set_position(39, 56).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(25, 56).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(18, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(29, 40).color_remap(None, 11),
            sprites["color_tile"].clone().set_position(11, 56).color_remap(None, 9),
            sprites["color_tile"].clone().set_position(4, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(32, 56).color_remap(None, 9),
            sprites["frame_3slot"].clone().set_position(21, 25).color_remap(None, 9),
            sprites["frame_3slot"].clone().set_position(21, 38).color_remap(None, 14),
            sprites["frame_3slot"].clone().set_position(21, 12).color_remap(None, 8),
            sprites["target_box"].clone().set_position(8, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(15, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(22, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(29, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(36, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(43, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(50, 1).color_remap(None, 8),
            sprites["slot_button"].clone().set_position(23, 14),
            sprites["slot_button"].clone().set_position(29, 14),
            sprites["slot_button"].clone().set_position(23, 27),
            sprites["slot_button"].clone().set_position(23, 40),
            sprites["slot_button"].clone().set_position(29, 27),
            sprites["slot_button"].clone().set_position(35, 14),
            sprites["slot_button"].clone().set_position(35, 27),
            sprites["slot_button"].clone().set_position(35, 40),
            sprites["bg_tile"].clone().set_position(7, 0),
            sprites["bg_tile"].clone().set_position(14, 0),
            sprites["bg_tile"].clone().set_position(21, 0),
            sprites["bg_tile"].clone().set_position(28, 0),
            sprites["bg_tile"].clone().set_position(35, 0),
            sprites["bg_tile"].clone().set_position(42, 0),
            sprites["bg_tile"].clone().set_position(49, 0),
            sprites["nested_tile"].clone().set_position(46, 56).color_remap(None, 14),
            sprites["nested_tile"].clone().set_position(53, 56).color_remap(None, 9),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),

    Level(
        sprites=[
            sprites["color_tile"].clone().set_position(39, 56).color_remap(None, 14),
            sprites["color_tile"].clone().set_position(25, 56).color_remap(None, 8),
            sprites["color_tile"].clone().set_position(32, 56).color_remap(None, 9),
            sprites["color_tile"].clone().set_position(18, 56).color_remap(None, 12),
            sprites["color_tile"].clone().set_position(11, 56).color_remap(None, 11),
            sprites["color_tile"].clone().set_position(4, 56).color_remap(None, 15),
            sprites["frame_4slot"].clone().set_position(18, 22),
            sprites["frame_4slot"].clone().set_position(18, 36).color_remap(None, 9),
            sprites["target_box"].clone().set_position(11, 1).color_remap(None, 8),
            sprites["target_box"].clone().set_position(18, 1).color_remap(None, 11),
            sprites["target_box"].clone().set_position(25, 1).color_remap(None, 12),
            sprites["target_box"].clone().set_position(32, 1).color_remap(None, 9),
            sprites["target_box"].clone().set_position(39, 1).color_remap(None, 14),
            sprites["target_box"].clone().set_position(46, 1).color_remap(None, 15),
            sprites["target_box"].clone().set_position(11, 8).color_remap(None, 8),
            sprites["target_box"].clone().set_position(18, 8).color_remap(None, 11),
            sprites["target_box"].clone().set_position(25, 8).color_remap(None, 12),
            sprites["target_box"].clone().set_position(32, 8).color_remap(None, 9),
            sprites["target_box"].clone().set_position(39, 8).color_remap(None, 14),
            sprites["target_box"].clone().set_position(46, 8).color_remap(None, 15),
            sprites["slot_button"].clone().set_position(20, 24),
            sprites["slot_button"].clone().set_position(26, 24),
            sprites["slot_button"].clone().set_position(32, 24),
            sprites["slot_button"].clone().set_position(26, 38),
            sprites["slot_button"].clone().set_position(20, 38),
            sprites["slot_button"].clone().set_position(38, 24),
            sprites["slot_button"].clone().set_position(38, 38),
            sprites["slot_button"].clone().set_position(32, 38),
            sprites["bg_tile"].clone().set_position(10, 0),
            sprites["bg_tile"].clone().set_position(17, 0),
            sprites["bg_tile"].clone().set_position(24, 0),
            sprites["bg_tile"].clone().set_position(31, 0),
            sprites["bg_tile"].clone().set_position(38, 0),
            sprites["bg_tile"].clone().set_position(45, 0),
            sprites["bg_tile"].clone().set_position(10, 7),
            sprites["bg_tile"].clone().set_position(17, 7),
            sprites["bg_tile"].clone().set_position(24, 7),
            sprites["bg_tile"].clone().set_position(31, 7),
            sprites["bg_tile"].clone().set_position(38, 7),
            sprites["bg_tile"].clone().set_position(45, 7),
            sprites["nested_tile"].clone().set_position(46, 56).color_remap(None, 8),
            sprites["nested_tile"].clone().set_position(53, 56).color_remap(None, 9),
            sprites["divider_line"].clone().set_position(0, 53),
        ],
        grid_size=(64, 64),
    ),
]
BACKGROUND_COLOR = 4
PADDING_COLOR = 3
DIVIDER_Y       = 53
COLOR_ENERGY_FILL = 2
COLOR_ENERGY_BG   = 3
COLOR_BLACK       = 0
SLOT_SPACING      = 6
UNUSED_0          = 0
COLOR_BRACKET     = 3
COLOR_GRAY        = 8
COLOR_CURSOR      = 5


class EnergyBarDisplay(RenderableUserDisplay):
    def __init__(self, game: "Sb26") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """"""
        energy_ratio = self.game.energy / self.game.initial_energy
        bar_width = math.ceil(frame.shape[1] * energy_ratio)
        frame[DIVIDER_Y, :] = COLOR_ENERGY_BG
        frame[DIVIDER_Y, 0:bar_width] = COLOR_ENERGY_FILL
        return frame


class Sb26(ARCBaseGame):
    def __init__(self) -> None:
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[EnergyBarDisplay(self)],
        )




        self._log_state = ActionLogState(aggregator=default_aggregator, post_process=compose_default_summary)
        super().__init__(game_id="sb26", levels=levels, camera=camera, available_actions=[5, 6, 7])

        self._frame_sprite_log:       list[list[dict]] = []
        self._frame_nonsprite_log:    list[list[dict]] = []

    def on_set_level(self, level: Level) -> None:
        self._levels[self._current_level_index] = self._clean_levels[self._current_level_index].clone()
        self.initial_energy = 64
        self.energy = self.initial_energy
        self.current_level.remove_sprite(self.current_level.get_sprites_by_name("divider_line")[0])
        self.frames = self.current_level.get_sprites_by_tag("frame")
        self.frames.sort(key=lambda spr: (spr.y, spr.x))
        self.frame_slots: dict[Sprite, list[Sprite]] = {}
        self.targets = self.current_level.get_sprites_by_name("target_box")
        self.targets.sort(key=lambda spr: (spr.y, spr.x))
        self.color_tiles = self.current_level.get_sprites_by_tag("color_tile")
        for tile in self.color_tiles:
            if tile.y > DIVIDER_Y:

                slot = sprites["slot_button"].clone().set_position(tile.x, tile.y).set_visible(False)
                self.current_level.add_sprite(slot)
            else:

                tile.tags.remove("sys_click")
        self.slot_buttons = self.current_level.get_sprites_by_tag("slot_button")
        self.selected_tile: Sprite | None = None
        self.cursor_boxes = [sprites["cursor_box"].clone() for i in range(2)]
        for i, frame_spr in enumerate(self.cursor_boxes):
            frame_spr.color_remap(None, BACKGROUND_COLOR).set_position(-10, -10).set_layer(1 - i)
            self.current_level.add_sprite(frame_spr)
        self.swap_anim_timer = -1
        self.frame_cursor_spr = sprites["frame_cursor"].clone()
        self._reposition_frame(self.frame_cursor_spr, self.frames[0], self.frames[0], 0)
        self.current_level.add_sprite(self.frame_cursor_spr)
        self.slot_cursor_spr = sprites["cursor_box"].clone().set_layer(10)
        self.slot_cursor_spr.color_remap(None, BACKGROUND_COLOR).set_visible(False)
        self.current_level.add_sprite(self.slot_cursor_spr)
        self.delay_timer = 0
        self.frame_stack: list[tuple[Sprite, int]] = []
        self.bracket_trail: list[Sprite] = []
        self.is_backtracking = False
        self.target_idx = 0
        self.answer_cursor_spr = sprites["selected_highlight"].clone().color_remap(None, COLOR_CURSOR).set_layer(10)
        self.current_level.add_sprite(self.answer_cursor_spr.set_position(-100, -100))
        self.flash_timer = -1
        self.color_anim_targets: dict[Sprite, int] = {}
        self.lerp_queue: list[tuple[Sprite, Sprite, Sprite]] = []
        self.lerp_timer = -1
        self.lerp_duration = 6
        blank_pixels = [[-1 for _ in range(64)] for _ in range(64)]
        self.snapshot_sprite = Sprite(pixels=blank_pixels, visible=False, collidable=False, name="snapshot_overlay")
        self.current_level.add_sprite(self.snapshot_sprite)
        self.hidden_sprites: list[Sprite] = []



        self._toggle_hidden_ids_this_step: set[int] = set()
        self.reveal_anim_timer = -1
        self.cleanup_anim_timer = -1
        self.win_anim_timer = -1
        self.bg_tiles = self.current_level.get_sprites_by_name("bg_tile")
        self.wrong_anim_timer = -1
        overlay_name = "overlay_medium" if len(self.targets) < 12 else "overlay_wide"
        self.wrong_overlay_spr = sprites[overlay_name].clone().set_visible(False)
        self.current_level.add_sprite(self.wrong_overlay_spr)
        self.undo_history: list[list[tuple[Sprite, int, int, bool]]] = []




        self._cursor_bright_state: dict[int, str] = {}

        for spr in self.current_level.get_sprites():
            if any(spr.name.startswith(p) for p in ("cursor_box", "frame_cursor", "selected_highlight")):
                if spr.is_visible and int(spr.pixels[0, 0]) != BACKGROUND_COLOR:
                    self._cursor_bright_state[id(spr)] = "on"
        self._emit_initial_layout()
        self._save_state()


    def _emit_initial_layout(self) -> None:
        """













        """
        _layout_rows: list[dict] = []


        for _tag, _role, _role_l2 in (
            ("color_tile",  "piece", ""),
            ("slot_button", "slot",  ""),
            ("frame",       "frame", ""),
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


        for _spr in self.current_level.get_sprites_by_name("target_box"):
            if not _spr.is_visible:
                continue
            _dx, _dy, _dw, _dh = self._to_display_bbox(_spr)
            _layout_rows.append({
                "role":    "target",
                "role_l2": ''                  ,
                "sname":   _spr.name,
                "dx":      _dx, "dy": _dy,
                "dw":      _dw, "dh": _dh,
            })


        try:
            _ns_status = self.get_nonsprite_status() or []
        except Exception:
            _ns_status = []
        for _ns in _ns_status:
            if _ns.get("name") != "energy_bar":
                continue
            _layout_rows.append({
                "role":    "energy",
                "role_l2": ''                     ,
                "sname":   "energy_bar",
                "dx":      int(_ns.get("x", 0)),
                "dy":      int(_ns.get("y", 0)),
                "dw":      int(_ns.get("width", 0)),
                "dh":      int(_ns.get("height", 0)),
            })

        if _layout_rows:
            self._set_event("initial_layout", _layout_rows)

    def step(self) -> None:

        self.action_log    = []
        self.action_log_l2 = []

        if self.action.id not in [
            GameAction.ACTION5,
            GameAction.ACTION6,
            GameAction.ACTION7,
        ]:
            self.complete_action()
            return







        _before = {
            id(spr): {
                "x": spr.x, "y": spr.y,
                "width": spr.width, "height": spr.height,
                "visible": spr.is_visible, "name": spr.name,
                "pixels_bytes": spr.pixels.tobytes(),
                "pixels_top_left": int(spr.pixels[0, 0]),
                "primary_color": self._primary_color_name(spr.pixels),
                "display_bbox": self._to_display_bbox(spr),
            }
            for spr in self.current_level.get_sprites()
        }
        _before_energy = self.energy


        self._before_anim_targets: dict[int, int] = {
            id(spr): tgt for spr, tgt in self.color_anim_targets.items()
        }


        self._before_lerp_ids: set[int] = {id(s) for s, _, _ in self.lerp_queue}


        self._toggle_hidden_ids_this_step = set()


        if len(self.color_anim_targets) > 0:
            for sprite, target_color in list(self.color_anim_targets.items()):
                color = sprite.pixels[0, 0] + np.sign(target_color - sprite.pixels[0, 0])
                sprite.color_remap(None, color)
                if color == target_color:
                    del self.color_anim_targets[sprite]
        if self.wrong_anim_timer >= 0:
            blink = self.wrong_anim_timer // 3 % 2
            self.wrong_overlay_spr.set_visible(blink == 0)
            self.wrong_anim_timer += 1
            if self.wrong_anim_timer >= 18:
                self.wrong_anim_timer = -1
                self.cleanup_anim_timer = 0
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.win_anim_timer >= 0:
            self.win_anim_timer += 1
            if self.win_anim_timer > 15:
                self.win_anim_timer = -1
                self.complete_action()
                self.next_level()
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.reveal_anim_timer >= 0:
            row_i = self.reveal_anim_timer % 4
            cur_frame, slot_idx = self.frame_stack[-1]
            target = self.targets[self.target_idx]
            color = COLOR_BLACK if self.reveal_anim_timer < 4 else target.pixels[0, 0]



            before_pixels = [
                int(spr.pixels[row_i + 1, 1])
                for spr in (target, self.frame_slots[cur_frame][slot_idx])
            ]
            for sprite in (target, self.frame_slots[cur_frame][slot_idx]):
                sprite.pixels[row_i + 1, 1:-1] = color




            after_name = COLOR_PALETTE.get(int(color), f"color{int(color)}")
            after_name_l2 = self._color_l2(after_name)
            phase_en = "covers" if self.reveal_anim_timer < 4 else "reveals"
            phase_l2 = "" if self.reveal_anim_timer < 4 else ""
            for sprite, before_px in zip(
                (target, self.frame_slots[cur_frame][slot_idx]),
                before_pixels,
            ):
                if not sprite.is_visible:
                    continue

                resolved_before = BACKGROUND_COLOR if before_px == -1 else before_px
                before_name = COLOR_PALETTE.get(resolved_before, f"color{resolved_before}")
                before_name_l2 = self._color_l2(before_name)
                dx, dy, dw, dh = self._to_display_bbox(sprite)
                self.action_log.append(
                    f"[change]:[{sprite.name}]:[sname({sprite.name});color({before_name})_({after_name});position({dx}_{dy});bbox({dw}_{dh})]:"
                    f"{sprite.name} at ({dx},{dy}) {phase_en} interior row {row_i+1} from the top: {before_name} to {after_name}."
                    f"|The Object at ({dx},{dy}) {phase_en} interior row {row_i+1} from the top: {before_name} to {after_name}."
                )
                self.action_log_l2.append(
                    f"[change]:[{sprite.name}]:[sname({sprite.name});color({before_name})_({after_name});position({dx}_{dy});bbox({dw}_{dh})]:"
                    f"({dx},{dy}){sprite.name} {row_i+1} {before_name_l2}{after_name_l2}{phase_l2}"
                    f"|({dx},{dy})Object {row_i+1} {before_name_l2}{after_name_l2}{phase_l2}"
                )
            self.reveal_anim_timer += 1
            if self.reveal_anim_timer == 8:
                self.reveal_anim_timer = -1
                self.delay_timer = 5
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.cleanup_anim_timer >= 0:
            row_i = min(self.cleanup_anim_timer, 3)
            for target in self.targets:
                target.pixels[row_i + 1, 1:-1] = -1
            if self.frame_cursor_spr.pixels[0, 0] == BACKGROUND_COLOR:
                self._reposition_frame(self.frame_cursor_spr, self.frames[0], self.frames[0], 0)
                self.frame_cursor_spr.color_remap(None, BACKGROUND_COLOR - 1)
                self.color_anim_targets[self.frame_cursor_spr] = COLOR_BLACK
            self.cleanup_anim_timer += 1
            if self.cleanup_anim_timer >= 8:
                self.cleanup_anim_timer = -1
                for bracket in self.bracket_trail:
                    self.current_level.remove_sprite(bracket)
                self.bracket_trail.clear()
                self.slot_cursor_spr.set_visible(False)
                self._toggle_game_view(False)
                self.complete_action()
                if self.energy == 0:
                    self.lose()
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.flash_timer >= 0:
            blink = self.flash_timer // 3 % 2
            color = COLOR_GRAY if blink == 0 else COLOR_BLACK
            self.slot_cursor_spr.color_remap(None, color)
            self.answer_cursor_spr.color_remap(None, color)
            self.flash_timer += 1
            if self.flash_timer >= 18:
                self.flash_timer = -1
                self.cleanup_anim_timer = 0
                cur_frame = self.frame_stack[-1][0]
                for s in [
                    self.frame_cursor_spr,
                    self.slot_cursor_spr,
                ] + self.bracket_trail:
                    if s == self.frame_cursor_spr and cur_frame == self.frames[0]:
                        continue
                    self.color_anim_targets[s] = BACKGROUND_COLOR
                self.color_anim_targets[self.answer_cursor_spr] = COLOR_CURSOR
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.lerp_queue:
            self.lerp_timer += 1
            t = self.lerp_timer / self.lerp_duration
            t = 1 - (1 - t) * (1 - t)
            for sprite, from_spr, to_spr in self.lerp_queue:
                if sprite.name == "selected_highlight":
                    self._move_sprite_lerp(sprite, from_spr, to_spr, t, 1)
                else:
                    self._reposition_frame(sprite, from_spr, to_spr, t)
            if self.lerp_timer == self.lerp_duration:
                self.lerp_queue.clear()
                self.lerp_timer = -1
                cur_frame, slot_idx = self.frame_stack[-1]
                tile = self.frame_slots[cur_frame][slot_idx]
                self.delay_timer = 10 if slot_idx == 0 or tile.name == "nested_tile" or self.is_backtracking else 5
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.delay_timer > 0:
            self.delay_timer -= 1
            if self.delay_timer == 0:
                self._on_slot_resolved()
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.swap_anim_timer >= 0:
            for frame_spr in self.cursor_boxes:
                frame_spr.color_remap(None, max(0, self.swap_anim_timer - 1))
            self.swap_anim_timer += 1
            if self.cursor_boxes[0].pixels[0, 0] == BACKGROUND_COLOR:
                self.swap_anim_timer = -1
                self.complete_action()
                if self.energy == 0:
                    self.lose()
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        if self.action.id == GameAction.ACTION5:


            self.energy -= 1
            for sprite in (self.frame_cursor_spr, self.cursor_boxes[0]):
                sprite.set_visible(False)
            self.snapshot_sprite.pixels = self.camera.render(self.current_level.get_sprites())
            for sprite in (self.frame_cursor_spr, self.cursor_boxes[0]):
                sprite.set_visible(True)
            if self.selected_tile:
                self.color_anim_targets[self.cursor_boxes[0]] = BACKGROUND_COLOR
                self.selected_tile = None
            self._start_round()
            self._emit_diff_and_frame_logs(_before, _before_energy)
            return
        elif self.action.id == GameAction.ACTION7:
            self._undo()
        else:
            click_x, click_y = (
                self.action.data.get("x", -1),
                self.action.data.get("y", -1),
            )
            if clicked_spr := self.current_level.get_sprite_at(click_x, click_y, "sys_click"):
                self._on_tile_clicked(clicked_spr)
                if self.swap_anim_timer == 0:
                    self._save_state()
                    self._emit_diff_and_frame_logs(_before, _before_energy)
                    return
        self.complete_action()
        self._emit_diff_and_frame_logs(_before, _before_energy)

    def _emit_diff_and_frame_logs(self, _before: dict, _before_energy: int) -> None:
        """











        """








        _BRIGHT_AS_HIGHLIGHT = ("cursor_box", "frame_cursor", "selected_highlight")
        _BRIGHT_AS_FLASH     = ("wrong_overlay", "overlay_medium", "overlay_wide")




        _SUPPRESS_APPEAR     = ("slot_button",)
        _SUPPRESS_DISAPPEAR  = ("slot_button",)

        def _emit_bright(spr_name: str, dx: int, dy: int, dw: int, dh: int, state_en: str, state_l2: str, desc_en: str, desc_l2: str) -> None:
            self.action_log.append(
                f"[bright]:[{spr_name}]:[sname({spr_name});position({dx}_{dy});bbox({dw}_{dh});state({state_en})]:{desc_en}"
            )
            self.action_log_l2.append(
                f"[bright]:[{spr_name}]:[sname({spr_name});position({dx}_{dy});bbox({dw}_{dh});state({state_en})]:{desc_l2}"
            )

        def _is_offscreen(dx: int, dy: int, dw: int, dh: int) -> bool:
            """"""
            return dx + dw <= 0 or dy + dh <= 0 or dx >= 64 or dy >= 64

        _after_sprs = {id(spr): spr for spr in self.current_level.get_sprites()}

        for spr_id, spr in _after_sprs.items():
            b = _before.get(spr_id)
            if b is None or (not b["visible"] and spr.is_visible):



                if spr_id in self._toggle_hidden_ids_this_step:
                    continue
                if spr.name == "snapshot_overlay":
                    continue
                dx, dy, dw, dh = self._to_display_bbox(spr)
                if _is_offscreen(dx, dy, dw, dh):
                    if spr.name.startswith(_BRIGHT_AS_HIGHLIGHT):

                        self._cursor_bright_state[spr_id] = "on" if int(spr.pixels[0, 0]) != BACKGROUND_COLOR else "off"
                    continue
                if spr.name.startswith(_SUPPRESS_APPEAR):

                    continue
                if spr.name.startswith(_BRIGHT_AS_HIGHLIGHT):


                    if int(spr.pixels[0, 0]) != BACKGROUND_COLOR:
                        _emit_bright(
                            spr.name, dx, dy, dw, dh,
                            "selected", "",
                            f"{spr.name} at ({dx},{dy}) appears as a highlight (selection cursor)."
                            f"|The Object at ({dx},{dy}) appears as a highlight.",
                            f"({dx},{dy}){spr.name}"
                            f"|({dx},{dy})Object",
                        )
                        self._cursor_bright_state[spr_id] = "on"
                    else:
                        self._cursor_bright_state[spr_id] = "off"
                elif spr.name.startswith(_BRIGHT_AS_FLASH):
                    _emit_bright(
                        spr.name, dx, dy, dw, dh,
                        "flash", "",
                        f"{spr.name} at ({dx},{dy}) flashes on as part of the wrong-answer animation."
                        f"|The Object at ({dx},{dy}) flashes on.",
                        f"({dx},{dy}){spr.name}"
                        f"|({dx},{dy})Object",
                    )
                else:
                    self.action_log.append(f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).")
                    self.action_log_l2.append(f"[appear]:[{spr.name}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name}({dx},{dy})|Object({dx},{dy})")


        _arc_scale_d = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in _before.items():
            spr = _after_sprs.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bname = b["name"]

                if spr_id in self._toggle_hidden_ids_this_step:
                    continue
                if bname == "snapshot_overlay":
                    continue


                bdx, bdy, bdw, bdh = b["display_bbox"]
                if bname.startswith(_SUPPRESS_DISAPPEAR):


                    continue
                if _is_offscreen(bdx, bdy, bdw, bdh):
                    if bname.startswith(_BRIGHT_AS_HIGHLIGHT):
                        self._cursor_bright_state[spr_id] = "off"
                    continue
                if bname.startswith(_BRIGHT_AS_HIGHLIGHT):

                    if b["pixels_top_left"] != BACKGROUND_COLOR:
                        _emit_bright(
                            bname, bdx, bdy, bdw, bdh,
                            "off", "",
                            f"{bname} at ({bdx},{bdy}) turns off (selection cursor disappears)."
                            f"|The Object at ({bdx},{bdy}) turns off.",
                            f"({bdx},{bdy}){bname}"
                            f"|({bdx},{bdy})Object",
                        )
                    self._cursor_bright_state[spr_id] = "off"
                    continue
                if bname.startswith(_BRIGHT_AS_FLASH):
                    _emit_bright(
                        bname, bdx, bdy, bdw, bdh,
                        "off", "",
                        f"{bname} at ({bdx},{bdy}) flashes off as part of the wrong-answer animation."
                        f"|The Object at ({bdx},{bdy}) flashes off.",
                        f"({bdx},{bdy}){bname}"
                        f"|({bdx},{bdy})Object",
                    )
                    continue
                self.action_log.append(f"[disappear]:[{bname}]:[sname({bname});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{bname} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).")
                self.action_log_l2.append(f"[disappear]:[{bname}]:[sname({bname});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{bname}({bdx},{bdy})|Object({bdx},{bdy})")

        for spr_id, spr in _after_sprs.items():
            if not spr.is_visible:
                continue
            b = _before.get(spr_id)
            if b is None or not b["visible"]:
                continue












            if spr.name.startswith(_BRIGHT_AS_HIGHLIGHT):
                target = self.color_anim_targets.get(spr)
                if target is None:
                    target = self._before_anim_targets.get(spr_id)

                if target is None:
                    intent = None
                elif target == BACKGROUND_COLOR:
                    intent = "off"
                else:
                    intent = "on"
                before_at_bg = b["pixels_top_left"] == BACKGROUND_COLOR
                after_at_bg  = int(spr.pixels[0, 0]) == BACKGROUND_COLOR
                pos_changed = (spr.x, spr.y) != (b["x"], b["y"])
                pixels_changed = spr.pixels.tobytes() != b["pixels_bytes"]


                bdx, bdy, bdw, bdh = b["display_bbox"]
                adx, ady, adw, adh = self._to_display_bbox(spr)
                before_offscreen = _is_offscreen(bdx, bdy, bdw, bdh)
                after_offscreen  = _is_offscreen(adx, ady, adw, adh)


                if pos_changed and not pixels_changed and spr_id in self._before_lerp_ids:
                    if not (before_offscreen and after_offscreen):
                        self.action_log.append(
                            f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:"
                            f"{spr.name} moves from ({bdx},{bdy}) to ({adx},{ady})."
                            f"|Object moves from ({bdx},{bdy}) to ({adx},{ady})."
                        )
                        self.action_log_l2.append(
                            f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:"
                            f"{spr.name}({bdx},{bdy})({adx},{ady})"
                            f"|Object({bdx},{bdy})({adx},{ady})"
                        )
                    continue


                if pos_changed:
                    if (not before_at_bg) and not before_offscreen:

                        _emit_bright(
                            spr.name, bdx, bdy, bdw, bdh,
                            "off", "",
                            f"{spr.name} at ({bdx},{bdy}) turns off (selection cursor leaves)."
                            f"|The Object at ({bdx},{bdy}) turns off.",
                            f"({bdx},{bdy}){spr.name}"
                            f"|({bdx},{bdy})Object",
                        )
                    if (not after_at_bg) and not after_offscreen:

                        _emit_bright(
                            spr.name, adx, ady, adw, adh,
                            "selected", "",
                            f"{spr.name} at ({adx},{ady}) lights up as a highlight (selection cursor)."
                            f"|The Object at ({adx},{ady}) lights up as a highlight.",
                            f"({adx},{ady}){spr.name}"
                            f"|({adx},{ady})Object",
                        )
                    self._cursor_bright_state[spr_id] = "off" if after_at_bg else "on"
                    continue




                prev_state = self._cursor_bright_state.get(spr_id, "off")
                crossed_to_bg   = (not before_at_bg) and after_at_bg
                crossed_from_bg = before_at_bg and (not after_at_bg)
                new_state = prev_state
                if crossed_to_bg and intent != "on":
                    new_state = "off"
                elif crossed_from_bg and intent != "off":
                    new_state = "on"
                if new_state != prev_state and not after_offscreen:
                    if new_state == "on":
                        _emit_bright(
                            spr.name, adx, ady, adw, adh,
                            "selected", "",
                            f"{spr.name} at ({adx},{ady}) lights up as a highlight (selection cursor)."
                            f"|The Object at ({adx},{ady}) lights up as a highlight.",
                            f"({adx},{ady}){spr.name}"
                            f"|({adx},{ady})Object",
                        )
                    else:
                        _emit_bright(
                            spr.name, adx, ady, adw, adh,
                            "off", "",
                            f"{spr.name} at ({adx},{ady}) turns off (selection cursor disappears)."
                            f"|The Object at ({adx},{ady}) turns off.",
                            f"({adx},{ady}){spr.name}"
                            f"|({adx},{ady})Object",
                        )
                self._cursor_bright_state[spr_id] = new_state



                if pixels_changed and not after_offscreen:
                    after_color = self._primary_color_name(spr.pixels)
                    before_color = b["primary_color"]
                    if after_color != before_color:
                        self.action_log.append(f"[change]:[{spr.name}]:[sname({spr.name});color({before_color})_({after_color});position({adx}_{ady});bbox({adw}_{adh})]:{spr.name} at ({adx},{ady}) changes color from {before_color} to {after_color}.|Object at ({adx},{ady}) changes color from {before_color} to {after_color}.")
                        self.action_log_l2.append(f"[change]:[{spr.name}]:[sname({spr.name});color({before_color})_({after_color});position({adx}_{ady});bbox({adw}_{adh})]:({adx},{ady}){spr.name}{self._color_l2(before_color)}{self._color_l2(after_color)}|({adx},{ady})Object{self._color_l2(before_color)}{self._color_l2(after_color)}")
                continue

            if (spr.x, spr.y) != (b["x"], b["y"]):


                bdx, bdy = b["display_bbox"][0], b["display_bbox"][1]
                adx, ady, adw, adh = self._to_display_bbox(spr)
                self.action_log.append(f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady}).")
                self.action_log_l2.append(f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})")
            if spr.pixels.tobytes() != b["pixels_bytes"]:
                after_color = self._primary_color_name(spr.pixels)
                before_color = b["primary_color"]
                if after_color != before_color:
                    adx, ady, adw, adh = self._to_display_bbox(spr)
                    self.action_log.append(f"[change]:[{spr.name}]:[sname({spr.name});color({before_color})_({after_color});position({adx}_{ady});bbox({adw}_{adh})]:{spr.name} at ({adx},{ady}) changes color from {before_color} to {after_color}.|Object at ({adx},{ady}) changes color from {before_color} to {after_color}.")
                    self.action_log_l2.append(f"[change]:[{spr.name}]:[sname({spr.name});color({before_color})_({after_color});position({adx}_{ady});bbox({adw}_{adh})]:({adx},{ady}){spr.name}{self._color_l2(before_color)}{self._color_l2(after_color)}|({adx},{ady})Object{self._color_l2(before_color)}{self._color_l2(after_color)}")

        if self.energy != _before_energy:
            before_px = self._energy_bar_pixel_len(_before_energy)
            after_px  = self._energy_bar_pixel_len(self.energy)
            if before_px != after_px:
                if self.energy < _before_energy:
                    self.action_log.append(f"[change]:[energy]:[sname(energy_bar);count({_before_energy})_({self.energy})]:The gray energy bar at the middle (y={DIVIDER_Y}) shrinks from the right ({_before_energy}→{self.energy}).|The gray Object at the middle (y={DIVIDER_Y}) shrinks from the right ({_before_energy}→{self.energy}).")
                    self.action_log_l2.append(f"[change]:[energy]:[sname(energy_bar);count({_before_energy})_({self.energy})]:(y={DIVIDER_Y}){_before_energy}→{self.energy}|(y={DIVIDER_Y})Object{_before_energy}→{self.energy}")
                else:
                    self.action_log.append(f"[change]:[energy]:[sname(energy_bar);count({_before_energy})_({self.energy})]:The gray energy bar at the middle (y={DIVIDER_Y}) extends to the right ({_before_energy}→{self.energy}).|The gray Object at the middle (y={DIVIDER_Y}) extends to the right ({_before_energy}→{self.energy}).")
                    self.action_log_l2.append(f"[change]:[energy]:[sname(energy_bar);count({_before_energy})_({self.energy})]:(y={DIVIDER_Y}){_before_energy}→{self.energy}|(y={DIVIDER_Y})Object{_before_energy}→{self.energy}")
        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()

    def _energy_bar_pixel_len(self, energy: int) -> int:
        """



        """
        if self.initial_energy <= 0:
            return 0
        ratio = max(0.0, min(1.0, energy / self.initial_energy))
        return math.ceil(64 * ratio)

    def _on_slot_resolved(self) -> None:
        """"""
        cur_frame, slot_idx = self.frame_stack[-1]
        tile = self.frame_slots[cur_frame][slot_idx]
        target = self.targets[self.target_idx]
        is_filled = target.pixels[1, 1] != -1

        if not is_filled and tile.name == "color_tile":
            if tile.pixels[1, 1] == target.pixels[0, 0]:
                self.reveal_anim_timer = 0
            else:
                self._flash_error()
            return

        if self.target_idx == len(self.targets) - 1 and is_filled:
            self.win_anim_timer = 0
            self.color_anim_targets[self.frame_cursor_spr] = BACKGROUND_COLOR
            self.color_anim_targets[self.slot_cursor_spr] = BACKGROUND_COLOR
            for background in self.bg_tiles:
                self.color_anim_targets[background] = COLOR_BLACK
            self._toggle_game_view(False)
            return
        if self.is_backtracking or tile.name == "color_tile":
            if self.is_backtracking:
                self.is_backtracking = False
                self.current_level.remove_sprite(self.bracket_trail.pop())
            slot_idx += 1
            if slot_idx < int(re.search(r"frame_(\d+)slot", cur_frame.name).group(1)):

                self.lerp_queue.append((self.slot_cursor_spr, tile, self.frame_slots[cur_frame][slot_idx]))
                self.target_idx += 1
                next_target = self.targets[self.target_idx]
                self.lerp_queue.append((self.answer_cursor_spr, target, next_target))
                self.lerp_timer = 0
                self.frame_stack[-1] = (cur_frame, slot_idx)
            elif len(self.frame_stack) > 1:

                self.frame_stack.pop()
                self.is_backtracking = True
                parent_frame, slot_idx = self.frame_stack[-1]
                self.lerp_queue.append((self.frame_cursor_spr, cur_frame, parent_frame))
                self.lerp_queue.append(
                    (
                        self.slot_cursor_spr,
                        tile,
                        self.frame_slots[parent_frame][slot_idx],
                    )
                )
                self.lerp_timer = 0
                if len(self.bracket_trail) > 1:
                    self.color_anim_targets[self.bracket_trail[-2]] = COLOR_BRACKET
            else:

                self.color_anim_targets[self.frame_cursor_spr] = BACKGROUND_COLOR
                self.color_anim_targets[self.slot_cursor_spr] = BACKGROUND_COLOR
                self.color_anim_targets[self.answer_cursor_spr] = COLOR_CURSOR
                target = self.targets[self.target_idx + 1]
                pos_x, pos_y = (target.x - 1, target.y - 1)
                if self.wrong_overlay_spr.name == "overlay_medium":
                    width = 7 * (len(self.targets) - self.target_idx - 1) + 1
                    self._resize_frame(self.wrong_overlay_spr, width)
                else:
                    pos_x -= 28
                self.wrong_overlay_spr.set_position(pos_x, pos_y).set_visible(True)
                self.wrong_anim_timer = 0
                return
        elif tile.name == "nested_tile":

            if slot_idx == 0 and (cur_frame, slot_idx) in self.frame_stack[:-1] and (self.frame_stack[-2][1] == 0):
                self._flash_error()
                return
            if len(self.bracket_trail) > 0:
                self.color_anim_targets[self.bracket_trail[-1]] = BACKGROUND_COLOR
            bracket = sprites["bracket_box"].clone().set_position(tile.x, tile.y)
            self.bracket_trail.append(bracket)
            self.current_level.add_sprite(bracket)
            parent_frame = next((spr for spr in self.frames if spr.pixels[0, 0] == tile.pixels[1, 1]))
            self.frame_stack.append((parent_frame, 0))
            self.lerp_queue.append((self.frame_cursor_spr, cur_frame, parent_frame))
            first_slot = self.frame_slots[parent_frame][0]
            self.lerp_queue.append((self.slot_cursor_spr, tile, first_slot))
            self.lerp_timer = 0
        else:
            self._flash_error()

    def _flash_error(self) -> None:
        """"""
        self.slot_cursor_spr.color_remap(None, COLOR_GRAY)
        self.answer_cursor_spr.color_remap(None, COLOR_GRAY)
        self.flash_timer = 0

    def _start_round(self) -> None:
        """
        """
        frame_pad = 2
        for frame in self.frames:
            self.frame_slots[frame] = []
            for i in range(int(re.search(r"frame_(\d+)slot", frame.name).group(1))):
                x, y = (frame.x + frame_pad + i * SLOT_SPACING, frame.y + frame_pad)
                if tile := self._find_tile_at(x, y):
                    self.frame_slots[frame].append(tile)
                elif slot := self._find_slot_at(x, y):
                    self.frame_slots[frame].append(slot)
                else:
                    raise ValueError(f"No item or spot found at expected position ({x}, {y}) for frame {frame.name}")
        self.frame_stack = [(self.frames[0], 0)]
        for frame_spr in (self.slot_cursor_spr, self.answer_cursor_spr):
            frame_spr.set_visible(True)
            self.color_anim_targets[frame_spr] = COLOR_BLACK
        first_slot = self.frame_slots[self.frames[0]][0]
        self._reposition_frame(self.slot_cursor_spr, first_slot, first_slot, 0)
        self.is_backtracking = False
        self.target_idx = 0
        first_target = self.targets[self.target_idx]
        self.answer_cursor_spr.set_visible(True).set_position(first_target.x - 1, first_target.y - 1)
        self._toggle_game_view(True)
        self.delay_timer = 15

    def _toggle_game_view(self, show_snapshot: bool) -> None:
        """








        """
        if show_snapshot:
            for sprite in self.current_level.get_sprites():
                if sprite.name in [
                    "frame_cursor",
                    "cursor_box",
                    "selected_highlight",
                    "target_box",
                ]:
                    continue
                if sprite.is_visible:
                    sprite.set_visible(False)
                    self.hidden_sprites.append(sprite)
                    self._toggle_hidden_ids_this_step.add(id(sprite))
        else:
            for sprite in self.hidden_sprites:
                sprite.set_visible(True)
                self._toggle_hidden_ids_this_step.add(id(sprite))
            self.hidden_sprites.clear()
        self.snapshot_sprite.set_visible(show_snapshot)

    def _on_tile_clicked(self, clicked_spr: Sprite) -> None:
        """





        """
        if not self.selected_tile:
            if "color_tile" in clicked_spr.tags:
                self.selected_tile = clicked_spr
                self._snap_cursor(self.cursor_boxes[0], (clicked_spr.x, clicked_spr.y))

            return
        if "color_tile" in clicked_spr.tags:
            if clicked_spr == self.selected_tile:

                self.selected_tile = None
                self.cursor_boxes[0].color_remap(None, BACKGROUND_COLOR)


            elif self.selected_tile.y > DIVIDER_Y and clicked_spr.y > DIVIDER_Y:

                self.selected_tile = clicked_spr
                self.cursor_boxes[0].set_position(clicked_spr.x, clicked_spr.y)

            else:

                old_pos = (self.selected_tile.x, self.selected_tile.y)
                self._snap_cursor(self.cursor_boxes[1], (clicked_spr.x, clicked_spr.y))
                self.selected_tile.set_position(clicked_spr.x, clicked_spr.y)
                clicked_spr.set_position(*old_pos)
                self.swap_anim_timer = 0
                self.selected_tile = None
                self.energy -= 1
        elif "slot_button" in clicked_spr.tags:

            hidden_slot = self.current_level.get_sprite_at(self.selected_tile.x, self.selected_tile.y, "slot_button")
            assert hidden_slot
            hidden_slot.set_visible(True)
            clicked_spr.set_visible(False)
            self.selected_tile.set_position(clicked_spr.x, clicked_spr.y)
            self.swap_anim_timer = 0
            self.selected_tile = None
            self._snap_cursor(self.cursor_boxes[1], (clicked_spr.x, clicked_spr.y))
            self.energy -= 1

    def _save_state(self) -> None:
        """
        """
        state: list[tuple[Sprite, int, int, bool]] = []
        for sprite in self.color_tiles + self.slot_buttons:
            state.append((sprite, sprite.x, sprite.y, sprite.is_visible))
        self.undo_history.append(state)

    def _undo(self) -> None:
        """"""
        if len(self.undo_history) < 2:
            return
        self.selected_tile = None
        self.cursor_boxes[0].color_remap(None, BACKGROUND_COLOR)
        self.undo_history.pop()
        prev_state = self.undo_history[-1]
        for sprite, x, y, is_visible in prev_state:
            sprite.set_position(x, y).set_visible(is_visible)

    def _find_tile_at(self, x: int, y: int) -> Sprite | None:
        """"""
        for tile in self.color_tiles:
            if tile.x == x and tile.y == y:
                return tile
        return None

    def _find_slot_at(self, x: int, y: int) -> Sprite | None:
        """"""
        for slot in self.slot_buttons:
            if slot.x == x and slot.y == y:
                return slot
        return None

    def _snap_cursor(self, frame_spr: Sprite, pos: tuple[int, int]) -> None:
        """"""
        frame_spr.set_position(*pos).color_remap(None, COLOR_BLACK)

    def _move_sprite_lerp(
        self,
        sprite: Sprite,
        from_spr: Sprite,
        to_spr: Sprite,
        t: float,
        offset: int,
    ) -> None:
        """"""
        from_x, from_y = (from_spr.x - offset, from_spr.y - offset)
        to_x, to_y = (to_spr.x - offset, to_spr.y - offset)
        x = round(from_x + t * (to_x - from_x))
        y = round(from_y + t * (to_y - from_y))
        sprite.set_position(x, y)

    def _reposition_frame(
        self,
        frame_spr: Sprite,
        from_spr: Sprite,
        to_spr: Sprite,
        t: float,
    ) -> None:
        """"""
        pad = 1 if frame_spr.name == "frame_cursor" else 0
        self._move_sprite_lerp(frame_spr, from_spr, to_spr, t, offset=pad)
        from_w = from_spr.pixels.shape[1] + 2 * pad
        to_w   = to_spr.pixels.shape[1]   + 2 * pad
        new_w  = round(from_w + t * (to_w - from_w))
        if frame_spr.pixels[1, new_w - 1] != 1:
            self._resize_frame(frame_spr, new_w, rounded=True)

    def _resize_frame(self, frame_spr: Sprite, width: int, rounded: bool = False) -> None:
        """


        """
        color = frame_spr.pixels[0, 0]
        frame_spr.pixels[:] = -1
        frame_spr.pixels[:, 0] = color
        frame_spr.pixels[:, width - 1] = color
        frame_spr.pixels[0, :width] = color
        frame_spr.pixels[-1, :width] = color
        if rounded:
            corner = 3
            frame_spr.pixels[corner:-corner, 0] = -1
            frame_spr.pixels[corner:-corner, width - 1] = -1
            frame_spr.pixels[0, corner : width - corner] = -1
            frame_spr.pixels[-1, corner : width - corner] = -1

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

        visible_mask = spr.pixels >= 0
        if visible_mask.any():
            cols = visible_mask.any(axis=0)
            rows = visible_mask.any(axis=1)
            col_idx = np.where(cols)[0]
            row_idx = np.where(rows)[0]
            lx = int(col_idx[0])
            ly = int(row_idx[0])
            lw = int(col_idx[-1] - lx + 1)
            lh = int(row_idx[-1] - ly + 1)
        else:
            lx, ly, lw, lh = 0, 0, int(spr.width), int(spr.height)
        dx = (spr.x + lx - cam.x) * arc_scale + x_off
        dy = (spr.y + ly - cam.y) * arc_scale + y_off
        dw = lw * arc_scale
        dh = lh * arc_scale
        return dx, dy, dw, dh

    def _primary_color_name(self, pixels: np.ndarray) -> str:
        """


        """
        flat = pixels.flatten()
        visible = flat[flat >= 0]
        if visible.size == 0:
            return "transparent"
        values, counts = np.unique(visible, return_counts=True)
        dominant = int(values[np.argmax(counts)])
        return COLOR_PALETTE.get(dominant, f"color{dominant}")

    def _color_l2(self, name: str) -> str:
        """"""
        for idx, en in COLOR_PALETTE.items():
            if en == name:
                return COLOR_PALETTE_L2.get(idx, name)
        return name

    def _get_hidden_state(self) -> np.ndarray:
        return np.array([self.energy])

    def _get_valid_actions(self) -> list[ActionInput]:
        """"""
        actions = [ActionInput(id=GameAction.ACTION5)]
        if self.selected_tile:

            slot_at_sel = self.current_level.get_sprite_at(self.selected_tile.x, self.selected_tile.y, "slot_button")
            assert slot_at_sel
            for slot in self.current_level.get_sprites_by_tag("slot_button"):
                if slot == slot_at_sel or (slot_at_sel.y > DIVIDER_Y and slot.y > DIVIDER_Y):
                    continue
                actions.append(
                    ActionInput(
                        id=GameAction.ACTION6,
                        data={"x": slot.x + 1, "y": slot.y + 1},
                    )
                )
        else:

            for tile in self.current_level.get_sprites_by_tag("color_tile"):
                if "sys_click" in tile.tags:
                    actions.append(
                        ActionInput(
                            id=GameAction.ACTION6,
                            data={"x": tile.x + 1, "y": tile.y + 1},
                        )
                    )
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
            "name":     "energy_bar",
            "label":    "Energy",
            "label_l2": ''               ,
            "value":    self.energy,
            "max":      self.initial_energy,
            "x":        0,
            "y":        DIVIDER_Y,
            "width":    64,
            "height":   1,
        })
        return result

    def get_game_knowledge(self) -> dict:
        """"""
        return GAME_KNOWLEDGE
