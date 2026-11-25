"""




















































"""

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    InteractionMode,
    Level,
    RenderableUserDisplay,
    Sprite,
)


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
    "sys_click":   "Clickable",
    "maze_wall":   "Maze Wall",
    "obstacle":    "Obstacle",
    "normal_mode": "Normal Mode Orb",
    "trap":        "Trap Tile",

    "token":                "Token",
    "token-blue":           "Blue Token",
    "token-yellow":         "Yellow Token",
    "barrier-purple":       "Barrier (Purple)",
    "barrier-green":        "Barrier (Green)",
    "barrier-orange":       "Barrier (Orange)",
    "sensor-purple":        "Sensor (Purple)",
    "sensor-green":         "Sensor (Green)",
    "sensor-orange":        "Sensor (Orange)",
    "orb-normal-left":      "Orb (Normal Left)",
    "orb-normal-right":     "Orb (Normal Right)",
    "orb-mirror-left":      "Orb (Mirror Left)",
    "orb-mirror-right":     "Orb (Mirror Right)",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "sys_click":   "",
    "maze_wall":   "",
    "obstacle":    "",
    "normal_mode": "",
    "trap":        "",

    "token":                "",
    "token-blue":           "",
    "token-yellow":         "",
    "barrier-purple":       "",
    "barrier-green":        "",
    "barrier-orange":       "",
    "sensor-purple":        "",
    "sensor-green":         "",
    "sensor-orange":        "",
    "orb-normal-left":      "",
    "orb-normal-right":     "",
    "orb-mirror-left":      "",
    "orb-mirror-right":     "",
}



sprites = {


    "token": Sprite(
        pixels=[
            [BLUE],
        ],
        name="token",
        visible=True,
        collidable=True,
        tags=["sys_click", "obstacle"],
    ),

    "barrier-purple": Sprite(
        pixels=[
            [PURPLE],
            [PURPLE],
            [PURPLE],
        ],
        name="barrier-purple",
        visible=True,
        collidable=True,
    ),

    "barrier-green": Sprite(
        pixels=[
            [GREEN],
            [GREEN],
            [GREEN],
        ],
        name="barrier-green",
        visible=True,
        collidable=True,
    ),

    "barrier-orange": Sprite(
        pixels=[
            [ORANGE],
            [ORANGE],
            [ORANGE],
        ],
        name="barrier-orange",
        visible=True,
        collidable=True,
    ),

    "sensor-purple": Sprite(
        pixels=[
            [PURPLE],
        ],
        name="sensor-purple",
        visible=True,
        collidable=True,
    ),

    "sensor-green": Sprite(
        pixels=[
            [GREEN],
        ],
        name="sensor-green",
        visible=True,
        collidable=True,
    ),

    "sensor-orange": Sprite(
        pixels=[
            [ORANGE],
        ],
        name="sensor-orange",
        visible=True,
        collidable=True,
    ),

    "maze-Level1": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, -1, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level1",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level2": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level2",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level3": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, -1, 0, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0],
            [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1],
            [-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
            [-1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level3",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level4": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, 0],
            [0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level4",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level5": Sprite(
        pixels=[
            [0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
            [-1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1],
            [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1],
            [0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, -1, 0],
            [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0],
        ],
        name="maze-Level5",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level6": Sprite(
        pixels=[
            [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [0, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level6",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level7": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level7",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level8": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0],
            [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level8",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level9": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="maze-Level9",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level10": Sprite(
        pixels=[
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        ],
        name="maze-Level10",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),

    "maze-Level11": Sprite(
        pixels=[
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        ],
        name="maze-Level11",
        visible=True,
        collidable=True,
        tags=["maze_wall"],
    ),


    "orb-mirror-right": Sprite(
        pixels=[
            [LIGHT_BLUE],
        ],
        name="orb-mirror-right",
        visible=True,
        collidable=True,
        tags=["sys_click"],
        layer=2,
    ),


    "orb-mirror-left": Sprite(
        pixels=[
            [LIGHT_BLUE],
        ],
        name="orb-mirror-left",
        visible=True,
        collidable=True,
        tags=["sys_click"],
        layer=2,
    ),


    "orb-normal-right": Sprite(
        pixels=[
            [LIGHT_BLUE],
        ],
        name="orb-normal-right",
        visible=True,
        collidable=True,
        tags=["sys_click", "normal_mode"],
        layer=2,
    ),


    "orb-normal-left": Sprite(
        pixels=[
            [LIGHT_BLUE],
        ],
        name="orb-normal-left",
        visible=True,
        collidable=True,
        tags=["sys_click", "normal_mode"],
        layer=2,
    ),


    "trap": Sprite(
        pixels=[
            [RED],
        ],
        name="trap",
        visible=True,
        collidable=True,
        tags=["trap"],
    ),
}




levels = [

    Level(
        sprites=[
            sprites["maze-Level6"].clone().set_rotation(180),
            sprites["orb-normal-right"].clone().set_position(7, 9),
            sprites["orb-normal-left"].clone().set_position(3, 9),
        ],
        grid_size=(11, 11),
        data={
            "npwxa": [YELLOW, ORANGE],
        },
    ),

    Level(
        sprites=[
            sprites["maze-Level11"].clone().set_position(2, 0),
            sprites["orb-normal-right"].clone().set_position(8, 1),
            sprites["orb-normal-left"].clone().set_position(4, 1),
            sprites["trap"].clone().set_position(5, 5),
            sprites["trap"].clone().set_position(5, 6),
            sprites["trap"].clone().set_position(12, 8),
            sprites["trap"].clone().set_position(11, 8),
            sprites["trap"].clone().set_position(10, 8),
            sprites["trap"].clone().set_position(9, 8),
            sprites["trap"].clone().set_position(8, 8),
            sprites["trap"].clone().set_position(4, 8),
            sprites["trap"].clone().set_position(2, 8),
            sprites["trap"].clone().set_position(1, 8),
            sprites["trap"].clone().set_position(0, 8),
            sprites["trap"].clone().set_position(5, 8),
            sprites["trap"].clone().set_position(5, 7),
            sprites["trap"].clone().set_position(4, 12),
            sprites["trap"].clone().set_position(3, 12),
            sprites["trap"].clone().set_position(2, 12),
            sprites["trap"].clone().set_position(1, 12),
            sprites["trap"].clone().set_position(0, 12),
            sprites["trap"].clone().set_position(9, 12),
            sprites["trap"].clone().set_position(8, 12),
            sprites["trap"].clone().set_position(7, 12),
            sprites["trap"].clone().set_position(6, 12),
            sprites["trap"].clone().set_position(5, 12),
            sprites["trap"].clone().set_position(12, 12),
            sprites["trap"].clone().set_position(11, 12),
            sprites["trap"].clone().set_position(10, 12),
        ],
        grid_size=(13, 13),
        data={
            "npwxa": [MAGENTA, PURPLE],
        },
    ),

    Level(
        sprites=[
            sprites["token"].clone().set_position(1, 3),
            sprites["token"].clone().set_position(6, 2),
            sprites["token"].clone().set_position(8, 6),
            sprites["maze-Level1"].clone(),
            sprites["orb-normal-right"].clone().set_position(8, 10),
            sprites["orb-normal-left"].clone().set_position(4, 10),
        ],
        grid_size=(13, 13),
        data={
            "npwxa": [PURPLE, RED],
        },
    ),

    Level(
        sprites=[
            sprites["token"].clone().set_position(5, 5),
            sprites["maze-Level9"].clone(),
            sprites["orb-normal-right"].clone().set_position(8, 4),
            sprites["orb-normal-left"].clone().set_position(2, 6),
            sprites["trap"].clone().set_position(1, 1),
            sprites["trap"].clone().set_position(2, 1),
            sprites["trap"].clone().set_position(3, 1),
            sprites["trap"].clone().set_position(9, 1),
            sprites["trap"].clone().set_position(8, 1),
            sprites["trap"].clone().set_position(7, 1),
            sprites["trap"].clone().set_position(7, 9),
            sprites["trap"].clone().set_position(9, 9),
            sprites["trap"].clone().set_position(8, 9),
            sprites["trap"].clone().set_position(3, 9),
            sprites["trap"].clone().set_position(2, 9),
            sprites["trap"].clone().set_position(1, 9),
            sprites["trap"].clone().set_position(4, 6),
            sprites["trap"].clone().set_position(5, 6),
            sprites["trap"].clone().set_position(6, 6),
            sprites["trap"].clone().set_position(4, 4),
            sprites["trap"].clone().set_position(5, 4),
            sprites["trap"].clone().set_position(6, 4),
        ],
        grid_size=(11, 11),
        data={
            "npwxa": [YELLOW, PURPLE],
        },
    ),

    Level(
        sprites=[
            sprites["barrier-purple"].clone().set_position(10, 9).set_rotation(90),
            sprites["barrier-purple"].clone().set_position(10, 5).set_rotation(90),
            sprites["barrier-green"].clone().set_position(3, 5).set_rotation(90),
            sprites["barrier-orange"].clone().set_position(2, 9).set_rotation(90),
            sprites["sensor-purple"].clone().set_position(3, 12),
            sprites["sensor-purple"].clone().set_position(3, 1),
            sprites["sensor-green"].clone().set_position(8, 6),
            sprites["sensor-orange"].clone().set_position(14, 6),
            sprites["maze-Level2"].clone(),
            sprites["orb-normal-right"].clone().set_position(1, 12),
            sprites["orb-normal-left"].clone().set_position(13, 12),
        ],
        grid_size=(15, 15),
        data={
            "npwxa": [MAGENTA, PINK],
        },
    ),

    Level(
        sprites=[
            sprites["token"].clone().set_position(6, 9),
            sprites["barrier-green"].clone().set_position(2, 6).set_rotation(90),
            sprites["barrier-orange"].clone().set_position(8, 6).set_rotation(90),
            sprites["sensor-green"].clone().set_position(9, 2),
            sprites["sensor-green"].clone().set_position(3, 9),
            sprites["sensor-orange"].clone().set_position(3, 2),
            sprites["orb-normal-right"].clone().set_position(9, 4),
            sprites["orb-normal-left"].clone().set_position(3, 4),
            sprites["trap"].clone(),
            sprites["trap"].clone().set_position(1, 0),
            sprites["trap"].clone().set_position(2, 0),
            sprites["trap"].clone().set_position(3, 0),
            sprites["trap"].clone().set_position(4, 0),
            sprites["trap"].clone().set_position(5, 0),
            sprites["trap"].clone().set_position(6, 0),
            sprites["trap"].clone().set_position(7, 0),
            sprites["trap"].clone().set_position(8, 0),
            sprites["trap"].clone().set_position(9, 0),
            sprites["trap"].clone().set_position(10, 0),
            sprites["trap"].clone().set_position(11, 0),
            sprites["trap"].clone().set_position(12, 0),
            sprites["trap"].clone().set_position(0, 12),
            sprites["trap"].clone().set_position(1, 12),
            sprites["trap"].clone().set_position(2, 12),
            sprites["trap"].clone().set_position(3, 12),
            sprites["trap"].clone().set_position(4, 12),
            sprites["trap"].clone().set_position(5, 12),
            sprites["trap"].clone().set_position(6, 12),
            sprites["trap"].clone().set_position(7, 12),
            sprites["trap"].clone().set_position(8, 12),
            sprites["trap"].clone().set_position(9, 12),
            sprites["trap"].clone().set_position(10, 12),
            sprites["trap"].clone().set_position(11, 12),
            sprites["trap"].clone().set_position(12, 12),
            sprites["trap"].clone().set_position(0, 11),
            sprites["trap"].clone().set_position(0, 9),
            sprites["trap"].clone().set_position(0, 10),
            sprites["trap"].clone().set_position(0, 5),
            sprites["trap"].clone().set_position(0, 6),
            sprites["trap"].clone().set_position(0, 7),
            sprites["trap"].clone().set_position(0, 8),
            sprites["trap"].clone().set_position(0, 1),
            sprites["trap"].clone().set_position(0, 2),
            sprites["trap"].clone().set_position(0, 3),
            sprites["trap"].clone().set_position(0, 4),
            sprites["trap"].clone().set_position(12, 11),
            sprites["trap"].clone().set_position(12, 9),
            sprites["trap"].clone().set_position(12, 10),
            sprites["trap"].clone().set_position(12, 5),
            sprites["trap"].clone().set_position(12, 6),
            sprites["trap"].clone().set_position(12, 7),
            sprites["trap"].clone().set_position(12, 8),
            sprites["trap"].clone().set_position(12, 1),
            sprites["trap"].clone().set_position(12, 2),
            sprites["trap"].clone().set_position(12, 3),
            sprites["trap"].clone().set_position(12, 4),
            sprites["trap"].clone().set_position(6, 11),
            sprites["trap"].clone().set_position(6, 10),
            sprites["trap"].clone().set_position(6, 5),
            sprites["trap"].clone().set_position(6, 6),
            sprites["trap"].clone().set_position(6, 7),
            sprites["trap"].clone().set_position(6, 8),
            sprites["trap"].clone().set_position(6, 1),
            sprites["trap"].clone().set_position(6, 2),
            sprites["trap"].clone().set_position(6, 3),
            sprites["trap"].clone().set_position(6, 4),
            sprites["trap"].clone().set_position(5, 6),
            sprites["trap"].clone().set_position(1, 6),
            sprites["trap"].clone().set_position(7, 6),
            sprites["trap"].clone().set_position(11, 6),
        ],
        grid_size=(13, 13),
        data={
            "npwxa": [MAGENTA, PINK],
        },
    ),
]


BACKGROUND_COLOR = BLACK
PADDING_COLOR    = WHITE




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


def _build_level_knowledge(level_index: int, level: Level) -> dict:
    """"""
    npwxa = None
    try:
        npwxa = level.get_data("npwxa")
    except Exception:
        pass

    grid = getattr(level, "grid_size", (11, 11)) or (11, 11)

    level_sprites_info: list[dict] = []
    for spr in getattr(level, "_sprites", []) or []:
        tags = list(spr._tags) if spr._tags else []
        level_sprites_info.append({
            "name":     spr.name,
            "x":        spr.x,
            "y":        spr.y,
            "width":    int(spr.width),
            "height":   int(spr.height),
            "rotation": spr.rotation,
            "tags":     tags,
        })


    _analyses: list[dict] = [

        {
            "gimmicks":    ["Maze rotated 180 degrees. One normal orb pair. No traps."],
            "gimmicks_l2": [''                                                                                  ],
            "subgoals":    ["Move orbs so the normal-left and normal-right overlap at the same cell."],
            "subgoals_l2": [''                                                                                 ],
            "difficulty":  "easy",
        },

        {
            "gimmicks":    ["Many trap tiles form a fence pattern. Avoid landing orbs on traps."],
            "gimmicks_l2": [''                                                                                                      ],
            "subgoals":    ["Navigate the orbs through the T-shaped maze without touching traps."],
            "subgoals_l2": [''                                                                            ],
            "difficulty":  "medium",
        },

        {
            "gimmicks":    ["Three tokens act as movable obstacles. Click a token to move it individually and clear a path."],
            "gimmicks_l2": [''                                                                                                                   ],
            "subgoals":    ["Rearrange tokens to open a path, then merge the normal orb pair."],
            "subgoals_l2": [''                                                                                       ],
            "difficulty":  "medium",
        },

        {
            "gimmicks":    ["One token, many traps at corners and center rows. Square-shaped maze."],
            "gimmicks_l2": [''                                                                               ],
            "subgoals":    ["Use the token to block or redirect one orb, then merge the pair in the safe zone."],
            "subgoals_l2": [''                                                                                                         ],
            "difficulty":  "hard",
        },

        {
            "gimmicks":    ["Two pairs of orbs (normal + mirror). Barriers block paths; orbs landing on sensors remove the matching barrier."],
            "gimmicks_l2": [''                                                                                                                                         ],
            "subgoals":    ["Trigger sensors to open paths through barriers, then merge each pair."],
            "subgoals_l2": [''                                                                     ],
            "difficulty":  "hard",
        },

        {
            "gimmicks":    ["Dense trap border. Token, two barrier types, sensors, and two orb pairs."],
            "gimmicks_l2": [''                                                                                                     ],
            "subgoals":    ["Carefully move token to open routes, trigger sensors, and merge both orb pairs."],
            "subgoals_l2": [''                                                                                                            ],
            "difficulty":  "very_hard",
        },
    ]

    analysis = _analyses[level_index] if level_index < len(_analyses) else {}

    return {
        "level_index":   level_index + 1,
        "grid_size":     list(grid),
        "bg_colors":     npwxa,
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
        "game_id":        "m0r0",
        "title":          "M0r0 (Mirror Orb)",
        "title_l2":       ''                            ,
        "description":    "Move mirrored orb pairs so they overlap at the same cell to clear each pair.",
        "description_l2": ''                                                                                          ,
        "num_levels":     len(levels),

        "goal":           "Make every orb pair (normal-left+normal-right, mirror-left+mirror-right) meet at the same cell.",
        "goal_l2":        ''                                                                                                     ,
        "win_condition":  "All orb pairs have merged (set to INTANGIBLE). No active orbs remain.",
        "win_condition_l2": ''                                                                ,
        "lose_condition": "Action count exceeds 150 steps.",
        "lose_condition_l2": ''                                             ,


        "actions": {
            "ACTION1": {
                "key":             "Up",
                "effect":          "Move all active orbs up by 1 cell (mirror-right and orb-mirror-left move DOWN)",
                "effect_l2":       ''                                                                                     ,
                "step_cost":       1,
            },
            "ACTION2": {
                "key":             "Down",
                "effect":          "Move all active orbs down by 1 cell (mirror pairs inverted)",
                "effect_l2":       ''                                                                ,
                "step_cost":       1,
            },
            "ACTION3": {
                "key":             "Left",
                "effect":          "Move all active orbs left by 1 cell (orb-normal-right and orb-mirror-right move RIGHT)",
                "effect_l2":       ''                                                                                          ,
                "step_cost":       1,
            },
            "ACTION4": {
                "key":             "Right",
                "effect":          "Move all active orbs right by 1 cell (right variants inverted)",
                "effect_l2":       ''                                                                   ,
                "step_cost":       1,
            },
            "ACTION5": {
                "key":             "Space / Action5",
                "effect":          "No effect in this game (unused)",
                "effect_l2":       ''                              ,
                "step_cost":       0,
            },
            "ACTION6": {
                "key":             "Click",
                "effect":          "Click a token to enter individual-move mode; subsequent directional actions move only that token",
                "effect_l2":       ''                                                                                                                        ,
                "step_cost":       1,
            },
        },


        "sprite_roles": {
            "orb-normal-left":     "Orb that moves in the same direction as input (+dx, +dy). Blue (9) when selected.",
            "orb-normal-left_l2":  ''                                                                                ,
            "orb-normal-right":    "Orb that moves with horizontal flip (-dx, +dy).",
            "orb-normal-right_l2": ''                                                  ,
            "orb-mirror-left":     "Orb that moves with vertical flip (+dx, -dy).",
            "orb-mirror-left_l2":  ''                                                  ,
            "orb-mirror-right":    "Orb that moves with both flips (-dx, -dy).",
            "orb-mirror-right_l2": ''                                                                 ,
            "token":               "Movable blue dot. Click to select, then use directional actions to push it. Blocked by maze walls and obstacle-tagged sprites.",
            "token_l2":            ''                                                                                                                       ,
            "maze_wall":           "Static maze structure. Orbs and tokens cannot pass through solid (0/white) pixels.",
            "maze_wall_l2":        ''                                                                                                        ,
            "barrier":             "Collidable vertical strip (3 cells). Removed when a matching sensor is activated.",
            "barrier_l2":          ''                                                                                                    ,
            "sensor":              "1x1 colored cell. When an active orb occupies the same position, the matching barrier is removed.",
            "sensor_l2":           ''                                                                                                            ,
            "trap":                "Red (8) 1x1 tile. If an orb lands on a trap, all clickable sprites flash and reset to initial positions.",
            "trap_l2":             ''                                                                                                               ,
        },


        "mechanics": {
            "mirror_movement":        "Each orb pair has a 'left' and 'right' variant. Action1-4 moves all active orbs simultaneously but with direction transforms per variant.",
            "mirror_movement_l2":     ''                                                                                                                                                               ,
            "orb_matching":           "When two orbs of a pair occupy the exact same grid cell after movement, they 'match' and become INTANGIBLE (invisible/removed). Clearing all pairs advances to next level.",
            "orb_matching_l2":        ''                                                                                                                                                                          ,
            "collision_model":        "Orbs cannot move into maze_wall solid pixels (0). They also cannot cross barrier sprites. Token blocks with 'obstacle' tag.",
            "collision_model_l2":     ''                                                                                                                                                  ,
            "trap_reset":             "If an orb lands on a trap tile, a 6-frame flash animation plays, then all clickable sprites return to their original positions from when the level started.",
            "trap_reset_l2":          ''                                                                                                                                                                         ,
            "sensor_barrier":         "Sensors and barriers are color-coded (purple/green/orange). An orb landing on a sensor of color X removes all barrier-X sprites.",
            "sensor_barrier_l2":      ''                                                                                                                                                         ,
            "token_individual_move":  "Clicking a token enters individual-move mode. Directional actions then move only the token (not orbs). Clicking empty space or a non-token sprite exits this mode.",
            "token_individual_move_l2": ''                                                                                                                                                                                                ,
            "step_limit":             "The level has a 150-step budget. Exceeding it causes an immediate loss.",
            "step_limit_l2":          ''                                                                  ,
            "background_coloring":    "Background cells are colored using two colors from level data['npwxa']. When all 4 orb types are active, the background uses a 2x2 quadrant pattern; otherwise a 2-column split.",
            "background_coloring_l2": ''                                                                                                                                                                           ,
        },
    },


    "levels": [
        _build_level_knowledge(i, lv)
        for i, lv in enumerate(levels)
    ],
}




class StepProgressBar(RenderableUserDisplay):
    """






    """

    def __init__(self, max_steps: int):
        """"""
        self.max_steps = max_steps
        self.current_steps = max_steps

    def set_steps(self, steps: int) -> None:
        """"""
        self.current_steps = max(0, min(steps, self.max_steps))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """"""
        if self.max_steps == 0:
            return frame
        ratio = self.current_steps / self.max_steps
        filled = round(64 * ratio)
        filled = min(filled, 64)

        for x in range(64):
            frame[0, x] = BLACK if x < filled else WHITE

        for x in range(64):
            frame[63, x] = BLACK if 63 - x < filled else WHITE
        return frame




class BackgroundColorizer(RenderableUserDisplay):
    """






    """

    def __init__(self, game: "M0r0") -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """"""

        npwxa = self.game.current_level.get_data("npwxa")
        if npwxa and len(npwxa) >= 2:
            color1, color2 = (npwxa[0], npwxa[1])
            white_mask = frame == WHITE
            if np.any(white_mask):

                active_orb_count = 0
                for orb_name in [
                    "orb-normal-left",
                    "orb-normal-right",
                    "orb-mirror-left",
                    "orb-mirror-right",
                ]:
                    if orb_name not in self.game.matched_orbs:
                        orb_list = self.game.current_level.get_sprites_by_name(orb_name)
                        if orb_list and orb_list[0].interaction != InteractionMode.REMOVED:
                            active_orb_count += 1
                bg_canvas = np.zeros((64, 64), dtype=np.int8)
                if active_orb_count == 4:

                    bg_canvas[:32, :32] = color1
                    bg_canvas[:32, 32:] = color2
                    bg_canvas[32:, :32] = color2
                    bg_canvas[32:, 32:] = color1
                else:

                    bg_canvas[:, :32] = color1
                    bg_canvas[:, 32:] = color2
                frame[white_mask] = bg_canvas[white_mask]


        grid_width, grid_height = self.game.current_level.grid_size or (64, 64)
        scale_x = 64 // grid_width
        scale_y = 64 // grid_height
        scale = min(scale_x, scale_y)
        scaled_width = grid_width * scale
        scaled_height = grid_height * scale
        x_offset = (64 - scaled_width) // 2
        y_offset = (64 - scaled_height) // 2




        for token_spr in self.game.current_level.get_sprites_by_tag("obstacle"):
            if token_spr.is_visible:
                px = token_spr.x * scale + x_offset
                py = token_spr.y * scale + y_offset
                for y in range(py, min(py + scale, 64)):
                    for x in range(px, min(px + scale, 64)):

                        if y == py or y == py + scale - 1 or x == px or x == px + scale - 1:
                            frame[y, x] = BLACK


        for trap_spr in self.game.current_level.get_sprites_by_name("trap"):

            if self.game.current_level.get_sprite_at(trap_spr.x, trap_spr.y, "normal_mode"):
                continue
            px = trap_spr.x * scale + x_offset
            py = trap_spr.y * scale + y_offset
            for y in range(py, min(py + scale, 64)):
                for x in range(px, min(px + scale, 64)):
                    if (x + y) % 2 == 1:
                        frame[y, x] = BLACK

        return frame




class M0r0(ARCBaseGame):
    """


















    """





    def __init__(self) -> None:
        """"""

        camera = Camera(background=BACKGROUND_COLOR, letter_box=PADDING_COLOR)


        self.bwyvb: dict[str, tuple[int, int]] = {}
        self.matched_orbs: set[str] = set()
        self.erwkb = BackgroundColorizer(self)
        self.tmbjb = StepProgressBar(max_steps=150)
        self.cfwgj: Sprite | None = None
        self.vmcbq = True






        from gamesofts._action_log import ActionLogState
        self._log_state = ActionLogState(
            aggregator   = type(self)._aggregate_logs,
            post_process = self._compose_summary,
        )

        self._frame_sprite_log:      list[list[dict]] = []
        self._frame_nonsprite_log:   list[list[dict]] = []






        self._last_logged_action_count: int = 0

        super().__init__(
            game_id="m0r0",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5, 6],
        )

        self._camera.replace_interface([self.erwkb, self.tmbjb])



    def on_set_level(self, level: Level) -> None:
        """"""
        self.bwyvb = {}
        self.matched_orbs = set()
        self.tmbjb.set_steps(150)

        for token_spr in self.current_level.get_sprites_by_tag("obstacle"):
            token_spr.color_remap(None, BLUE)
            token_spr.set_name("token-blue")
        self.cfwgj = None
        self.vmcbq = True

        self.yvmbd: list[Sprite] = []
        self.zgdmc = -1

        self.myxvz = [(s, s.x, s.y) for s in self.current_level.get_sprites_by_tag("sys_click")]




        _LAYOUT_ROLE_L2: dict[str, tuple[str, str]] = {

            "sys_click":   ("clickable",  ""),
            "trap":        ("trap_tile",  ""),
        }
        _layout_rows: list[dict] = []
        for _tag, (_role, _role_l2) in _LAYOUT_ROLE_L2.items():
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
        if _layout_rows:
            self._set_event("initial_layout", _layout_rows)



    def _get_hidden_state(self) -> np.ndarray:
        """"""
        state = np.zeros((4, 4), dtype=np.int16)
        state[0, 0] = int(self._action_count)
        return state



    def step(self) -> None:
        """






        """










        _before_sprites = {
            id(spr): {
                "name":    spr.name,
                "x":       spr.x,
                "y":       spr.y,
                "width":   spr.width,
                "height":  spr.height,
                "rotation": spr.rotation,
                "visible": spr.is_visible,


                "pixels_repr": int(spr.pixels[1, 1]) if (spr.pixels.shape[0] > 1 and spr.pixels.shape[1] > 1) else int(spr.pixels.flat[0]),
            }
            for spr in self.current_level.get_sprites()
        }
        _before_action_count = self._action_count








        if self.zgdmc >= 0:
            flash_yellow = self.zgdmc % 2 == 0 and self.zgdmc < 5
            flash_color = YELLOW if flash_yellow else LIGHT_BLUE
            after_name    = COLOR_PALETTE[flash_color]["name"].lower()
            after_name_l2 = COLOR_PALETTE[flash_color]["label_l2"]
            for flash_spr in self.yvmbd:

                if flash_spr.pixels.shape[0] > 1 and flash_spr.pixels.shape[1] > 1:
                    before_color = int(flash_spr.pixels[1, 1])
                else:
                    before_color = int(flash_spr.pixels.flat[0])
                flash_spr.color_remap(None, flash_color)
                if before_color != flash_color and before_color >= 0:
                    before_name    = COLOR_PALETTE.get(before_color, {}).get("name", str(before_color)).lower()
                    before_name_l2 = COLOR_PALETTE.get(before_color, {}).get("label_l2", str(before_color))
                    fdx, fdy, fdw, fdh = self._to_display_bbox(flash_spr)
                    _t, _le, _l2 = self._display_for(flash_spr.name)
                    self._step_log.append(
                        f"[change]:[{_t}]:[sname({flash_spr.name});color({before_name})_({after_name});position({fdx}_{fdy});bbox({fdw}_{fdh})]:"
                        f"{_le} at ({fdx},{fdy}) flashes from {before_name} to {after_name}."
                        f"|Object at ({fdx},{fdy}) flashes from {before_name} to {after_name}."
                    )
                    self._step_log_l2.append(
                        f"[change]:[{_t}]:[sname({flash_spr.name});color({before_name})_({after_name});position({fdx}_{fdy});bbox({fdw}_{fdh})]:"
                        f"({fdx},{fdy}){_l2}{before_name_l2}{after_name_l2}"
                        f"|({fdx},{fdy})Object{before_name_l2}{after_name_l2}"
                    )
            self.zgdmc += 1
            if self.zgdmc > 6:

                for spr, init_x, init_y in self.myxvz:
                    spr.set_position(init_x, init_y)
                self.zgdmc = -1
                self.yvmbd = []
                self.complete_action()




                self._log_diffs(_before_sprites, _before_action_count)
            self._append_frame_logs()
            return






        steps_remaining = 150 - self._action_count
        self.tmbjb.set_steps(steps_remaining)
        if self._action_count > 150:

            self._set_event("gameover", {
                "en": "[gameover]:[game]:[]:Step limit (150) exceeded, game over.|Game over.",
                "l2": "[gameover]:[game]:[]:150|",
            })
            self.lose()
            self._append_frame_logs()
            self.complete_action()
            return


        if self.action.id == GameAction.ACTION6:
            click_x = self.action.data["x"]
            click_y = self.action.data["y"]
            grid_pos = self.camera.display_to_grid(click_x, click_y)
            if grid_pos:
                grid_col, grid_row = grid_pos
                clicked_spr = self.current_level.get_sprite_at(grid_col, grid_row, tag="sys_click")
                if clicked_spr and clicked_spr.name in ("token", "token-blue", "token-yellow"):

                    self.vmcbq = False

                    for orb_name in [
                        "orb-normal-left",
                        "orb-normal-right",
                        "orb-mirror-left",
                        "orb-mirror-right",
                    ]:
                        orb_list = self.current_level.get_sprites_by_name(orb_name)
                        if orb_list and orb_name not in self.matched_orbs:
                            orb_list[0].color_remap(None, OFF_WHITE)

                    if self.cfwgj:
                        prev_token = self.cfwgj
                        prev_dx, prev_dy, prev_dw, prev_dh = self._to_display_bbox(prev_token)
                        prev_token.color_remap(None, BLUE)
                        prev_token.set_name("token-blue")
                        self._step_log.append(f"[change]:[blue_token]:[sname(token-blue);color(yellow)_(blue);position({prev_dx}_{prev_dy});bbox({prev_dw}_{prev_dh})]:Blue Token at ({prev_dx},{prev_dy}) recolors from yellow to blue (previous selection cleared).|Object at ({prev_dx},{prev_dy}) recolors from yellow to blue.")
                        self._step_log_l2.append(f"[change]:[blue_token]:[sname(token-blue);color(yellow)_(blue);position({prev_dx}_{prev_dy});bbox({prev_dw}_{prev_dh})]:({prev_dx},{prev_dy})|({prev_dx},{prev_dy})Object")

                    self.cfwgj = clicked_spr
                    self.set_placeable_sprite(self.cfwgj)
                    self.cfwgj.color_remap(None, YELLOW)
                    self.cfwgj.set_name("token-yellow")
                    sdx, sdy, sdw, sdh = self._to_display_bbox(self.cfwgj)
                    self._step_log.append(f"[change]:[yellow_token]:[sname(token-yellow);color(blue)_(yellow);position({sdx}_{sdy});bbox({sdw}_{sdh})]:Yellow Token at ({sdx},{sdy}) recolors from blue to yellow (becomes active for individual move).|Object at ({sdx},{sdy}) recolors from blue to yellow.")
                    self._step_log_l2.append(f"[change]:[yellow_token]:[sname(token-yellow);color(blue)_(yellow);position({sdx}_{sdy});bbox({sdw}_{sdh})]:({sdx},{sdy})|({sdx},{sdy})Object")
                else:

                    self.vmcbq = True
                    for orb_name in [
                        "orb-normal-left",
                        "orb-normal-right",
                        "orb-mirror-left",
                        "orb-mirror-right",
                    ]:
                        orb_list = self.current_level.get_sprites_by_name(orb_name)
                        if orb_list and orb_name not in self.matched_orbs:
                            orb_list[0].color_remap(None, LIGHT_BLUE)
                    if self.cfwgj:
                        prev_token = self.cfwgj
                        prev_dx, prev_dy, prev_dw, prev_dh = self._to_display_bbox(prev_token)
                        prev_token.color_remap(None, BLUE)
                        prev_token.set_name("token-blue")
                        self.cfwgj = None
                        self.set_placeable_sprite(self.cfwgj)
                        self._step_log.append(f"[change]:[blue_token]:[sname(token-blue);color(yellow)_(blue);position({prev_dx}_{prev_dy});bbox({prev_dw}_{prev_dh})]:Blue Token at ({prev_dx},{prev_dy}) recolors from yellow to blue (selection cleared, returns to bulk move).|Object at ({prev_dx},{prev_dy}) recolors from yellow to blue.")
                        self._step_log_l2.append(f"[change]:[blue_token]:[sname(token-blue);color(yellow)_(blue);position({prev_dx}_{prev_dy});bbox({prev_dw}_{prev_dh})]:({prev_dx},{prev_dy})|({prev_dx},{prev_dy})Object")
            self._log_diffs(_before_sprites, _before_action_count)
            self._append_frame_logs()
            self.complete_action()
            return


        dx = 0
        dy = 0
        if self.action.id == GameAction.ACTION1:
            dy = -1
            self.set_placeable_sprite(None)
        elif self.action.id == GameAction.ACTION2:
            dy = 1
            self.set_placeable_sprite(None)
        elif self.action.id == GameAction.ACTION3:
            dx = -1
            self.set_placeable_sprite(None)
        elif self.action.id == GameAction.ACTION4:
            dx = 1
            self.set_placeable_sprite(None)


        if self.cfwgj and (not self.vmcbq):
            grid_width, grid_height = self.current_level.grid_size or (64, 64)
            new_x = self.cfwgj.x + dx
            new_y = self.cfwgj.y + dy
            if new_x < 0 or new_x >= grid_width or new_y < 0 or new_y >= grid_height:

                self._append_frame_logs()
                self.complete_action()
                return
            self.try_move_sprite(self.cfwgj, dx, dy)

            self._check_sensors()

            self._log_diffs(_before_sprites, _before_action_count)
            self._append_frame_logs()
            self.complete_action()
            return


        if self.vmcbq:

            active_orbs = []
            for orb_name in [
                "orb-normal-left",
                "orb-normal-right",
                "orb-mirror-left",
                "orb-mirror-right",
            ]:
                orb_list = self.current_level.get_sprites_by_name(orb_name)
                if orb_list and orb_name not in self.matched_orbs:
                    active_orbs.append((orb_name, orb_list[0]))

            for name, spr in active_orbs:
                self.bwyvb[name] = (spr.x, spr.y)


            move_queue = []
            for name, spr in active_orbs:
                if "orb-normal-left" in name:

                    move_queue.append((spr, dx, dy))
                elif "orb-normal-right" in name:

                    move_queue.append((spr, -dx, dy))
                elif "orb-mirror-left" in name:

                    move_queue.append((spr, dx, -dy))
                elif "orb-mirror-right" in name:

                    move_queue.append((spr, -dx, -dy))


            for spr, move_dx, move_dy in move_queue:
                if move_dx != 0 or move_dy != 0:
                    self._move_orb(spr, move_dx, move_dy)


            for orb_name, orb_spr in active_orbs:
                if self.current_level.get_sprite_at(orb_spr.x, orb_spr.y, "trap"):
                    self.yvmbd.append(orb_spr)
            if self.yvmbd:






                self._set_event("flash", {
                    "orbs": [
                        {"name": _o.name, "pos_grid": (_o.x, _o.y)}
                        for _o in self.yvmbd
                    ],
                })
                self.zgdmc = 0
                self._log_diffs(_before_sprites, _before_action_count)
                self._append_frame_logs()
                return


            active_pair = [(name, spr) for name, spr in active_orbs if name not in self.matched_orbs and spr.interaction != InteractionMode.REMOVED]
            for i, (name_a, spr_a) in enumerate(active_pair):
                if name_a in self.matched_orbs:
                    continue
                for name_b, spr_b in active_pair[i + 1:]:
                    if name_b in self.matched_orbs:
                        continue
                    if name_a in self.bwyvb and name_b in self.bwyvb:
                        prev_ax, prev_ay = self.bwyvb[name_a]
                        prev_bx, prev_by = self.bwyvb[name_b]




                        horizontal_adj = abs(prev_ax - prev_bx) == 1 and prev_ay == prev_by
                        vertical_adj   = abs(prev_ay - prev_by) == 1 and prev_ax == prev_bx
                        if horizontal_adj or vertical_adj:
                            crossed = (
                                (spr_a.x == prev_bx and spr_a.y == prev_by) or
                                (spr_b.x == prev_ax and spr_b.y == prev_ay)
                            )
                            if crossed:

                                mid_x = (spr_a.x + spr_b.x) // 2
                                mid_y = (spr_a.y + spr_b.y) // 2
                                spr_a.set_position(mid_x, mid_y)
                                spr_b.set_position(mid_x, mid_y)


            pos_map: dict[tuple[int, int], list[tuple[str, Sprite]]] = {}
            for name, spr in active_pair:
                pos = (spr.x, spr.y)
                if pos not in pos_map:
                    pos_map[pos] = []
                pos_map[pos].append((name, spr))





            _pair_events: list[dict] = []
            for pos, orb_group in pos_map.items():
                if len(orb_group) == 2:


                    mpx, mpy, mpw, mph = self._to_display_bbox(orb_group[0][1])
                    for name, spr in orb_group:
                        self.matched_orbs.add(name)
                        spr.set_interaction(InteractionMode.INTANGIBLE)




                    _pair_events.append({
                        "a": orb_group[0][0], "b": orb_group[1][0],
                        "dx": mpx, "dy": mpy, "dw": mpw, "dh": mph,
                    })
                elif len(orb_group) > 2:

                    mpx, mpy, mpw, mph = self._to_display_bbox(orb_group[0][1])
                    for name, spr in orb_group[:2]:
                        self.matched_orbs.add(name)
                        spr.set_interaction(InteractionMode.INTANGIBLE)



                    _pair_events.append({
                        "a": orb_group[0][0], "b": orb_group[1][0],
                        "dx": mpx, "dy": mpy, "dw": mpw, "dh": mph,
                    })
                    for name, spr in orb_group[2:]:
                        if name in self.bwyvb:
                            prev_x, prev_y = self.bwyvb[name]
                            spr.set_position(prev_x, prev_y)
            if _pair_events:


                prev = self._log_state.get_event("orb_pair_overlap", [])
                self._set_event("orb_pair_overlap", list(prev) + _pair_events)


            self._check_sensors()


            remaining = sum(
                1 for name, spr in active_orbs
                if name not in self.matched_orbs and spr.interaction != InteractionMode.INTANGIBLE
            )
            if remaining == 0:
                self._set_event("win", {
                    "en": "[win]:[game]:[]:All orbs matched, the level is cleared.|The level is cleared.",
                    "l2": "[win]:[game]:[]:|",
                })
                self._log_diffs(_before_sprites, _before_action_count)
                self._append_frame_logs()
                self.next_level()




                self.complete_action()
                return

        self._log_diffs(_before_sprites, _before_action_count)
        self._append_frame_logs()
        self.complete_action()



    def _move_orb(self, spr: Sprite, dx: int, dy: int) -> None:
        """



        """
        grid_width, grid_height = self.current_level.grid_size or (64, 64)
        new_x = spr.x + dx
        new_y = spr.y + dy

        if new_x < 0 or new_x >= grid_width or new_y < 0 or new_y >= grid_height:
            return
        orig_x, orig_y = spr.x, spr.y
        spr.move(dx, dy)

        for wall_spr in self.current_level.get_sprites_by_tag("maze_wall"):
            if spr.collides_with(wall_spr):
                spr.set_position(orig_x, orig_y)
                return

        for obs_spr in self.current_level.get_sprites_by_tag("obstacle"):
            if spr.collides_with(obs_spr):
                spr.set_position(orig_x, orig_y)
                return

        for color_suffix in ["purple", "green", "orange"]:
            for barrier_spr in self.current_level.get_sprites_by_name(f"barrier-{color_suffix}"):
                if barrier_spr.is_collidable and spr.collides_with(barrier_spr):
                    spr.set_position(orig_x, orig_y)
                    return



    def _check_sensors(self) -> None:
        """"""

        active_positions = []
        for orb_name in [
            "orb-normal-left",
            "orb-normal-right",
            "orb-mirror-left",
            "orb-mirror-right",
        ]:
            if orb_name not in self.matched_orbs:
                orb_list = self.current_level.get_sprites_by_name(orb_name)
                if orb_list:
                    active_positions.append(orb_list[0])

        for color_suffix in ["green", "orange", "purple"]:

            sensor_list = self.current_level.get_sprites_by_name(f"sensor-{color_suffix}")
            barrier_list = self.current_level.get_sprites_by_name(f"barrier-{color_suffix}")
            if not sensor_list or not barrier_list:
                continue
            sensor_triggered = False
            for sensor_spr in sensor_list:
                for orb_spr in active_positions:
                    if orb_spr.x == sensor_spr.x and orb_spr.y == sensor_spr.y:
                        sensor_triggered = True
                        break
                if sensor_triggered:
                    break
            for barrier_spr in barrier_list:
                if sensor_triggered:
                    barrier_spr.set_interaction(InteractionMode.REMOVED)
                else:
                    barrier_spr.set_interaction(InteractionMode.TANGIBLE)


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

    @property
    def action_log(self) -> list[str]:
        """"""
        agg, _ = self._log_state.get_action_log()
        return agg
    @action_log.setter
    def action_log(self, value: list[str]) -> None:


        pass

    @property
    def action_log_l2(self) -> list[str]:
        agg, agg_l2 = self._log_state.get_action_log()
        return agg_l2
    @action_log_l2.setter
    def action_log_l2(self, value: list[str]) -> None:
        pass


    def _log(self, en: str, l2: str) -> None:
        """

        """
        self._log_state.log(en, l2)

    def _set_event(self, key: str, value) -> None:
        """

        """
        self._log_state.set_event(key, value)


    def reset_logs(self) -> None:
        """"""
        self._log_state.reset()

    def get_action_log_frame(self) -> tuple[list[list[str]], list[list[str]]]:
        """"""
        return self._log_state.get_frame_logs()

    def get_action_log(self) -> tuple[list[str], list[str]]:
        """

        """
        return self._log_state.get_action_log()


    def _compose_summary(
        self,
        agg: list[str],
        agg_l2: list[str],
        events: dict,
    ) -> tuple[list[str], list[str]]:
        """



        """
        from gamesofts._action_log import compose_default_summary
        agg, agg_l2 = compose_default_summary(agg, agg_l2, events)

        flash = events.get("flash")
        if flash:

            if "orbs" in flash:
                orbs = flash["orbs"]
            else:
                orbs = [{"name": flash["orb_name"], "pos_grid": flash["orb_pos_grid"]}]
            arc   = min(64 // self.camera.width, 64 // self.camera.height)
            count = len(orbs)
            for orb in orbs:
                gx, gy = orb["pos_grid"]
                name   = orb["name"]
                dx, dy = self._to_display_xy(gx, gy)
                _t, _le, _l2 = self._display_for(name)


                agg.append(
                    f"[bright]:[orb]:[sname({name});position({dx}_{dy});bbox({arc}_{arc});"
                    f"state(flash);count({count})]:"
                    f"{_le} flashes at ({dx},{dy})."
                    f"|Object flashes at ({dx},{dy})."
                )
                agg_l2.append(
                    f"[bright]:[orb]:[sname({name});position({dx}_{dy});bbox({arc}_{arc});"
                    f"state(flash);count({count})]:"
                    f"({dx},{dy}){_l2}"
                    f"|({dx},{dy})Object"
                )





        overlaps = events.get("orb_pair_overlap")
        if overlaps:
            for pair in overlaps:
                a_name = pair["a"]
                b_name = pair["b"]
                dx, dy = pair["dx"], pair["dy"]
                dw, dh = pair["dw"], pair["dh"]
                _at, _ale, _al2 = self._display_for(a_name)
                _bt, _ble, _bl2 = self._display_for(b_name)
                agg.append(
                    f"[overlap]:[{_at}]:[sname({a_name});ssub({b_name});position({dx}_{dy});bbox({dw}_{dh})]:"
                    f"{_ale} overlaps {_ble} at ({dx},{dy})."
                    f"|Object overlaps Object at ({dx},{dy})."
                )
                agg_l2.append(
                    f"[overlap]:[{_at}]:[sname({a_name});ssub({b_name});position({dx}_{dy});bbox({dw}_{dh})]:"
                    f"({dx},{dy}){_al2}{_bl2}"
                    f"|({dx},{dy})ObjectObject"
                )
        return agg, agg_l2


    @staticmethod
    def _aggregate_logs(frame_logs: list[list[str]], is_l2: bool = False) -> list[str]:
        """









        """
        import re
        _RE_MOVE_V2 = re.compile(
            r'(\[.+?\]):\[(.+?)\]:\[[^\]]*?position\((-?\d+)_(-?\d+)\)_\((-?\d+)_(-?\d+)\)[^\]]*\]:(.*)'
        )
        _RE_POS_SUB = re.compile(
            r'(position)\((-?\d+)_(-?\d+)\)_\((-?\d+)_(-?\d+)\)'
        )
        _RE_CHANGE_COLOR = re.compile(
            r'^(\[change\]):\[(.+?)\]:\[(.*?)color\((\w+)\)_\((\w+)\)(.*?)\]:(.*)$'
        )
        _RE_POS_IN_PARAMS = re.compile(r'position\(([^)]+)\)')
        _DIR_EN = {( 1,  0): "right", (-1,  0): "left",
                   ( 0,  1): "down",  ( 0, -1): "up"}
        _DIR_L2 = {( 1,  0): "",   (-1,  0): "",
                   ( 0,  1): "",   ( 0, -1): ""}
        _NON_CHAIN_VERBS = {"[extend]", "[shrink]"}

        def _sign(n: int) -> int:
            return 0 if n == 0 else (1 if n > 0 else -1)

        def _make_move(name: str, x1: int, y1: int, x2: int, y2: int, template: str) -> str:
            dx, dy = x2 - x1, y2 - y1
            dk = (_sign(dx), _sign(dy))
            m = _RE_MOVE_V2.fullmatch(template)
            if m is not None:
                head, _, rest = template.partition(":")
                rest_subj, _, rest = rest.partition(":")
                params_with_brackets, _, _ = rest.partition(":")
                new_params = _RE_POS_SUB.sub(
                    lambda mm: f"{mm.group(1)}({x1}_{y1})_({x2}_{y2})",
                    params_with_brackets,
                )
                if is_l2:
                    dir_str = _DIR_L2.get(dk, f"({dx:+d},{dy:+d})")
                    desc = f"{name}({x1},{y1})({x2},{y2}){dir_str}"
                else:
                    dir_str = _DIR_EN.get(dk, f"({dx:+d},{dy:+d})")
                    desc = f"{name} moves {dir_str} from ({x1},{y1}) to ({x2},{y2})."
                return f"{head}:{rest_subj}:{new_params}:{desc}"

            if is_l2:
                dir_str = _DIR_L2.get(dk, f"({dx:+d},{dy:+d})")
                return f"[move]:[{name}]:[position({x1}_{y1})_({x2}_{y2})]:{name}({x1},{y1})({x2},{y2}){dir_str}"
            dir_str = _DIR_EN.get(dk, f"({dx:+d},{dy:+d})")
            return f"[move]:[{name}]:[position({x1}_{y1})_({x2}_{y2})]:{name} moves {dir_str} from ({x1},{y1}) to ({x2},{y2})."


        class _OpenSegment:
            __slots__ = ("name", "x1", "y1", "x2", "y2",
                         "prev_dx", "prev_dy", "orig", "count")
            def __init__(self, name, x1, y1, x2, y2, orig):
                self.name = name
                self.x1, self.y1 = x1, y1
                self.x2, self.y2 = x2, y2
                self.prev_dx = x2 - x1
                self.prev_dy = y2 - y1
                self.orig = orig
                self.count = 1

        open_segments: dict[str, list[_OpenSegment]] = {}
        all_segments: list[_OpenSegment] = []
        misc_seen:    set[str]  = set()
        misc_order:   list[str] = []
        _extent_key_idx: dict[tuple[str, str], int] = {}

        for frame in frame_logs:
            for entry in frame:
                m = _RE_MOVE_V2.fullmatch(entry)
                if not m or m.group(1) in _NON_CHAIN_VERBS:

                    _es_match = re.match(r'(\[(?:extend|shrink)\]):\[(.+?)\]:', entry)
                    if _es_match:
                        _key = (_es_match.group(1), _es_match.group(2))
                        if _key in _extent_key_idx:
                            misc_order[_extent_key_idx[_key]] = entry
                        else:
                            _extent_key_idx[_key] = len(misc_order)
                            misc_order.append(entry)
                    elif entry not in misc_seen:
                        misc_seen.add(entry)
                        misc_order.append(entry)
                    continue
                name = m.group(2)
                x1, y1 = int(m.group(3)), int(m.group(4))
                x2, y2 = int(m.group(5)), int(m.group(6))
                dx, dy = x2 - x1, y2 - y1
                bucket = open_segments.setdefault(name, [])
                match: _OpenSegment | None = None
                match_idx = -1
                for i, seg in enumerate(bucket):
                    if (seg.x2, seg.y2) == (x1, y1):
                        match = seg
                        match_idx = i
                        break
                if match is None:
                    new_seg = _OpenSegment(name, x1, y1, x2, y2, entry)
                    bucket.append(new_seg)
                    all_segments.append(new_seg)
                    continue
                prev_dir = (_sign(match.prev_dx), _sign(match.prev_dy))
                cur_dir  = (_sign(dx), _sign(dy))
                dir_change = (
                    cur_dir != (0, 0)
                    and prev_dir != (0, 0)
                    and cur_dir != prev_dir
                )
                if dir_change:
                    bucket.pop(match_idx)
                    new_seg = _OpenSegment(name, x1, y1, x2, y2, entry)
                    bucket.append(new_seg)
                    all_segments.append(new_seg)
                else:
                    match.x2, match.y2 = x2, y2
                    if dx != 0: match.prev_dx = dx
                    if dy != 0: match.prev_dy = dy
                    match.count += 1

        result: list[str] = []
        for seg in all_segments:


            if seg.count == 1:
                result.append(seg.orig)
            else:
                result.append(_make_move(seg.name, seg.x1, seg.y1, seg.x2, seg.y2, seg.orig))


        color_chain_idx: dict[tuple[str, str], int] = {}
        color_chain_last: dict[tuple[str, str], str] = {}
        color_chain_first: dict[tuple[str, str], str] = {}
        last_pair_per_key: dict[tuple[str, str], tuple[str, str]] = {}
        _suppress_idx: set[int] = set()
        for idx, entry in enumerate(misc_order):
            cm = _RE_CHANGE_COLOR.match(entry)
            if not cm:
                continue
            verb, target, params_pre, from_c, to_c, params_post, desc = cm.groups()
            params_full = params_pre + f"color({from_c})_({to_c})" + params_post
            pos_m = _RE_POS_IN_PARAMS.search(params_full)
            pos_key = pos_m.group(1) if pos_m else ""
            key = (target, pos_key)
            if last_pair_per_key.get(key) == (from_c, to_c):

                color_chain_idx.pop(key, None)
                color_chain_first.pop(key, None)
                color_chain_last.pop(key, None)
                continue
            if key in color_chain_idx and color_chain_last.get(key) == from_c:
                first_c = color_chain_first[key]
                merged_params = params_pre + f"color({first_c})_({to_c})" + params_post
                if is_l2:
                    merged_desc = f"{target}{first_c}{to_c}"
                else:
                    merged_desc = f"{target} gradually changes color from {first_c} to {to_c}."
                merged = f"{verb}:[{target}]:[{merged_params}]:{merged_desc}"
                existing_idx = color_chain_idx[key]
                misc_order[existing_idx] = merged
                _suppress_idx.add(idx)
                color_chain_last[key] = to_c
            else:
                color_chain_idx[key] = idx
                color_chain_first[key] = from_c
                color_chain_last[key] = to_c
            last_pair_per_key[key] = (from_c, to_c)

        for key, idx in color_chain_idx.items():
            if color_chain_first.get(key) == color_chain_last.get(key):
                _suppress_idx.add(idx)
        if _suppress_idx:
            misc_order = [e for i, e in enumerate(misc_order) if i not in _suppress_idx]

        result.extend(misc_order)
        return result



    def _display_for(self, name: str) -> tuple[str, str, str]:
        """




        """
        label_en = TAG_DISPLAY_NAMES.get(name)
        label_l2 = TAG_DISPLAY_NAMES_L2.get(name)
        if label_en is not None and label_l2 is not None:

            target = (label_en.lower()
                      .replace(" ", "_")
                      .replace("(", "").replace(")", "")
                      .replace("-", "_"))

            while "__" in target:
                target = target.replace("__", "_")
            return target.strip("_"), label_en, label_l2
        return (name, name, name)

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
        max_steps = self.tmbjb.max_steps
        if max_steps == 0:
            return 0
        ratio = value / max_steps
        filled = round(64 * ratio)
        return min(filled, 64)



    def _log_diffs(self, _before_sprites: dict, _before_action_count: int) -> None:
        """




        """
        after_sprites = {id(spr): spr for spr in self.current_level.get_sprites()}


        for spr_id, spr in after_sprites.items():
            b = _before_sprites.get(spr_id)
            if b is None or (not b["visible"] and spr.is_visible):
                dx, dy, dw, dh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(spr.name)
                self._step_log.append(f"[appear]:[{_t}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{_le} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).")
                self._step_log_l2.append(f"[appear]:[{_t}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{_l2}({dx},{dy})|Object({dx},{dy})")



        _arc_scale_d = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in _before_sprites.items():
            spr = after_sprites.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = int(b.get("width", 0))  * _arc_scale_d
                bdh = int(b.get("height", 0)) * _arc_scale_d
                _t, _le, _l2 = self._display_for(b['name'])
                self._step_log.append(f"[disappear]:[{_t}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{_le} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).")
                self._step_log_l2.append(f"[disappear]:[{_t}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{_l2}({bdx},{bdy})|Object({bdx},{bdy})")


        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            b = _before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                continue
            if (spr.x, spr.y) != (b["x"], b["y"]):
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                adx, ady, adw, adh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(spr.name)
                self._step_log.append(f"[move]:[{_t}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{_le} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady}).")
                self._step_log_l2.append(f"[move]:[{spr.name}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{_l2}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})")
            if spr.rotation != b["rotation"]:
                _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
                _t, _le, _l2 = self._display_for(spr.name)
                self._step_log.append(f"[rotate]:[{_t}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:{_le} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.")
                self._step_log_l2.append(f"[rotate]:[{_t}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:({_rdx},{_rdy}){_l2}{b['rotation']}°{spr.rotation}°|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°")



            cur_repr = int(spr.pixels[1, 1]) if (spr.pixels.shape[0] > 1 and spr.pixels.shape[1] > 1) else int(spr.pixels.flat[0])
            if cur_repr != b["pixels_repr"] and cur_repr >= 0 and b["pixels_repr"] >= 0:
                before_name    = COLOR_PALETTE.get(b["pixels_repr"], {}).get("name", str(b["pixels_repr"])).lower()
                after_name     = COLOR_PALETTE.get(cur_repr,           {}).get("name", str(cur_repr)).lower()
                before_name_l2 = COLOR_PALETTE.get(b["pixels_repr"], {}).get("label_l2", str(b["pixels_repr"]))
                after_name_l2  = COLOR_PALETTE.get(cur_repr,           {}).get("label_l2", str(cur_repr))
                cdx, cdy, cdw, cdh = self._to_display_bbox(spr)





                already = any(
                    (f"color({before_name})_({after_name})" in line and f"sname({spr.name})" in line)
                    for line in self._step_log
                )
                if not already:
                    _t, _le, _l2 = self._display_for(spr.name)
                    self._step_log.append(
                        f"[change]:[{_t}]:[sname({spr.name});color({before_name})_({after_name});position({cdx}_{cdy});bbox({cdw}_{cdh})]:"
                        f"{_le} at ({cdx},{cdy}) changes color from {before_name} to {after_name}."
                        f"|Object at ({cdx},{cdy}) changes color from {before_name} to {after_name}."
                    )
                    self._step_log_l2.append(
                        f"[change]:[{_t}]:[sname({spr.name});color({before_name})_({after_name});position({cdx}_{cdy});bbox({cdw}_{cdh})]:"
                        f"({cdx},{cdy}){_l2}{before_name_l2}{after_name_l2}"
                        f"|({cdx},{cdy})Object{before_name_l2}{after_name_l2}"
                    )




        after_action_count = self._action_count
        before_for_log = self._last_logged_action_count
        if after_action_count != before_for_log:
            before_steps = max(0, 150 - before_for_log)
            after_steps  = max(0, 150 - after_action_count)
            if after_steps != before_steps and self._bar_pixel_len(before_steps) != self._bar_pixel_len(after_steps):
                self._last_logged_action_count = after_action_count


                if after_steps < before_steps:

                    self._step_log.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:The black step bar at the top shrinks from the right ({before_steps}→{after_steps} steps remain).|The black Object at the top shrinks from the right ({before_steps}→{after_steps}).")
                    self._step_log_l2.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:{before_steps}→{after_steps}|Object{before_steps}→{after_steps}")

                    self._step_log.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:The black step bar at the bottom shrinks from the left ({before_steps}→{after_steps} steps remain).|The black Object at the bottom shrinks from the left ({before_steps}→{after_steps}).")
                    self._step_log_l2.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:{before_steps}→{after_steps}|Object{before_steps}→{after_steps}")
                else:
                    self._step_log.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:The black step bar at the top extends to the right ({before_steps}→{after_steps} steps).|The black Object at the top extends to the right ({before_steps}→{after_steps}).")
                    self._step_log_l2.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:{before_steps}→{after_steps}|Object{before_steps}→{after_steps}")
                    self._step_log.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:The black step bar at the bottom extends to the left ({before_steps}→{after_steps} steps).|The black Object at the bottom extends to the left ({before_steps}→{after_steps}).")
                    self._step_log_l2.append(f"[change]:[steps]:[sname(steps_remaining);count({before_steps})_({after_steps})]:{before_steps}→{after_steps}|Object{before_steps}→{after_steps}")



    def _append_frame_logs(self) -> None:
        """
        """
        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()



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
                TAG_DISPLAY_NAMES.get(spr.name, spr.name),
            )
            label_l2 = next(
                (TAG_DISPLAY_NAMES_L2[t] for t in tags if t in TAG_DISPLAY_NAMES_L2),
                TAG_DISPLAY_NAMES_L2.get(spr.name, spr.name),
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
            "label_l2": ''                  ,
            "value":    self.tmbjb.current_steps,
            "max":      self.tmbjb.max_steps,
            "x":        0, "y":      0,
            "width":    64, "height": 1,
        })

        result.append({
            "name":     "matched_orbs",
            "label":    "Matched Orb Pairs",
            "label_l2": ''                           ,
            "value":    len(self.matched_orbs),
            "max":      0,
            "x":        0, "y": 0,
            "width":    0, "height": 0,
        })
        return result

    def get_game_knowledge(self) -> dict:
        """


        """
        return GAME_KNOWLEDGE
