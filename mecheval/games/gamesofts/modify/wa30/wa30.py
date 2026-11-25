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

    "background_64x20_a": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="background_64x20_a",
        visible=True,
        collidable=True,
    ),

    "orange_agent": Sprite(
        pixels=[
            [12, 12, 12, 12],
            [12, 12, 12, 12],
            [12, 12, 12, 12],
            [12, 12, 12, 12],
        ],
        name="orange_agent",
        visible=True,
        collidable=True,
        tags=["blue_agent"],
        layer=1,
    ),

    "background_64x16": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="background_64x16",
        visible=True,
        collidable=True,
    ),


    "target_8x12": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="target_8x12",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),


    "floor_area_4x8": Sprite(
        pixels=[
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ],
        name="floor_area_4x8",
        visible=True,
        collidable=False,
        tags=["walkable_zone"],
    ),

    "target_8x8": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="target_8x8",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),

    "target_12x12": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="target_12x12",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),


    "floor_area_8x4": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="floor_area_8x4",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),


    "walkable_zone_8x8": Sprite(
        pixels=[
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
        ],
        name="walkable_zone_8x8",
        visible=True,
        collidable=False,
        tags=["walkable_zone"],
    ),

    "purple_agent": Sprite(
        pixels=[
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
        ],
        name="purple_agent",
        visible=True,
        collidable=True,
        tags=["purple_agent"],
        layer=1,
    ),

    "target_8x16": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="target_8x16",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),



    "wall_block": Sprite(
        pixels=[
            [4, 4, 4, 4],
            [4, 9, 9, 4],
            [4, 9, 9, 4],
            [4, 4, 4, 4],
        ],
        name="wall_block",
        visible=True,
        collidable=True,
        tags=["target"],
        layer=1,
    ),

    "hazard": Sprite(
        pixels=[
            [2, -2, 2, 2],
            [-2, 2, 2, 2],
            [2, 2, 2, -2],
            [2, 2, -2, 2],
        ],
        name="hazard",
        visible=True,
        collidable=False,
        tags=["hazard"],
    ),

    "obstacle_block": Sprite(
        pixels=[
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
        ],
        name="obstacle_block",
        visible=True,
        collidable=True,
        tags=["obstacle_block"],
    ),

    "target_16x12": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="target_16x12",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),

    "target_8x4": Sprite(
        pixels=[
            [9, 9, 9, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 9, 9, 9],
        ],
        name="target_8x4",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),

    "walkable_zone_12x8": Sprite(
        pixels=[
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
        ],
        name="walkable_zone_12x8",
        visible=True,
        collidable=False,
        tags=["walkable_zone"],
    ),

    "target_12x12b": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="target_12x12b",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),

    "selector": Sprite(
        pixels=[
            [0, 0, 0, 0],
            [14, 14, 14, 14],
            [14, 14, 14, 14],
            [14, 14, 14, 14],
        ],
        name="selector",
        visible=True,
        collidable=True,
        tags=["selector"],
        layer=1,
    ),

    "walkable_zone_12x12": Sprite(
        pixels=[
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ],
        name="walkable_zone_12x12",
        visible=True,
        collidable=False,
        tags=["walkable_zone"],
    ),

    "target_4x4": Sprite(
        pixels=[
            [9, 9, 9, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 9, 9, 9],
        ],
        name="target_4x4",
        visible=True,
        collidable=False,
        tags=["floor_area"],
    ),
}




levels = [

    Level(
        sprites=[
            sprites["target_12x12"].clone().set_position(28, 28),
            sprites["wall_block"].clone().set_position(44, 24),
            sprites["wall_block"].clone().set_position(16, 28),
            sprites["wall_block"].clone().set_position(32, 36),
            sprites["selector"].clone().set_position(32, 48),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 200,
        },
    ),

    Level(
        sprites=[
            sprites["orange_agent"].clone().set_position(24, 36),
            sprites["target_8x12"].clone().set_position(12, 28),
            sprites["wall_block"].clone().set_position(48, 32),
            sprites["wall_block"].clone().set_position(36, 28),
            sprites["wall_block"].clone().set_position(40, 20),
            sprites["wall_block"].clone().set_position(48, 24),
            sprites["wall_block"].clone().set_position(44, 40),
            sprites["selector"].clone().set_position(12, 8),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 70,
        },
    ),

    Level(
        sprites=[
            sprites["orange_agent"].clone().set_position(48, 12),
            sprites["target_8x16"].clone().set_position(52, 24),
            sprites["wall_block"].clone().set_position(32, 32),
            sprites["wall_block"].clone().set_position(20, 20),
            sprites["wall_block"].clone().set_position(12, 44),
            sprites["wall_block"].clone().set_position(8, 16),
            sprites["wall_block"].clone().set_position(32, 12),
            sprites["hazard"].clone().set_position(32, 0),
            sprites["hazard"].clone().set_position(32, 4),
            sprites["hazard"].clone().set_position(32, 8),
            sprites["hazard"].clone().set_position(32, 12),
            sprites["hazard"].clone().set_position(32, 16),
            sprites["hazard"].clone().set_position(32, 20),
            sprites["hazard"].clone().set_position(32, 24),
            sprites["hazard"].clone().set_position(32, 28),
            sprites["hazard"].clone().set_position(32, 32),
            sprites["hazard"].clone().set_position(32, 36),
            sprites["hazard"].clone().set_position(32, 40),
            sprites["hazard"].clone().set_position(32, 44),
            sprites["hazard"].clone().set_position(32, 48),
            sprites["hazard"].clone().set_position(32, 52),
            sprites["hazard"].clone().set_position(32, 56),
            sprites["hazard"].clone().set_position(32, 60),
            sprites["selector"].clone().set_position(16, 36),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 100,
        },
    ),

    Level(
        sprites=[
            sprites["orange_agent"].clone().set_position(56, 4),
            sprites["orange_agent"].clone().set_position(8, 12),
            sprites["orange_agent"].clone().set_position(24, 56),
            sprites["target_8x4"].clone().set_position(4, 24),
            sprites["wall_block"].clone().set_position(32, 36),
            sprites["wall_block"].clone().set_position(24, 24),
            sprites["wall_block"].clone().set_position(36, 40),
            sprites["wall_block"].clone().set_position(24, 40),
            sprites["wall_block"].clone().set_position(32, 24),
            sprites["wall_block"].clone().set_position(36, 24),
            sprites["wall_block"].clone().set_position(24, 4),
            sprites["hazard"].clone().set_position(28, 20),
            sprites["hazard"].clone().set_position(24, 20),
            sprites["hazard"].clone().set_position(20, 20),
            sprites["hazard"].clone().set_position(20, 24),
            sprites["hazard"].clone().set_position(20, 28),
            sprites["hazard"].clone().set_position(20, 32),
            sprites["hazard"].clone().set_position(20, 36),
            sprites["hazard"].clone().set_position(20, 40),
            sprites["hazard"].clone().set_position(20, 44),
            sprites["hazard"].clone().set_position(32, 20),
            sprites["hazard"].clone().set_position(36, 20),
            sprites["hazard"].clone().set_position(40, 20),
            sprites["hazard"].clone().set_position(40, 24),
            sprites["hazard"].clone().set_position(24, 44),
            sprites["hazard"].clone().set_position(28, 44),
            sprites["hazard"].clone().set_position(32, 44),
            sprites["hazard"].clone().set_position(36, 44),
            sprites["hazard"].clone().set_position(40, 44),
            sprites["hazard"].clone().set_position(40, 40),
            sprites["hazard"].clone().set_position(40, 36),
            sprites["hazard"].clone().set_position(40, 32),
            sprites["hazard"].clone().set_position(40, 28),
            sprites["obstacle_block"].clone().set_position(16, 48),
            sprites["obstacle_block"].clone().set_position(12, 52),
            sprites["obstacle_block"].clone().set_position(8, 56),
            sprites["obstacle_block"].clone().set_position(4, 60),
            sprites["obstacle_block"].clone().set_position(44, 48),
            sprites["obstacle_block"].clone().set_position(48, 52),
            sprites["obstacle_block"].clone().set_position(52, 56),
            sprites["obstacle_block"].clone().set_position(56, 60),
            sprites["obstacle_block"].clone().set_position(28, 4),
            sprites["obstacle_block"].clone().set_position(28, 0),
            sprites["obstacle_block"].clone().set_position(28, 8),
            sprites["obstacle_block"].clone().set_position(28, 12),
            sprites["obstacle_block"].clone().set_position(28, 16),
            sprites["floor_area_8x4"].clone().set_position(36, 56),
            sprites["selector"].clone().set_position(28, 32),
            sprites["target_4x4"].clone().set_position(8, 36),
            sprites["target_4x4"].clone().set_position(52, 28),
            sprites["target_4x4"].clone().set_position(56, 20),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 100,
        },
    ),

    Level(
        sprites=[
            sprites["orange_agent"].clone().set_position(20, 28),
            sprites["target_8x16"].clone().set_position(8, 24),
            sprites["wall_block"].clone().set_position(52, 48),
            sprites["wall_block"].clone().set_position(60, 52),
            sprites["wall_block"].clone().set_position(48, 4),
            sprites["wall_block"].clone().set_position(56, 8),
            sprites["wall_block"].clone().set_position(44, 56),
            sprites["wall_block"].clone().set_position(44, 28),
            sprites["obstacle_block"].clone().set_position(24, 24),
            sprites["obstacle_block"].clone().set_position(28, 24),
            sprites["obstacle_block"].clone().set_position(32, 24),
            sprites["obstacle_block"].clone().set_position(36, 24),
            sprites["obstacle_block"].clone().set_position(36, 20),
            sprites["obstacle_block"].clone().set_position(36, 16),
            sprites["obstacle_block"].clone().set_position(36, 12),
            sprites["obstacle_block"].clone().set_position(36, 8),
            sprites["obstacle_block"].clone().set_position(36, 4),
            sprites["obstacle_block"].clone().set_position(36, 0),
            sprites["obstacle_block"].clone().set_position(36, 60),
            sprites["obstacle_block"].clone().set_position(36, 56),
            sprites["obstacle_block"].clone().set_position(36, 52),
            sprites["obstacle_block"].clone().set_position(36, 48),
            sprites["obstacle_block"].clone().set_position(36, 44),
            sprites["obstacle_block"].clone().set_position(36, 40),
            sprites["obstacle_block"].clone().set_position(36, 36),
            sprites["obstacle_block"].clone().set_position(24, 36),
            sprites["obstacle_block"].clone().set_position(28, 36),
            sprites["obstacle_block"].clone().set_position(32, 36),
            sprites["selector"].clone().set_position(44, 36),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 125,
        },
    ),

    Level(
        sprites=[
            sprites["target_8x8"].clone().set_position(28, 12),
            sprites["purple_agent"].clone().set_position(16, 16),
            sprites["walkable_zone_8x8"].clone().set_position(52, 24),
            sprites["wall_block"].clone().set_position(56, 28),
            sprites["wall_block"].clone().set_position(28, 16),
            sprites["obstacle_block"].clone().set_position(44, 0),
            sprites["obstacle_block"].clone().set_position(44, 4),
            sprites["obstacle_block"].clone().set_position(44, 8),
            sprites["obstacle_block"].clone().set_position(44, 12),
            sprites["obstacle_block"].clone().set_position(44, 16),
            sprites["obstacle_block"].clone().set_position(44, 20),
            sprites["obstacle_block"].clone().set_position(48, 0),
            sprites["obstacle_block"].clone().set_position(48, 4),
            sprites["obstacle_block"].clone().set_position(48, 8),
            sprites["obstacle_block"].clone().set_position(48, 12),
            sprites["obstacle_block"].clone().set_position(48, 16),
            sprites["obstacle_block"].clone().set_position(48, 20),
            sprites["obstacle_block"].clone().set_position(44, 28),
            sprites["obstacle_block"].clone().set_position(44, 32),
            sprites["obstacle_block"].clone().set_position(44, 36),
            sprites["obstacle_block"].clone().set_position(44, 40),
            sprites["obstacle_block"].clone().set_position(44, 44),
            sprites["obstacle_block"].clone().set_position(44, 48),
            sprites["obstacle_block"].clone().set_position(48, 28),
            sprites["obstacle_block"].clone().set_position(48, 32),
            sprites["obstacle_block"].clone().set_position(48, 36),
            sprites["obstacle_block"].clone().set_position(48, 40),
            sprites["obstacle_block"].clone().set_position(48, 44),
            sprites["obstacle_block"].clone().set_position(48, 48),
            sprites["obstacle_block"].clone().set_position(44, 52),
            sprites["obstacle_block"].clone().set_position(44, 56),
            sprites["obstacle_block"].clone().set_position(48, 52),
            sprites["obstacle_block"].clone().set_position(48, 56),
            sprites["obstacle_block"].clone().set_position(44, 60),
            sprites["obstacle_block"].clone().set_position(48, 60),
            sprites["selector"].clone().set_position(20, 52),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 75,
        },
    ),

    Level(
        sprites=[
            sprites["background_64x20_a"].clone().set_position(0, 44),
            sprites["background_64x16"].clone(),
            sprites["floor_area_4x8"].clone().set_position(48, 28),
            sprites["purple_agent"].clone().set_position(40, 32),
            sprites["target_8x4"].clone().set_position(12, 28),
            sprites["wall_block"].clone().set_position(32, 24),
            sprites["wall_block"].clone().set_position(28, 32),
            sprites["obstacle_block"].clone().set_position(0, 16),
            sprites["obstacle_block"].clone().set_position(4, 16),
            sprites["obstacle_block"].clone().set_position(8, 16),
            sprites["obstacle_block"].clone().set_position(12, 16),
            sprites["obstacle_block"].clone().set_position(16, 16),
            sprites["obstacle_block"].clone().set_position(20, 16),
            sprites["obstacle_block"].clone().set_position(24, 16),
            sprites["obstacle_block"].clone().set_position(28, 16),
            sprites["obstacle_block"].clone().set_position(32, 16),
            sprites["obstacle_block"].clone().set_position(36, 16),
            sprites["obstacle_block"].clone().set_position(40, 16),
            sprites["obstacle_block"].clone().set_position(44, 16),
            sprites["obstacle_block"].clone().set_position(48, 16),
            sprites["obstacle_block"].clone().set_position(52, 16),
            sprites["obstacle_block"].clone().set_position(56, 16),
            sprites["obstacle_block"].clone().set_position(60, 16),
            sprites["obstacle_block"].clone().set_position(0, 40),
            sprites["obstacle_block"].clone().set_position(4, 40),
            sprites["obstacle_block"].clone().set_position(8, 40),
            sprites["obstacle_block"].clone().set_position(12, 40),
            sprites["obstacle_block"].clone().set_position(16, 40),
            sprites["obstacle_block"].clone().set_position(20, 40),
            sprites["obstacle_block"].clone().set_position(24, 40),
            sprites["obstacle_block"].clone().set_position(28, 40),
            sprites["obstacle_block"].clone().set_position(32, 40),
            sprites["obstacle_block"].clone().set_position(36, 40),
            sprites["obstacle_block"].clone().set_position(40, 40),
            sprites["obstacle_block"].clone().set_position(44, 40),
            sprites["obstacle_block"].clone().set_position(48, 40),
            sprites["obstacle_block"].clone().set_position(52, 40),
            sprites["obstacle_block"].clone().set_position(56, 40),
            sprites["obstacle_block"].clone().set_position(60, 40),
            sprites["selector"].clone().set_position(20, 32),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 125,
        },
    ),

    Level(
        sprites=[
            sprites["orange_agent"].clone().set_position(28, 4),
            sprites["orange_agent"].clone().set_position(36, 44),
            sprites["purple_agent"].clone().set_position(32, 16),
            sprites["purple_agent"].clone().set_position(32, 56),
            sprites["target_12x12b"].clone().set_position(48, 48),
            sprites["wall_block"].clone().set_position(32, 12),
            sprites["wall_block"].clone().set_position(28, 8),
            sprites["wall_block"].clone().set_position(40, 52),
            sprites["wall_block"].clone().set_position(36, 48),
            sprites["wall_block"].clone().set_position(28, 52),
            sprites["wall_block"].clone().set_position(32, 52),
            sprites["wall_block"].clone().set_position(36, 8),
            sprites["wall_block"].clone().set_position(24, 12),
            sprites["wall_block"].clone().set_position(4, 8),
            sprites["wall_block"].clone().set_position(8, 12),
            sprites["wall_block"].clone().set_position(8, 8),
            sprites["wall_block"].clone().set_position(4, 12),
            sprites["wall_block"].clone().set_position(24, 44),
            sprites["obstacle_block"].clone().set_position(24, 24),
            sprites["obstacle_block"].clone().set_position(28, 24),
            sprites["obstacle_block"].clone().set_position(8, 24),
            sprites["obstacle_block"].clone().set_position(12, 24),
            sprites["obstacle_block"].clone().set_position(0, 24),
            sprites["obstacle_block"].clone().set_position(4, 24),
            sprites["obstacle_block"].clone().set_position(32, 24),
            sprites["obstacle_block"].clone().set_position(36, 24),
            sprites["obstacle_block"].clone().set_position(40, 24),
            sprites["obstacle_block"].clone().set_position(44, 24),
            sprites["obstacle_block"].clone().set_position(48, 24),
            sprites["obstacle_block"].clone().set_position(52, 24),
            sprites["obstacle_block"].clone().set_position(56, 24),
            sprites["obstacle_block"].clone().set_position(60, 24),
            sprites["obstacle_block"].clone().set_position(44, 36),
            sprites["obstacle_block"].clone().set_position(48, 36),
            sprites["obstacle_block"].clone().set_position(52, 36),
            sprites["obstacle_block"].clone().set_position(56, 36),
            sprites["obstacle_block"].clone().set_position(60, 36),
            sprites["obstacle_block"].clone().set_position(0, 36),
            sprites["obstacle_block"].clone().set_position(4, 36),
            sprites["obstacle_block"].clone().set_position(8, 36),
            sprites["obstacle_block"].clone().set_position(12, 36),
            sprites["obstacle_block"].clone().set_position(16, 36),
            sprites["obstacle_block"].clone().set_position(20, 36),
            sprites["obstacle_block"].clone().set_position(24, 36),
            sprites["obstacle_block"].clone().set_position(28, 36),
            sprites["obstacle_block"].clone().set_position(32, 36),
            sprites["target_16x12"].clone().set_position(44, 8),
            sprites["selector"].clone().set_position(4, 32),
            sprites["walkable_zone_12x12"].clone().set_position(4, 8),
            sprites["walkable_zone_12x12"].clone().set_position(12, 48),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 150,
        },
    ),

    Level(
        sprites=[
            sprites["orange_agent"].clone().set_position(16, 28),
            sprites["orange_agent"].clone().set_position(44, 4),
            sprites["purple_agent"].clone().set_position(60, 56),
            sprites["walkable_zone_8x8"].clone().set_position(4, 28),
            sprites["target_12x12b"].clone().set_position(20, 12),
            sprites["wall_block"].clone().set_position(12, 20),
            sprites["wall_block"].clone().set_position(8, 28),
            sprites["wall_block"].clone().set_position(4, 20),
            sprites["wall_block"].clone().set_position(4, 32),
            sprites["wall_block"].clone().set_position(56, 32),
            sprites["wall_block"].clone().set_position(44, 20),
            sprites["wall_block"].clone().set_position(8, 12),
            sprites["wall_block"].clone().set_position(48, 28),
            sprites["wall_block"].clone().set_position(4, 28),
            sprites["hazard"].clone().set_position(52, 12),
            sprites["hazard"].clone().set_position(40, 12),
            sprites["hazard"].clone().set_position(56, 12),
            sprites["hazard"].clone().set_position(60, 12),
            sprites["hazard"].clone().set_position(48, 12),
            sprites["hazard"].clone().set_position(44, 12),
            sprites["obstacle_block"].clone().set_position(40, 40),
            sprites["obstacle_block"].clone().set_position(36, 40),
            sprites["obstacle_block"].clone().set_position(32, 40),
            sprites["obstacle_block"].clone().set_position(56, 40),
            sprites["obstacle_block"].clone().set_position(52, 40),
            sprites["obstacle_block"].clone().set_position(48, 40),
            sprites["obstacle_block"].clone().set_position(44, 40),
            sprites["obstacle_block"].clone().set_position(60, 40),
            sprites["obstacle_block"].clone().set_position(32, 44),
            sprites["obstacle_block"].clone().set_position(32, 48),
            sprites["obstacle_block"].clone().set_position(40, 56),
            sprites["obstacle_block"].clone().set_position(40, 60),
            sprites["obstacle_block"].clone().set_position(48, 44),
            sprites["obstacle_block"].clone().set_position(48, 48),
            sprites["obstacle_block"].clone().set_position(56, 48),
            sprites["obstacle_block"].clone().set_position(56, 52),
            sprites["obstacle_block"].clone().set_position(24, 56),
            sprites["obstacle_block"].clone().set_position(24, 60),
            sprites["obstacle_block"].clone().set_position(24, 52),
            sprites["obstacle_block"].clone().set_position(24, 48),
            sprites["obstacle_block"].clone().set_position(56, 56),
            sprites["obstacle_block"].clone().set_position(56, 60),
            sprites["obstacle_block"].clone().set_position(48, 52),
            sprites["obstacle_block"].clone().set_position(48, 56),
            sprites["obstacle_block"].clone().set_position(40, 48),
            sprites["obstacle_block"].clone().set_position(40, 52),
            sprites["obstacle_block"].clone().set_position(32, 56),
            sprites["obstacle_block"].clone().set_position(32, 52),
            sprites["obstacle_block"].clone().set_position(24, 40),
            sprites["obstacle_block"].clone().set_position(28, 40),
            sprites["obstacle_block"].clone().set_position(16, 40),
            sprites["obstacle_block"].clone().set_position(20, 40),
            sprites["obstacle_block"].clone().set_position(16, 44),
            sprites["obstacle_block"].clone().set_position(16, 48),
            sprites["obstacle_block"].clone().set_position(16, 56),
            sprites["obstacle_block"].clone().set_position(16, 52),
            sprites["obstacle_block"].clone().set_position(36, 0),
            sprites["obstacle_block"].clone().set_position(36, 4),
            sprites["obstacle_block"].clone().set_position(36, 8),
            sprites["obstacle_block"].clone().set_position(36, 16),
            sprites["obstacle_block"].clone().set_position(36, 12),
            sprites["obstacle_block"].clone().set_position(36, 20),
            sprites["obstacle_block"].clone().set_position(36, 24),
            sprites["obstacle_block"].clone().set_position(36, 28),
            sprites["obstacle_block"].clone().set_position(8, 60),
            sprites["obstacle_block"].clone().set_position(8, 56),
            sprites["obstacle_block"].clone().set_position(8, 52),
            sprites["obstacle_block"].clone().set_position(8, 48),
            sprites["obstacle_block"].clone().set_position(8, 44),
            sprites["obstacle_block"].clone().set_position(8, 40),
            sprites["floor_area_8x4"].clone().set_position(52, 24),
            sprites["floor_area_8x4"].clone().set_position(52, 8),
            sprites["selector"].clone().set_position(32, 32),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 70,
        },
    ),
]


BACKGROUND_COLOR = 1
PADDING_COLOR    = 0


GRID_STEP       = 4
COLOR_FREE      = 4
COLOR_NEAR      = 3
COLOR_GRABBED   = 0
COLOR_STOLEN    = 5
COLOR_PURPLE_AT = 15
COLOR_YELLOW    = 11




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
    "blue_agent":    "Blue Agent",
    "purple_agent":  "Purple Agent",
    "selector":      "Selector (Cursor)",
    "target":        "Target",
    "floor_area":    "Floor Area",
    "walkable_zone": "Walkable Zone",
    "hazard":        "Hazard",
    "obstacle_block":"Obstacle Block",
}

TAG_DISPLAY_NAMES_L2: dict[str, str] = {
    "blue_agent":    "",
    "purple_agent":  "",
    "selector":      "",
    "target":        "",
    "floor_area":    "",
    "walkable_zone": "",
    "hazard":        "",
    "obstacle_block":"",
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
            "gimmicks":     ["Single target, few walls — tutorial level"],
            "gimmicks_l2":  [''                                                                     ],
            "subgoals":     ["Move selector to target position", "Guide blue agent to target"],
            "subgoals_l2":  [''                                                   , ''                                                            ],
            "goal_flow":    "Use the selector to intercept the blue agent and steer it onto the target tile.",
            "goal_flow_l2": ''                                                                                                ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Blue agent, tall target, tight step budget"],
            "gimmicks_l2":  [''                                                                        ],
            "subgoals":     ["Position selector to block agent's path", "Funnel agent onto target"],
            "subgoals_l2":  [''                                                      , ''                                                   ],
            "goal_flow":    "Block the blue agent's BFS route with the selector to redirect it to the tall target.",
            "goal_flow_l2": ''                                                                                                            ,
            "difficulty":   "easy",
        },
        {
            "gimmicks":     ["Hazard strip divides the grid, tall target on right side"],
            "gimmicks_l2":  [''                                                                           ],
            "subgoals":     ["Keep agent away from hazards", "Navigate around hazard column to reach target"],
            "subgoals_l2":  [''                                                   , ''                                                         ],
            "goal_flow":    "Guide the blue agent around the hazard strip to reach the tall target on the right half.",
            "goal_flow_l2": ''                                                                                                         ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Multiple blue agents, hazard ring, diagonal obstacle rows"],
            "gimmicks_l2":  [''                                                                              ],
            "subgoals":     ["Grab agents one by one", "Route each through the obstacle maze to its target"],
            "subgoals_l2":  [''                                       , ''                                                                           ],
            "goal_flow":    "Use the selector to pick up each blue agent and route it through the hazard-ringed maze to its matching target.",
            "goal_flow_l2": ''                                                                                                                                                               ,
            "difficulty":   "medium",
        },
        {
            "gimmicks":     ["Blue agent + obstacle cross grid dividing the arena"],
            "gimmicks_l2":  [''                                                                  ],
            "subgoals":     ["Navigate through the cross-shaped obstacle wall", "Reach tall target on left"],
            "subgoals_l2":  [''                                       , ''                                             ],
            "goal_flow":    "Steer the blue agent through the only gaps in the cross-shaped obstacle wall to the tall target.",
            "goal_flow_l2": ''                                                                                                               ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Purple agent + walkable zone + dense obstacle column"],
            "gimmicks_l2":  [''                                                                           ],
            "subgoals":     ["Separate purple agent from its target zone", "Guide blue agent to square target"],
            "subgoals_l2":  [''                                                                           , ''                                                                  ],
            "goal_flow":    "Block the purple agent with the selector to free the walkable zone, then guide blue agent to the square target.",
            "goal_flow_l2": ''                                                                                                                                                                           ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Background blocks split grid, floor_area + purple agent, two obstacle rows"],
            "gimmicks_l2":  [''                                                                                                          ],
            "subgoals":     ["Navigate through the narrow corridor between obstacle rows", "Place agent on floor area"],
            "subgoals_l2":  [''                                          , ''                                                ],
            "goal_flow":    "Move the selector through the corridor between obstacle rows to reach and capture the purple agent in the floor area.",
            "goal_flow_l2": ''                                                                                                                        ,
            "difficulty":   "hard",
        },
        {
            "gimmicks":     ["Multiple orange+purple agents, walkable zones, horizontal obstacle rows, wide target"],
            "gimmicks_l2":  [''                                                                                                                           ],
            "subgoals":     ["Manage multiple agents simultaneously", "Route each to their target across obstacle rows"],
            "subgoals_l2":  [''                                                , ''                                                                              ],
            "goal_flow":    "Coordinate selector to handle multiple agents across horizontal obstacle rows and navigate them to the wide target.",
            "goal_flow_l2": ''                                                                                                                     ,
            "difficulty":   "very_hard",
        },
        {
            "gimmicks":     ["Multiple agents + complex maze of obstacles, hazards, walkable zones, tight budget"],
            "gimmicks_l2":  [''                                                                                                                  ],
            "subgoals":     ["Separate agents into correct zones", "Navigate past hazards and obstacle maze"],
            "subgoals_l2":  [''                                                      , ''                                                ],
            "goal_flow":    "In 70 steps, coordinate multiple agents through a complex obstacle maze with hazard strips to their respective targets.",
            "goal_flow_l2": ''                                                                                                                                                  ,
            "difficulty":   "very_hard",
        },
    ]

    analysis = _analyses[level_index] if level_index < len(_analyses) else {}

    return {
        "level_index":  level_index + 1,
        "grid_size":    list(grid),
        "step_counter": step_count,
        "sprites":      level_sprites_info,
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
        "game_id":          "wa30",
        "title":            "Warehouse 30",
        "title_l2":         ''                    ,
        "description":      "A warehouse puzzle game where you use a selector cursor to intercept and guide autonomous agents to their target tiles before running out of steps.",
        "description_l2":   ''                                                                                                                                                               ,
        "default_grid_size": [64, 64],
        "num_levels":        len(levels),

        "goal":             "Guide all agents to their corresponding target tiles before the step counter reaches zero.",
        "goal_l2":          ''                                                                                                             ,
        "win_condition":    "_check_all_targets_reached() returns True when all target sprites are inside floor_area zones and none are held by agents.",
        "win_condition_l2": ''                                                                                                                              ,
        "lose_condition":   "StepCounter reaches 0.",
        "lose_condition_l2":''                              ,

        "actions": {
            "ACTION1": {
                "key":             "w / ArrowUp",
                "effect":          "Move selector up by one grid step (4px). If carrying an agent, the agent moves too.",
                "effect_l2":       ''                                                                                                                            ,
                "restrictions":    "Cannot move into collidable sprites or outside grid boundary.",
                "restrictions_l2": ''                                                                                 ,
                "step_cost":       1,
            },
            "ACTION2": {
                "key":             "s / ArrowDown",
                "effect":          "Move selector down by one grid step (4px).",
                "effect_l2":       ''                                                       ,
                "restrictions":    "Cannot move into collidable sprites or outside grid boundary.",
                "restrictions_l2": ''                                                                                 ,
                "step_cost":       1,
            },
            "ACTION3": {
                "key":             "a / ArrowLeft",
                "effect":          "Move selector left by one grid step (4px).",
                "effect_l2":       ''                                                       ,
                "restrictions":    "Cannot move into collidable sprites or outside grid boundary.",
                "restrictions_l2": ''                                                                                 ,
                "step_cost":       1,
            },
            "ACTION4": {
                "key":             "d / ArrowRight",
                "effect":          "Move selector right by one grid step (4px).",
                "effect_l2":       ''                                                       ,
                "restrictions":    "Cannot move into collidable sprites or outside grid boundary.",
                "restrictions_l2": ''                                                                                 ,
                "step_cost":       1,
            },
            "ACTION5": {
                "key":             "space / Enter",
                "effect":          "Grab or release an agent adjacent to the selector in its facing direction. Also removes a purple agent if it is adjacent.",
                "effect_l2":       ''                                                                                                                                                    ,
                "restrictions":    "Only works on agents (blue_agent / purple_agent) adjacent in the facing direction.",
                "restrictions_l2": ''                                                                                           ,
                "step_cost":       1,
            },
        },

        "sprite_roles": {
            "blue_agent":      "Autonomous orange-colored agent. BFS-picks up the nearest target not yet inside a floor_area zone, then carries it onto a floor_area cell. Removable by selector grab of the target it carries.",
            "blue_agent_l2":   ''                                                                                                                                                                                                             ,
            "purple_agent":    "Autonomous purple agent. BFS-picks up the nearest target not yet inside a walkable_zone and carries it onto a walkable_zone cell. Removed when selector ACTION5 is used while adjacent.",
            "purple_agent_l2": ''                                                                                                                                                                                                        ,
            "selector":        "Player-controlled cursor. Moves in 4 directions, grabs/releases the target in its facing direction with ACTION5. Shows current facing via rotation.",
            "selector_l2":     ''                                                                                                                                                        ,
            "target":          "Small collidable 4x4 goal block carried by agents. Dark-gray border + blue interior. Border color indicates status (WHITE=grabbed by selector, GRAY=selector adjacent, BLACK=grabbed by other agent, DARK_GRAY=free).",
            "target_l2":       ''                                                                                                                                                                                                                                                                                                 ,
            "floor_area":      "Blue-bordered floor zone. Blue agents need to place targets here to win; acts as the goal region for blue agents.",
            "floor_area_l2":   ''                                                                                                              ,
            "walkable_zone":   "Light-gray walkable floor zone. Purple agents drop targets here (and the zone is counted for purple-agent BFS success).",
            "walkable_zone_l2":''                                                                                                              ,
            "hazard":          "Trap tile: agents cannot pass through it, and stepping on it blocks target placement in its cell.",
            "hazard_l2":       ''                                                                                                           ,
            "obstacle_block":  "Collidable 4×4 solid black block. Blocks all movement.",
            "obstacle_block_l2":''                                                                   ,
        },

        "mechanics": {
            "agent_bfs":          "Each step, blue agents BFS toward the nearest unheld target not yet inside floor_area; purple agents BFS toward unheld targets not yet inside walkable_zone. When carrying, they BFS toward the relevant zone.",
            "agent_bfs_l2":       ''                                                                                                                                                                                                                                                           ,
            "selector_grab":      "ACTION5 checks the cell in the selector's facing direction (rotation 0=up, 90=right, 180=down, 270=left). If a target is there, the selector grabs it; a purple_agent there is removed instead.",
            "selector_grab_l2":   ''                                                                                                                                                                                                   ,
            "target_colors":      "_update_target_colors() runs each step to recolor target borders: WHITE=grabbed by selector, GRAY=selector adjacent, BLACK=grabbed by other agent, DARK_GRAY=free. Purple agents are repainted YELLOW when selector faces them, otherwise PURPLE.",
            "target_colors_l2":   ''                                                                                                                                                                                                                                                                                                                          ,
            "win_check":          "After each action, _check_all_targets_reached() checks if all target sprites are inside a floor_area cell and unheld → level cleared.",
            "win_check_l2":       ''                                                                                                                                              ,
            "step_budget":        "Each of the 5 actions costs 1 step. When StepCounter reaches 0, the game ends in defeat.",
            "step_budget_l2":     ''                                                                                                        ,
        },
    },


    "levels": [
        _build_level_knowledge(i, lv)
        for i, lv in enumerate(levels)
    ],
}




class StepCounterDisplay(RenderableUserDisplay):
    """






    """

    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.current_steps = max_steps

    def set_value(self, value: int) -> None:
        """"""
        self.current_steps = max(0, min(value, self.max_steps))

    def tick(self) -> bool:
        """"""
        if self.current_steps > 0:
            self.current_steps -= 1
        return self.current_steps > 0

    def reset(self) -> None:
        """"""
        self.current_steps = self.max_steps

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """"""
        if self.max_steps == 0:
            return frame
        ratio = self.current_steps / self.max_steps
        filled = round(64 * ratio)
        for col in range(64):
            if col < filled:
                frame[63, col] = PINK
            else:
                frame[63, col] = DARK_GRAY
        return frame




class Wa30Camera(Camera):
    """



    """

    def _raw_render(self, sprites: list[Sprite]) -> np.ndarray:
        """"""
        output = np.full((self._height, self._width), self._background, dtype=np.int8)
        for sprite in sprites:
            sprite_pixels = sprite.pixels
            sprite_height, sprite_width = sprite_pixels.shape
            rel_x = sprite._x - self._x
            rel_y = sprite._y - self._y
            dest_x_start = max(0, rel_x)
            dest_x_end   = min(self._width,  rel_x + sprite_width)
            dest_y_start = max(0, rel_y)
            dest_y_end   = min(self._height, rel_y + sprite_height)
            if dest_x_end <= dest_x_start or dest_y_end <= dest_y_start:
                continue
            sprite_x_start = max(0, -rel_x)
            sprite_x_end   = sprite_width  - max(0, rel_x + sprite_width  - self._width)
            sprite_y_start = max(0, -rel_y)
            sprite_y_end   = sprite_height - max(0, rel_y + sprite_height - self._height)
            sprite_region  = sprite_pixels[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]
            visible_mask   = sprite_region >= 0
            output[dest_y_start:dest_y_end, dest_x_start:dest_x_end][visible_mask] = sprite_region[visible_mask]
        return output




def _manhattan_distance(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> int:
    """"""
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def _is_adjacent(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    """"""
    return _manhattan_distance(pos_a, pos_b) == GRID_STEP


def _selector_faces(selector: Sprite, target: Sprite) -> bool:
    """


    """
    if selector.rotation == 0:
        return selector.x == target.x and selector.y - GRID_STEP == target.y
    elif selector.rotation == 180:
        return selector.x == target.x and selector.y + GRID_STEP == target.y
    elif selector.rotation == 90:
        return selector.x + GRID_STEP == target.x and selector.y == target.y
    else:
        return selector.x - GRID_STEP == target.x and selector.y == target.y


def _paint_target_border(target: Sprite, color: int) -> None:
    """"""
    target.pixels[:, 0]  = color
    target.pixels[:, 3]  = color
    target.pixels[0, :]  = color
    target.pixels[3, :]  = color


def _rotation_from_delta(dx: int, dy: int) -> int:
    """



    """
    if dy < 0:
        return 0
    elif dx > 0:
        return 90
    elif dy > 0:
        return 180
    return 270


def _point_in_sprite(point: tuple[int, int], sprite: Sprite) -> bool:
    """"""
    return (sprite.x <= point[0] < sprite.x + sprite.width and
            sprite.y <= point[1] < sprite.y + sprite.height)




class Wa30(ARCBaseGame):
    """












    """

    step_counter: StepCounterDisplay
    grabbed_map:     dict[Sprite, Sprite]
    target_to_agent: dict[Sprite, Sprite]
    obstacle_cells:  set[tuple[int, int]]
    floor_cells:     set[tuple[int, int]]
    walkable_cells:  set[tuple[int, int]]
    hazard_cells:    set[tuple[int, int]]
    blue_targets:    set[tuple[int, int]]
    purple_targets:  set[tuple[int, int]]

    def __init__(self) -> None:
        """"""
        self.step_counter = StepCounterDisplay(0)
        camera = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.step_counter],
        )




        self._log_state = ActionLogState(aggregator=default_aggregator, post_process=compose_default_summary)
        self._frame_sprite_log:      list[list[dict]] = []
        self._frame_nonsprite_log:   list[list[dict]] = []

        super().__init__(
            game_id="wa30",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5],
        )

    def _get_hidden_state(self) -> np.ndarray:
        """"""
        state = np.zeros((1, 1), dtype=np.int16)
        state[0] = self.step_counter.current_steps
        return state



    def on_set_level(self, level: Level) -> None:
        """"""
        self._reset_step_counter()
        self.grabbed_map     = dict()
        self.target_to_agent = dict()
        self.obstacle_cells  = set()
        self.floor_cells     = set()
        self.walkable_cells  = set()
        self.hazard_cells    = set()
        self.blue_targets    = set()
        self.purple_targets  = set()


        for spr in self.current_level.get_sprites():
            if spr.is_collidable:
                self.obstacle_cells.add((spr.x, spr.y))


        for i in range(0, 64, GRID_STEP):
            self.obstacle_cells.add((-GRID_STEP, i))
            self.obstacle_cells.add((64, i))
            self.obstacle_cells.add((i, -GRID_STEP))
            self.obstacle_cells.add((i, 64))


        for spr in self.current_level.get_sprites_by_tag("floor_area"):
            for row in range(spr.height):
                for col in range(spr.width):
                    self.floor_cells.add((spr.x + col, spr.y + row))


        for spr in self.current_level.get_sprites_by_tag("walkable_zone"):
            for row in range(spr.height):
                for col in range(spr.width):
                    self.walkable_cells.add((spr.x + col, spr.y + row))


        for spr in self.current_level.get_sprites_by_tag("hazard"):
            self.hazard_cells.add((spr.x, spr.y))


        self._update_blue_targets()
        self._update_purple_targets()

        self._emit_initial_layout()


    def _emit_initial_layout(self) -> None:
        """















        """
        _layout_rows: list[dict] = []

        for _tag, _role, _role_l2 in (
            ("selector",      "player",       ""),
            ("target",        "box",          ""),
            ("floor_area",    "goal_blue",    ""),
            ("walkable_zone", "goal_purple",  ""),
            ("blue_agent",    "agent_blue",   ""),
            ("purple_agent",  "agent_purple", ""),
            ("hazard",        "hazard",       ""),
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

    def _reset_step_counter(self) -> None:
        """"""
        count = self.current_level.get_data("StepCounter")
        self.step_counter.max_steps = count
        self.step_counter.reset()



    def _update_blue_targets(self) -> None:
        """



        """
        self.blue_targets = set()
        for tgt in self.current_level.get_sprites_by_tag("target"):
            if tgt not in self.target_to_agent and not self._is_in_floor_area((tgt.x, tgt.y)):
                for neighbor in self._neighbors(tgt.x, tgt.y):
                    self.blue_targets.add(neighbor)

    def _update_purple_targets(self) -> None:
        """



        """
        self.purple_targets = set()
        for tgt in self.current_level.get_sprites_by_tag("target"):
            if not self._is_target_held_by_purple(tgt) and not self._is_in_walkable_zone((tgt.x, tgt.y)):
                for neighbor in self._neighbors(tgt.x, tgt.y):
                    self.purple_targets.add(neighbor)

    def _neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """"""
        return [
            (x - GRID_STEP, y),
            (x + GRID_STEP, y),
            (x, y - GRID_STEP),
            (x, y + GRID_STEP),
        ]



    def _try_move_sprite(self, sprite: Sprite, x: int, y: int) -> None:
        """



        """
        if sprite in self.grabbed_map:

            tgt = self.grabbed_map[sprite]
            if self._can_move_pair(sprite, tgt, (x, y)):
                dx = tgt.x - sprite.x
                dy = tgt.y - sprite.y

                self.obstacle_cells.discard((sprite.x, sprite.y))
                self.obstacle_cells.discard((tgt.x, tgt.y))
                sprite.set_position(x, y)
                tgt.set_position(x + dx, y + dy)
                self.obstacle_cells.add((sprite.x, sprite.y))
                self.obstacle_cells.add((tgt.x, tgt.y))
                self._update_purple_targets()
        elif self._is_free_cell((x, y)):

            self.obstacle_cells.discard((sprite.x, sprite.y))
            sprite.set_position(x, y)
            self.obstacle_cells.add((sprite.x, sprite.y))

    def _move_sprite_by(self, sprite: Sprite, dx: int, dy: int) -> None:
        """


        """
        if sprite not in self.grabbed_map:
            sprite.set_rotation(_rotation_from_delta(dx, dy))
        self._try_move_sprite(sprite, sprite.x + dx, sprite.y + dy)

    def _is_free_cell(self, pos: tuple[int, int]) -> bool:
        """"""
        return pos not in self.obstacle_cells and pos not in self.hazard_cells

    def _can_move_pair(self, agent: Sprite, tgt: Sprite, new_pos: tuple[int, int]) -> bool:
        """





        """
        dx = tgt.x - agent.x
        dy = tgt.y - agent.y
        new_tgt_pos = (new_pos[0] + dx, new_pos[1] + dy)

        agent_ok = (new_pos not in self.obstacle_cells or new_pos == (tgt.x, tgt.y)) \
                   and new_pos not in self.hazard_cells

        tgt_ok   = new_tgt_pos not in self.obstacle_cells or new_tgt_pos == (agent.x, agent.y)
        return agent_ok and tgt_ok



    def _is_in_floor_area(self, pos: tuple[int, int]) -> bool:
        """"""
        return pos in self.floor_cells

    def _is_in_walkable_zone(self, pos: tuple[int, int]) -> bool:
        """"""
        return pos in self.walkable_cells

    def _is_target_held_by_purple(self, tgt: Sprite) -> bool:
        """"""
        if tgt in self.target_to_agent:
            agent = self.target_to_agent[tgt]
            return "purple_agent" in agent.tags
        return False



    def _bfs_to_blue_targets(self, agent: Sprite) -> list[tuple[int, int]] | None:
        """



        """
        start   = (agent.x, agent.y)
        visited = {start}
        queue   = [[start]]
        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if curr in self.blue_targets:
                return path
            for nb in self._neighbors(*curr):
                if nb not in visited and self._is_free_cell(nb):
                    visited.add(nb)
                    queue.append(path + [nb])
        return None

    def _bfs_to_blue_targets_with_pair(self, agent: Sprite) -> list[tuple[int, int]] | None:
        """



        """
        start   = (agent.x, agent.y)
        tgt     = self.grabbed_map[agent]
        visited = {start}
        queue   = [[start]]
        dx = tgt.x - agent.x
        dy = tgt.y - agent.y
        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if self._is_in_floor_area((curr[0] + dx, curr[1] + dy)):
                return path
            for nb in self._neighbors(*curr):
                if nb not in visited and self._can_move_pair(agent, tgt, nb):
                    visited.add(nb)
                    queue.append(path + [nb])
        return None

    def _bfs_to_purple_targets(self, agent: Sprite) -> list[tuple[int, int]] | None:
        """"""
        start   = (agent.x, agent.y)
        visited = {start}
        queue   = [[start]]
        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if curr in self.purple_targets:
                return path
            for nb in self._neighbors(*curr):
                if nb not in visited and self._is_free_cell(nb):
                    visited.add(nb)
                    queue.append(path + [nb])
        return None

    def _bfs_to_purple_targets_with_pair(self, agent: Sprite) -> list[tuple[int, int]] | None:
        """"""
        start   = (agent.x, agent.y)
        tgt     = self.grabbed_map[agent]
        visited = {start}
        queue   = [[start]]
        dx = tgt.x - agent.x
        dy = tgt.y - agent.y
        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if self._is_in_walkable_zone((curr[0] + dx, curr[1] + dy)):
                return path
            for nb in self._neighbors(*curr):
                if nb not in visited and self._can_move_pair(agent, tgt, nb):
                    visited.add(nb)
                    queue.append(path + [nb])
        return None



    def _attach_agent_to_target(self, agent: Sprite, tgt: Sprite) -> None:
        """"""
        if tgt in self.target_to_agent:

            self._detach_agent(self.target_to_agent[tgt])
        self.grabbed_map[agent]     = tgt
        self.target_to_agent[tgt]   = agent
        self._update_blue_targets()
        self._update_purple_targets()

    def _detach_agent(self, agent: Sprite) -> None:
        """"""
        if agent in self.grabbed_map:
            tgt = self.grabbed_map[agent]
            del self.target_to_agent[tgt]
            del self.grabbed_map[agent]
            self._update_blue_targets()
            self._update_purple_targets()



    def _update_target_colors(self) -> None:
        """










        """
        selectors = self.current_level.get_sprites_by_tag("selector")
        targets   = self.current_level.get_sprites_by_tag("target")

        for tgt in targets:
            if tgt in self.target_to_agent:
                agent = self.target_to_agent[tgt]
                if any(s for s in selectors if agent is s):

                    _paint_target_border(tgt, WHITE)
                elif any(_selector_faces(s, tgt) for s in selectors):
                    _paint_target_border(tgt, GRAY)
                else:
                    _paint_target_border(tgt, BLACK)
            elif any(_selector_faces(s, tgt) for s in selectors):
                _paint_target_border(tgt, GRAY)
            else:
                _paint_target_border(tgt, DARK_GRAY)

        for purple in self.current_level.get_sprites_by_tag("purple_agent"):
            if any(_selector_faces(s, purple) for s in selectors):
                _paint_target_border(purple, YELLOW)
            else:
                _paint_target_border(purple, PURPLE)



    def _update_blue_agents(self) -> None:
        """



        """
        for agent in self.current_level.get_sprites_by_tag("blue_agent"):
            if agent in self.grabbed_map:

                tgt  = self.grabbed_map[agent]
                if self._is_in_floor_area((tgt.x, tgt.y)):

                    self._detach_agent(agent)
                else:
                    path = self._bfs_to_blue_targets_with_pair(agent)
                    if path and len(path) > 1:
                        next_cell = path[1]
                        self._try_move_sprite(agent, next_cell[0], next_cell[1])
            else:

                targets = self.current_level.get_sprites_by_tag("target")
                for tgt in targets:
                    if _is_adjacent((agent.x, agent.y), (tgt.x, tgt.y)) \
                            and tgt not in self.target_to_agent \
                            and not self._is_in_floor_area((tgt.x, tgt.y)):
                        self._attach_agent_to_target(agent, tgt)
                        return
                path = self._bfs_to_blue_targets(agent)
                if path and len(path) > 1:
                    next_cell = path[1]
                    self._try_move_sprite(agent, next_cell[0], next_cell[1])



    def _update_purple_agents(self) -> None:
        """"""
        for agent in self.current_level.get_sprites_by_tag("purple_agent"):
            if agent in self.grabbed_map:
                tgt = self.grabbed_map[agent]
                if self._is_in_walkable_zone((tgt.x, tgt.y)):
                    self._detach_agent(agent)
                else:
                    path = self._bfs_to_purple_targets_with_pair(agent)
                    if path and len(path) > 1:
                        next_cell = path[1]
                        self._try_move_sprite(agent, next_cell[0], next_cell[1])
            else:
                targets = self.current_level.get_sprites_by_tag("target")
                for tgt in targets:
                    if _is_adjacent((agent.x, agent.y), (tgt.x, tgt.y)) \
                            and not self._is_target_held_by_purple(tgt) \
                            and not self._is_in_walkable_zone((tgt.x, tgt.y)):
                        self._attach_agent_to_target(agent, tgt)
                        return
                path = self._bfs_to_purple_targets(agent)
                if path and len(path) > 1:
                    next_cell = path[1]
                    self._try_move_sprite(agent, next_cell[0], next_cell[1])



    def _check_all_targets_reached(self) -> bool:
        """"""
        targets = self.current_level.get_sprites_by_tag("target")
        return all(
            self._is_in_floor_area((tgt.x, tgt.y)) and tgt not in self.target_to_agent
            for tgt in targets
        )



    def _handle_action(self, action: ActionInput) -> None:
        """



        """
        selector = self.current_level.get_sprites_by_tag("selector")[0]

        if action.id == GameAction.ACTION1:
            self.step_counter.tick()
            self._move_sprite_by(selector, 0, -GRID_STEP)
            self._run_agents()
        elif action.id == GameAction.ACTION2:
            self.step_counter.tick()
            self._move_sprite_by(selector, 0, GRID_STEP)
            self._run_agents()
        elif action.id == GameAction.ACTION3:
            self.step_counter.tick()
            self._move_sprite_by(selector, -GRID_STEP, 0)
            self._run_agents()
        elif action.id == GameAction.ACTION4:
            self.step_counter.tick()
            self._move_sprite_by(selector, GRID_STEP, 0)
            self._run_agents()
        elif action.id == GameAction.ACTION5:
            self.step_counter.tick()
            if selector in self.grabbed_map:

                self._detach_agent(selector)
            else:

                targets = self.current_level.get_sprites_by_tag("target")
                for tgt in targets:
                    if _selector_faces(selector, tgt):
                        self._attach_agent_to_target(selector, tgt)
                        break

                for purple in self.current_level.get_sprites_by_tag("purple_agent"):
                    if _selector_faces(selector, purple):
                        self._detach_agent(purple)
                        self.obstacle_cells.discard((purple.x, purple.y))
                        self.current_level.remove_sprite(purple)
                        break
            self._run_agents()

    def _run_agents(self) -> None:
        """"""
        self._update_blue_agents()
        self._update_purple_agents()
        self._update_target_colors()



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
        max_s = self.step_counter.max_steps
        if max_s <= 0:
            return 0
        return round(64 * steps / max_s)

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




                "border_color": (
                    int(spr.pixels[0, 0])
                    if (spr._tags and ("target" in spr._tags or "purple_agent" in spr._tags))
                    else None
                ),
            }
            for spr in self.current_level.get_sprites()
        }
        _before_steps = self.step_counter.current_steps




        self._handle_action(self.action)




        _won = False
        if self._check_all_targets_reached():
            self.next_level()
            _won = True
        elif not self.step_counter.current_steps:
            self.lose()





        after_sprites = {id(spr): spr for spr in self.current_level.get_sprites()}







        _MULTI_INST_NAMES = {"wall_block", "obstacle_block", "hazard",
                             "orange_agent", "purple_agent", "selector"}
        def _tgt(name: str, dx: int, dy: int) -> str:
            if name in _MULTI_INST_NAMES:
                return f"{name}_at_{dx}_{dy}"
            return name


        for spr_id, spr in after_sprites.items():
            b = _before_sprites.get(spr_id)
            if b is None or (not b["visible"] and spr.is_visible):
                dx, dy, dw, dh = self._to_display_bbox(spr)
                tgt = _tgt(spr.name, dx, dy)
                self.action_log.append(f"[appear]:[{tgt}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name} appears at ({dx},{dy}).|Object appears at ({dx},{dy}).")
                self.action_log_l2.append(f"[appear]:[{tgt}]:[sname({spr.name});position({dx}_{dy});bbox({dw}_{dh})]:{spr.name}({dx},{dy})|Object({dx},{dy})")



        _arc_scale_d = min(64 // self.camera.width, 64 // self.camera.height)
        for spr_id, b in _before_sprites.items():
            spr = after_sprites.get(spr_id)
            gone = spr is None and b["visible"]
            hid  = spr is not None and b["visible"] and not spr.is_visible
            if gone or hid:
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                bdw = int(b.get("width", 0))  * _arc_scale_d
                bdh = int(b.get("height", 0)) * _arc_scale_d
                tgt = _tgt(b['name'], bdx, bdy)
                self.action_log.append(f"[disappear]:[{tgt}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']} disappears from ({bdx},{bdy}).|Object disappears from ({bdx},{bdy}).")
                self.action_log_l2.append(f"[disappear]:[{tgt}]:[sname({b['name']});position({bdx}_{bdy});bbox({bdw}_{bdh})]:{b['name']}({bdx},{bdy})|Object({bdx},{bdy})")


        for spr_id, spr in after_sprites.items():
            if not spr.is_visible:
                continue
            b = _before_sprites.get(spr_id)
            if b is None or not b["visible"]:
                continue
            if (spr.x, spr.y) != (b["x"], b["y"]):
                bdx, bdy = self._to_display_xy(b["x"], b["y"])
                adx, ady, adw, adh = self._to_display_bbox(spr)


                tgt = _tgt(spr.name, bdx, bdy)
                self.action_log.append(f"[move]:[{tgt}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name} moves from ({bdx},{bdy}) to ({adx},{ady}).|Object moves from ({bdx},{bdy}) to ({adx},{ady}).")
                self.action_log_l2.append(f"[move]:[{tgt}]:[sname({spr.name});position({bdx}_{bdy})_({adx}_{ady});bbox({adw}_{adh})]:{spr.name}({bdx},{bdy})({adx},{ady})|Object({bdx},{bdy})({adx},{ady})")
            if spr.rotation != b["rotation"]:
                _rdx, _rdy, _rdw, _rdh = self._to_display_bbox(spr)
                tgt = _tgt(spr.name, _rdx, _rdy)
                self.action_log.append(f"[rotate]:[{tgt}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:{spr.name} at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.|Object at ({_rdx},{_rdy}) rotates from {b['rotation']}° to {spr.rotation}°.")
                self.action_log_l2.append(f"[rotate]:[{tgt}]:[sname({spr.name});position({_rdx}_{_rdy});bbox({_rdw}_{_rdh});angle({b['rotation']})_({spr.rotation})]:({_rdx},{_rdy}){spr.name}{b['rotation']}°{spr.rotation}°|({_rdx},{_rdy})Object{b['rotation']}°{spr.rotation}°")


            if b["border_color"] is not None:
                after_border = int(spr.pixels[0, 0])
                if after_border != b["border_color"]:
                    before_name    = COLOR_PALETTE.get(b["border_color"], {}).get("name", str(b["border_color"]))
                    after_name     = COLOR_PALETTE.get(after_border,     {}).get("name", str(after_border))
                    before_name_l2 = COLOR_PALETTE.get(b["border_color"], {}).get("label_l2", str(b["border_color"]))
                    after_name_l2  = COLOR_PALETTE.get(after_border,     {}).get("label_l2", str(after_border))
                    adx, ady, adw, adh = self._to_display_bbox(spr)
                    tgt = _tgt(spr.name, adx, ady)
                    self.action_log.append(f"[change]:[{tgt}]:[sname({spr.name});position({adx}_{ady});bbox({adw}_{adh});color({before_name})_({after_name})]:{spr.name} at ({adx},{ady}) changes border color from {before_name} to {after_name}.|Object at ({adx},{ady}) changes border color from {before_name} to {after_name}.")
                    self.action_log_l2.append(f"[change]:[{tgt}]:[sname({spr.name});position({adx}_{ady});bbox({adw}_{adh});color({before_name})_({after_name})]:{spr.name}({adx},{ady}){before_name_l2}{after_name_l2}|Object({adx},{ady}){before_name_l2}{after_name_l2}")


        after_steps = self.step_counter.current_steps
        if after_steps != _before_steps and self._step_bar_pixel_len(_before_steps) != self._step_bar_pixel_len(after_steps):
            if after_steps < _before_steps:
                self.action_log.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:The pink step bar at the bottom shrinks from the right ({_before_steps}→{after_steps} steps remain).|The pink Object at the bottom shrinks from the right ({_before_steps}→{after_steps}).")
                self.action_log_l2.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:{_before_steps}→{after_steps}|Object{_before_steps}→{after_steps}")
            else:
                self.action_log.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:The pink step bar at the bottom extends to the right ({_before_steps}→{after_steps} steps).|The pink Object at the bottom extends to the right ({_before_steps}→{after_steps}).")
                self.action_log_l2.append(f"[change]:[steps]:[sname(step_counter);count({_before_steps})_({after_steps})]:{_before_steps}→{after_steps}|Object{_before_steps}→{after_steps}")


        if _won:
            self._set_event("win", {
                "en": "[win]:[game]:[]:The level is cleared — all targets reached.|The level is cleared.",
                "l2": "[win]:[game]:[]:|",
            })
        elif self.step_counter.current_steps == 0:
            self._set_event("gameover", {
                "en": "[gameover]:[game]:[]:Step counter reached zero, game over.|The Object reached zero, game over.",
                "l2": "[gameover]:[game]:[]:0|Object0",
            })




        if hasattr(self, "_frame_sprite_log"):
            self._frame_sprite_log.append(self.get_sprite_status())
        if hasattr(self, "_frame_nonsprite_log"):
            self._frame_nonsprite_log.append(self.get_nonsprite_status())
        self._log_state.snapshot_step()

        self.complete_action()



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
        return [
            {
                "name":     "step_counter",
                "label":    "Steps Remaining",
                "label_l2": ''                     ,
                "value":    self.step_counter.current_steps,
                "max":      self.step_counter.max_steps,
                "x":        0, "y":      63,
                "width":    64, "height": 1,
            }
        ]

    def get_game_knowledge(self) -> dict:
        """

        """
        return GAME_KNOWLEDGE
