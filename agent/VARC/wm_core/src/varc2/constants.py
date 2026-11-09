from __future__ import annotations
import numpy as np
FRAME_SIZE = 64
NUM_COLORS = 16
IGNORE_INDEX = 16
NUM_ACTIONS = 8
CLICK_ACTION_ID = 6
ACTION_STR_TO_INT = {'RESET': 0, 'ACTION1': 1, 'ACTION2': 2, 'ACTION3': 3, 'ACTION4': 4, 'ACTION5': 5, 'ACTION6': 6, 'ACTION7': 7}
ACTION_NAMES = ['RESET'] + [f'ACTION{i}' for i in range(1, NUM_ACTIONS)]
ACTION_REMAP = {'rotate90_cw': {1: 4, 4: 2, 2: 3, 3: 1}, 'rotate90_ccw': {1: 3, 3: 2, 2: 4, 4: 1}, 'rotate180': {1: 2, 2: 1, 3: 4, 4: 3}, 'flip_vertical': {1: 2, 2: 1}, 'flip_horizontal': {3: 4, 4: 3}}
GEOMETRIC_TRANSFORMS = ['rotate90_cw', 'rotate90_ccw', 'rotate180', 'flip_vertical', 'flip_horizontal']
FOLDS = {1: ['ls20', 'lp85', 'bp35', 'su15', 'tr87'], 2: ['cn04', 'ft09', 'ka59', 'tn36', 'tu93'], 3: ['cd82', 'dc22', 'lf52', 's5i5', 'wa30'], 4: ['m0r0', 're86', 'sb26', 'sc25', 'vc33'], 5: ['ar25', 'g50t', 'r11l', 'sk48', 'sp80']}
COLORS_UINT8 = np.array([[255, 255, 255], [204, 204, 204], [153, 153, 153], [102, 102, 102], [51, 51, 51], [0, 0, 0], [229, 58, 163], [255, 123, 204], [249, 60, 49], [30, 147, 255], [136, 216, 241], [255, 220, 0], [255, 133, 27], [146, 18, 49], [79, 204, 48], [163, 86, 214]], dtype=np.uint8)

def normalize_action_id(raw_id) -> int:
    if isinstance(raw_id, int):
        return raw_id
    return ACTION_STR_TO_INT.get(str(raw_id), 0)
