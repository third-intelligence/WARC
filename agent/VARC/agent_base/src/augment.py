from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import random
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from src.ARC_loader_AA3 import ACTION_REMAP, GEOMETRIC_TRANSFORMS, NUM_COLORS_AA3, apply_geometric_transform, transform_action6_coord
Item = Tuple[np.ndarray, int, int, int, np.ndarray]

def _apply_color_perm(grid: np.ndarray, color_map: Optional[np.ndarray]) -> np.ndarray:
    if color_map is None:
        return grid
    return color_map[grid]

def augment_one(item: Item, geo: str, color_map: Optional[np.ndarray]) -> Item:
    obs_t, action_id, x, y, obs_t1 = item
    if geo != 'identity':
        obs_t = apply_geometric_transform(obs_t, geo)
        obs_t1 = apply_geometric_transform(obs_t1, geo)
        action_id = ACTION_REMAP.get(geo, {}).get(action_id, action_id)
        if action_id == 6 and x >= 0:
            x, y = transform_action6_coord(x, y, geo)
    obs_t = _apply_color_perm(obs_t, color_map)
    obs_t1 = _apply_color_perm(obs_t1, color_map)
    return (np.ascontiguousarray(obs_t), int(action_id), int(x), int(y), np.ascontiguousarray(obs_t1))
GEOMETRIC_TRANSFORMS_NO_FLIP: List[str] = [t for t in GEOMETRIC_TRANSFORMS if 'flip' not in t]

def expand_items(items: List[Item], *, num_color_permute: int=9, rng: Optional[random.Random]=None, seed_salt: int=0, geometric_transforms: Optional[Sequence[str]]=None) -> List[Item]:
    rng = rng or random.Random()
    geos = list(GEOMETRIC_TRANSFORMS if geometric_transforms is None else geometric_transforms)
    out: List[Item] = []
    for idx, item in enumerate(items):
        out.append(augment_one(item, 'identity', None))
        for geo in geos:
            out.append(augment_one(item, geo, None))
            for _ in range(num_color_permute):
                perm = np.random.RandomState(rng.randrange(2 ** 31)).permutation(NUM_COLORS_AA3).astype(np.uint8)
                out.append(augment_one(item, geo, perm))
    return out

def num_views(num_color_permute: int, n_geo: int=None) -> int:
    if n_geo is None:
        n_geo = len(GEOMETRIC_TRANSFORMS)
    return 1 + n_geo * (1 + num_color_permute)
