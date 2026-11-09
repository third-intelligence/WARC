from __future__ import annotations
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Hashable, Optional, Sequence, Tuple
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.perception import CLICK_ACTION_ID, components, digest
ActionTriple = Tuple[int, int, int]
Descriptor = Tuple
EFFECTS = ('noop', 'local', 'other', 'global', 'level_up', 'game_over')
_K = len(EFFECTS)
_IDX = {e: i for i, e in enumerate(EFFECTS)}

class FrameObjects:

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = np.asarray(frame, np.uint8)
        self.labels = np.full(self.frame.shape, -1, np.int32)
        self.descs = []
        self.pixels = []
        for i, (color, px) in enumerate(components(self.frame)):
            self.labels[px[:, 0], px[:, 1]] = i
            y0, x0 = px.min(0)
            y1, x1 = px.max(0)
            bitmask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
            bitmask[px[:, 0] - y0, px[:, 1] - x0] = 1
            shape = hashlib.sha1(repr(bitmask.shape).encode() + bitmask.tobytes()).hexdigest()[:12]
            self.descs.append(('c', int(color), shape))
            self.pixels.append(px)

    def descriptor(self, action: ActionTriple) -> Descriptor:
        aid, x, y = (int(v) for v in action)
        if aid != CLICK_ACTION_ID:
            return ('k', aid)
        return self.descs[int(self.labels[y, x])]

    def target_mask(self, action: ActionTriple, dilate: int=1) -> Optional[np.ndarray]:
        aid, x, y = (int(v) for v in action)
        if aid != CLICK_ACTION_ID:
            return None
        px = self.pixels[int(self.labels[y, x])]
        m = np.zeros(self.frame.shape, bool)
        m[px[:, 0], px[:, 1]] = True
        for _ in range(dilate):
            g = m.copy()
            g[1:] |= m[:-1]
            g[:-1] |= m[1:]
            g[:, 1:] |= m[:, :-1]
            g[:, :-1] |= m[:, 1:]
            m = g
        return m

class ObjectCache:

    def __init__(self, size: int=64) -> None:
        self.size = size
        self.items: Dict[str, FrameObjects] = {}

    def get(self, frame: np.ndarray) -> FrameObjects:
        key = digest(frame)
        obj = self.items.get(key)
        if obj is None:
            if len(self.items) >= self.size:
                self.items.pop(next(iter(self.items)))
            obj = self.items[key] = FrameObjects(frame)
        return obj

def classify_effect(prev: np.ndarray, nxt: np.ndarray, target: Optional[np.ndarray], mask: np.ndarray, *, level_up: bool=False, game_over: bool=False) -> str:
    if game_over:
        return 'game_over'
    if level_up:
        return 'level_up'
    changed = (np.asarray(prev) != np.asarray(nxt)) & mask
    n = int(changed.sum())
    if n == 0:
        return 'noop'
    if n * 2 >= int(mask.sum()):
        return 'global'
    if target is not None and (not (changed & ~target).any()):
        return 'local'
    return 'other'

class AffordanceMemory:

    def __init__(self, alpha: float=0.1, lambda_h: float=1.0) -> None:
        self.alpha = float(alpha)
        self.lambda_h = float(lambda_h)
        self.counts: Dict[Hashable, np.ndarray] = defaultdict(lambda: np.zeros(_K))
        self.clear_bonus: Dict[Hashable, float] = defaultdict(float)
        self.updates = 0

    def add(self, desc: Descriptor, effect: str, weight: float=1.0) -> None:
        self.counts[desc][_IDX[effect]] += weight
        self.updates += 1

    def n(self, desc: Descriptor) -> float:
        c = self.counts.get(desc)
        return 0.0 if c is None else float(c.sum())

    def interest(self, desc: Descriptor) -> float:
        c = self.counts.get(desc)
        if c is None:
            return 1.0 + self.lambda_h
        p = (c + self.alpha) / (c.sum() + self.alpha * _K)
        h = float(-(p * np.log(p)).sum() / np.log(_K))
        return 1.0 / np.sqrt(1.0 + c.sum()) + self.lambda_h * h

    def interest_max(self) -> float:
        return 1.0 + self.lambda_h

    def p_game_over(self, desc: Descriptor) -> float:
        c = self.counts.get(desc)
        if c is None:
            return 0.0
        return float((c[_IDX['game_over']] + 0.1) / (c.sum() + 1.0)) if c[_IDX['game_over']] else 0.0

    def decay(self, factor: float) -> None:
        for d in list(self.counts):
            self.counts[d] = self.counts[d] * factor
        for d in list(self.clear_bonus):
            self.clear_bonus[d] *= factor

    def mark_clear(self, descs: Sequence[Descriptor]) -> None:
        n = len(descs)
        for i, d in enumerate(descs):
            self.clear_bonus[d] = min(1.0, self.clear_bonus[d] + (i + 1) / n)

    def summary(self) -> Dict:
        tot = np.zeros(_K)
        for c in self.counts.values():
            tot += c
        return dict(aff_descriptors=len(self.counts), aff_updates=self.updates, aff_effects={e: float(tot[i]) for i, e in enumerate(EFFECTS)}, aff_clear_descriptors=sum((1 for v in self.clear_bonus.values() if v > 0)))
