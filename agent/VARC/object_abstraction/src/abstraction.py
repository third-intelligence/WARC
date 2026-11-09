from __future__ import annotations
import sys
from pathlib import Path
from typing import List
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.novelty import Memory
from cell_effects.src.effects import EffectTracker
MIN_MASK_PIXELS = 64

class MaskedMemory(Memory):

    def __init__(self, mask: np.ndarray, block: int=4, minimum: int=5, window: int=20) -> None:
        super().__init__(mask, block, minimum, window)
        self.frames: List[np.ndarray] = []
        self.remasks = 0

    def add(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, np.uint8)
        f = self.encode(frame)
        if not any((np.array_equal(f, old) for old in self.observed)):
            self.observed.append(f)
            self.frames.append(frame.copy())

    def set_mask(self, mask: np.ndarray) -> bool:
        if int(mask.sum()) < MIN_MASK_PIXELS or np.array_equal(mask, self.mask):
            return False
        self.mask = mask.copy()
        frames, self.frames, self.observed = (self.frames, [], [])
        for fr in frames:
            self.add(fr)
        self.remasks += 1
        return True
