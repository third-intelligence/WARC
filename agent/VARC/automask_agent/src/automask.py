from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.perception import LARGE_COMPONENT, components
from graph_agent.src.novelty import Memory
from cell_effects.src.effects import EffectTracker
from object_abstraction.src.abstraction import MIN_MASK_PIXELS

class MajorityEffectTracker(EffectTracker):

    def __init__(self, quorum: float=0.9, expand: str='component') -> None:
        if not 0.5 < quorum <= 1.0:
            raise ValueError(f'quorum must be in (0.5, 1]: {quorum}')
        super().__init__(expand)
        self.quorum = float(quorum)

    def update(self, frame: np.ndarray, outcomes: Sequence[np.ndarray]) -> Dict:
        if len(outcomes) < 2:
            return dict(used=False, new_autonomous=0)
        frame = np.asarray(frame, np.uint8)
        outs = np.stack([np.asarray(o, np.uint8) for o in outcomes])
        n = len(outs)
        disagree = ~(outs == outs[0]).all(0)
        if not disagree.any():
            return dict(used=False, new_autonomous=0)
        counts = np.stack([(outs == c).sum(0) for c in range(16)])
        mode = counts.argmax(0)
        need = math.ceil(self.quorum * n - 1e-09)
        autonomous = (counts.max(0) >= need) & (mode != frame)
        controllable = disagree & ~autonomous
        self.auton_count += autonomous
        self.ctrl_count += controllable
        self.updates += 1
        ctrl = self.controllable_mask
        changed = bool((self.autonomous_mask & ctrl).any())
        self.autonomous_mask &= ~ctrl
        new = autonomous & ~self.autonomous_mask & ~ctrl
        if new.any():
            grow = new.copy()
            if self.expand == 'component':
                for _color, px in components(frame):
                    if len(px) > LARGE_COMPONENT:
                        continue
                    if new[px[:, 0], px[:, 1]].any():
                        grow[px[:, 0], px[:, 1]] = True
            self.autonomous_mask |= grow & ~ctrl
            changed = True
        if changed:
            self.version += 1
        return dict(used=True, new_autonomous=int(new.sum()), autonomous_now=int(autonomous.sum()), controllable_now=int(controllable.sum()))

def fresh_autonomous(frame: np.ndarray, outcomes: Sequence[np.ndarray], quorum: float=1.0) -> Optional[np.ndarray]:
    if len(outcomes) < 2:
        return None
    tracker = MajorityEffectTracker(quorum)
    if not tracker.update(frame, outcomes)['used']:
        return None
    return tracker.autonomous_mask

class FreshMaskMemory(Memory):

    def __init__(self, mask: np.ndarray, block: int=4, minimum: int=5, window: int=20) -> None:
        super().__init__(mask, block, minimum, window)
        self.frames: List[np.ndarray] = []
        self._raw: set = set()
        self.remasks = 0
        y, x = np.indices((64, 64))
        self._bid = (y // block * (64 // block) + x // block) * 16

    def add(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, np.uint8)
        raw = frame.tobytes()
        if raw not in self._raw:
            self._raw.add(raw)
            self.frames.append(frame.copy())
        super().add(frame)

    def _encode_many(self, frames: np.ndarray) -> np.ndarray:
        k = len(frames)
        ids = (self._bid[None] + frames.astype(np.int64))[:, self.mask]
        ids = ids + (np.arange(k, dtype=np.int64) * 4096)[:, None]
        counts = np.bincount(ids.ravel(), minlength=k * 4096).reshape(k, 4096)
        return (counts / self.mask.sum()).astype(np.float32)

    def set_mask(self, mask: np.ndarray) -> bool:
        if int(mask.sum()) < MIN_MASK_PIXELS or np.array_equal(mask, self.mask):
            return False
        self.mask = mask.copy()
        observed, seen = ([], set())
        if self.frames:
            for f in self._encode_many(np.stack(self.frames)):
                b = f.tobytes()
                if b not in seen:
                    seen.add(b)
                    observed.append(f)
        self.observed = observed
        self.remasks += 1
        return True

def apply_fresh_mask(memory: FreshMaskMemory, base_mask: np.ndarray, frame: np.ndarray, predictions: Sequence[np.ndarray], quorum: float, stats) -> int:
    auton = None
    if memory.ready and len(predictions) >= 2:
        stats['automask_checked'] += 1
        auton = fresh_autonomous(frame, list(predictions), quorum)
    mask = base_mask
    if auton is not None and (base_mask & auton).any():
        cand = base_mask & ~auton
        if int(cand.sum()) >= MIN_MASK_PIXELS:
            mask = cand
        else:
            stats['mask_rejected_small'] += 1
    memory.set_mask(mask)
    n = int((base_mask & ~mask).sum())
    if n:
        stats['automask_steps'] += 1
        stats['automask_px'] += n
        stats['masked_px_max'] = max(stats['masked_px_max'], n)
    return n
