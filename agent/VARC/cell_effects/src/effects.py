from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Sequence
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.perception import LARGE_COMPONENT, components
EXPAND_MODES = ('component', 'none')

class EffectTracker:

    def __init__(self, expand: str='component') -> None:
        if expand not in EXPAND_MODES:
            raise ValueError(f'unknown expand mode: {expand}')
        self.expand = expand
        self.autonomous_mask = np.zeros((64, 64), bool)
        self.auton_count = np.zeros((64, 64), np.int32)
        self.ctrl_count = np.zeros((64, 64), np.int32)
        self.updates = 0
        self.version = 0

    @property
    def controllable_mask(self) -> np.ndarray:
        return self.ctrl_count > 0

    def update(self, frame: np.ndarray, outcomes: Sequence[np.ndarray]) -> Dict:
        if len(outcomes) < 2:
            return dict(used=False, new_autonomous=0)
        frame = np.asarray(frame, np.uint8)
        outs = np.stack([np.asarray(o, np.uint8) for o in outcomes])
        same = (outs == outs[0]).all(0)
        controllable = ~same
        if not controllable.any():
            return dict(used=False, new_autonomous=0)
        autonomous = same & (outs[0] != frame)
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

    def diagnostics(self) -> Dict:
        return dict(masked_frac=float(self.autonomous_mask.mean()), controllable_frac=float(self.controllable_mask.mean()), tracker_updates=self.updates, mask_version=self.version)
