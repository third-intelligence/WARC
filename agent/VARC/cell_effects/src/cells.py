from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.novelty import feature
ABSTRACTIONS = ('raw', 'bar', 'hist4', 'auton', 'auton_c2', 'auton_c4')
MASKED = 16

def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:24]

def block_presence(frame: np.ndarray, block: int) -> np.ndarray:
    n = 64 // block
    onehot = np.eye(MASKED + 1, dtype=bool)[frame.astype(np.int64)]
    return onehot.reshape(n, block, n, block, MASKED + 1).any((1, 3))

def cell_key(name: str, frame: np.ndarray, *, bar_mask: np.ndarray, autonomous_mask: np.ndarray) -> str:
    frame = np.asarray(frame, np.uint8)
    if name == 'raw':
        return _digest(frame)
    if name == 'bar':
        return _digest(np.where(bar_mask, frame, 0))
    if name == 'hist4':
        return _digest(feature(frame, bar_mask))
    masked = np.where(autonomous_mask, MASKED, frame).astype(np.uint8)
    if name == 'auton':
        return _digest(masked)
    if name == 'auton_c2':
        return _digest(block_presence(masked, 2))
    if name == 'auton_c4':
        return _digest(block_presence(masked, 4))
    raise ValueError(f'unknown abstraction: {name}')

class CellCounter:

    def __init__(self, name: str, bar_mask: np.ndarray) -> None:
        if name not in ABSTRACTIONS:
            raise ValueError(f'unknown abstraction: {name}')
        self.name = name
        self.bar_mask = bar_mask.copy()
        self.frames: Dict[str, Tuple[np.ndarray, int]] = {}
        self._counts: Dict[str, int] = {}
        self._version = -1
        self._autonomous = np.zeros((64, 64), bool)

    def key(self, frame: np.ndarray, tracker) -> str:
        self._sync(tracker)
        return cell_key(self.name, frame, bar_mask=self.bar_mask, autonomous_mask=self._autonomous)

    def _sync(self, tracker) -> None:
        version = -1 if tracker is None else tracker.version
        if version == self._version:
            return
        self._version = version
        self._autonomous = np.zeros((64, 64), bool) if tracker is None else tracker.autonomous_mask.copy()
        self._counts = {}
        for frame, n in self.frames.values():
            k = cell_key(self.name, frame, bar_mask=self.bar_mask, autonomous_mask=self._autonomous)
            self._counts[k] = self._counts.get(k, 0) + n

    def count(self, frame: np.ndarray, tracker) -> int:
        k = self.key(frame, tracker)
        return self._counts.get(k, 0)

    def add(self, frame: np.ndarray, tracker) -> bool:
        frame = np.asarray(frame, np.uint8)
        k = self.key(frame, tracker)
        new = k not in self._counts
        self._counts[k] = self._counts.get(k, 0) + 1
        d = _digest(frame)
        old = self.frames.get(d)
        self.frames[d] = (frame.copy() if old is None else old[0], 1 if old is None else old[1] + 1)
        return new

    @property
    def distinct(self) -> int:
        return len(self._counts)
