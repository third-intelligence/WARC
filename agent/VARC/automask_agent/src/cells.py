from __future__ import annotations
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from cell_effects.src.cells import MASKED, block_presence
ActionTriple = Tuple[int, int, int]

def cell_key(frame: np.ndarray, mask: np.ndarray) -> str:
    masked = np.where(mask, np.asarray(frame, np.uint8), MASKED).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(block_presence(masked, 4)).tobytes()).hexdigest()[:24]

@dataclass
class Cell:
    path: Tuple[ActionTriple, ...]
    traj: Tuple[str, ...]
    visits: int = 1
    chosen: int = 0
    gains: int = 0
    fails: int = 0

class CellArchive:

    def __init__(self) -> None:
        self.cells: Dict[str, Cell] = {}

    def __len__(self) -> int:
        return len(self.cells)

    def visit(self, key: str, path: Tuple[ActionTriple, ...], traj: Tuple[str, ...]) -> bool:
        cell = self.cells.get(key)
        if cell is None:
            self.cells[key] = Cell(tuple(path), tuple(traj))
            return True
        cell.visits += 1
        if len(path) < len(cell.path):
            cell.path, cell.traj = (tuple(path), tuple(traj))
        return False

    def select(self, current: str, remaining: int, rng, *, reserve: int=20, cost_weight: float=0.1) -> Optional[str]:
        best, best_score = (None, -math.inf)
        for key, c in self.cells.items():
            if key == current or c.fails >= 2 or (not c.path):
                continue
            cost = len(c.path) + 1
            if cost + reserve > remaining:
                continue
            s = 1.0 / math.sqrt(1 + c.chosen) + 1.0 / math.sqrt(1 + c.visits) + 0.5 * c.gains / (1 + c.chosen) - cost_weight * cost / max(1, remaining)
            s += 0.05 * rng.random()
            if s > best_score:
                best, best_score = (key, s)
        return best
