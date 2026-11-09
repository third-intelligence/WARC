from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from hist_wm.src.hist_data import PAD_ACTION, PAD_ID
ActionTriple = Tuple[int, int, int]

def norm_action(aid: int, x: int, y: int) -> ActionTriple:
    aid = int(aid)
    if aid == 6:
        return (aid, int(x), int(y))
    return (aid, -1, -1)

class HistoryWindow:

    def __init__(self, n: int, image_size: int=64) -> None:
        if n < 1:
            raise ValueError(f'n must be >= 1, got {n}')
        self.n = int(n)
        self.image_size = int(image_size)
        self.frames: List[np.ndarray] = []
        self.inter: List[ActionTriple] = []

    def reset(self, obs: np.ndarray) -> None:
        self.frames = [self._as_frame(obs)]
        self.inter = []

    def push(self, action: Sequence[int], next_obs: np.ndarray) -> None:
        if not self.frames:
            raise RuntimeError('HistoryWindow.push before reset')
        self.inter.append(norm_action(*action))
        self.frames.append(self._as_frame(next_obs))
        if len(self.frames) > self.n:
            drop = len(self.frames) - self.n
            self.frames = self.frames[drop:]
            self.inter = self.inter[drop:]
        assert len(self.inter) == len(self.frames) - 1

    @property
    def current(self) -> np.ndarray:
        return self.frames[-1]

    def __len__(self) -> int:
        return len(self.frames)

    def _as_frame(self, obs) -> np.ndarray:
        f = np.asarray(obs).astype(np.uint8)
        if f.shape != (self.image_size, self.image_size):
            raise ValueError(f'frame shape {f.shape} != ({self.image_size},{self.image_size})')
        return f.copy()

    def window_item(self, action: Sequence[int], next_obs: np.ndarray, level: int) -> tuple:
        return (list(self.frames), list(self.inter), norm_action(*action), self._as_frame(next_obs), int(level))

    def as_tensors(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k = (self.n, max(self.n - 1, 0))
        m = len(self.frames)
        obs = np.full((n, self.image_size, self.image_size), PAD_ID, dtype=np.int64)
        obs[n - m:] = np.stack(self.frames).astype(np.int64)
        aid = np.full((k,), PAD_ACTION[0], dtype=np.int64)
        ax = np.full((k,), PAD_ACTION[1], dtype=np.int64)
        ay = np.full((k,), PAD_ACTION[2], dtype=np.int64)
        for j, (a, x, y) in enumerate(self.inter):
            slot = k - len(self.inter) + j
            aid[slot], ax[slot], ay[slot] = (a, x, y)
        t_obs = torch.from_numpy(obs).to(device).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        t_aid = torch.from_numpy(aid).to(device).unsqueeze(0).expand(B, -1).contiguous()
        t_ax = torch.from_numpy(ax).to(device).unsqueeze(0).expand(B, -1).contiguous()
        t_ay = torch.from_numpy(ay).to(device).unsqueeze(0).expand(B, -1).contiguous()
        return (t_obs, t_aid, t_ax, t_ay)
