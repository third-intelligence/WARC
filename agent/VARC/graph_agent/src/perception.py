from __future__ import annotations
import hashlib
from typing import Dict, List, Sequence, Tuple
import numpy as np
ActionTriple = Tuple[int, int, int]
CLICK_ACTION_ID = 6
LARGE_COMPONENT = 1024

def digest(frame) -> str:
    return hashlib.sha256(np.asarray(frame, dtype=np.uint8).tobytes()).hexdigest()[:24]

def components(frame: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    h, w = frame.shape
    seen = np.zeros((h, w), bool)
    result: List[Tuple[int, np.ndarray]] = []
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            color, todo, pixels = (int(frame[y, x]), [(y, x)], [])
            seen[y, x] = True
            while todo:
                yy, xx = todo.pop()
                pixels.append((yy, xx))
                for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (not seen[ny, nx]) and (frame[ny, nx] == color):
                        seen[ny, nx] = True
                        todo.append((ny, nx))
            result.append((color, np.asarray(pixels)))
    return result

def initial_mask(frame: np.ndarray, mode: str='border_bar') -> np.ndarray:
    mask = np.ones(frame.shape, bool)
    if mode == 'none':
        return mask
    if mode != 'border_bar':
        raise ValueError(f'unknown mask mode: {mode}')
    for _color, px in components(frame):
        y0, x0 = px.min(0)
        y1, x1 = px.max(0)
        if y1 - y0 < 2 and x1 - x0 >= 31 and (y1 < 2 or y0 >= 62):
            mask[y0:y1 + 1] = False
    return mask

def candidates(frame: np.ndarray, available: Sequence[int], with_groups: bool=False):
    available = sorted(set((int(a) for a in available)))
    out: List[ActionTriple] = [(a, -1, -1) for a in available if 0 < a != CLICK_ACTION_ID]
    priorities: Dict[ActionTriple, int] = {a: 0 for a in out}
    if CLICK_ACTION_ID in available:
        regions = components(frame)
        regions.sort(key=lambda r: (len(r[1]) > LARGE_COMPONENT, len(r[1]), tuple(r[1][0])))
        for _color, px in regions:
            center = px.mean(0)
            y, x = px[np.square(px - center).sum(1).argmin()]
            action = (CLICK_ACTION_ID, int(x), int(y))
            out.append(action)
            priorities[action] = 0 if len(px) <= LARGE_COMPONENT else 2
        for y in range(4, 64, 8):
            for x in range(4, 64, 8):
                action = (CLICK_ACTION_ID, x, y)
                out.append(action)
                priorities.setdefault(action, 1)
    actions = list(dict.fromkeys(out))
    return (actions, priorities) if with_groups else actions

def window_key(window, mask: np.ndarray, available: Sequence[int], use_history: bool=False) -> str:
    frames = window.frames if use_history else window.frames[-1:]
    canonical = [digest(np.where(mask, f, 0)) for f in frames]
    actions = list(window.inter) if use_history else []
    payload = repr((canonical, actions, sorted((int(a) for a in available))))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]
