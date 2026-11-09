from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
MOVE_KEYS = (1, 2, 3, 4)

def macro_candidates(available: Sequence[int]) -> List[Tuple[str, int]]:
    avail = {int(a) for a in available}
    return [('M', a) for a in MOVE_KEYS if a in avail]

def is_macro(action) -> bool:
    return isinstance(action, tuple) and len(action) == 2 and (action[0] == 'M')

def score_macros(predictor, window, macros, memory, *, length: int=8, gamma: float=0.85):
    out = []
    for m in macros:
        a = (int(m[1]), -1, -1)
        w = window.clone()
        best, steps, endpoint = (0.0, 0, None)
        for d in range(1, length + 1):
            _p, fut = predictor.predict(w, [a])
            f = fut[0]
            if np.array_equal(f, w.frames[-1]):
                break
            w.push(a, f)
            steps, endpoint = (d, f)
            if d >= 2 and memory.ready:
                n = float(memory.novelty([f], depth=d)[0][0])
                best = max(best, gamma ** (d - 1) * n)
        out.append((best, steps, endpoint))
    return out

def macro_should_stop(prev: np.ndarray, nxt: np.ndarray, mask: np.ndarray, *, valid: bool, first_change, remaining: int) -> Tuple[bool, str, int]:
    n = int(((np.asarray(prev) != np.asarray(nxt)) & mask).sum())
    if not valid:
        return (True, 'invalid', n)
    if n == 0:
        return (True, 'noop', n)
    if first_change is not None and abs(n - first_change) > max(4, first_change / 2):
        return (True, 'event', n)
    if remaining <= 0:
        return (True, 'length', n)
    return (False, '', n)
