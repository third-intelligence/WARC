from __future__ import annotations
import heapq
import sys
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.perception import candidates, digest
from wide_agent.src.agent import group_by_prediction

def _children(frame: np.ndarray, available: Sequence[int], cap: int, rng) -> List:
    acts, pri = candidates(frame, available, with_groups=True)
    if len(acts) <= cap:
        return acts
    first = [a for a in acts if pri.get(a, 0) == 0][:cap]
    rest = [a for a in acts if pri.get(a, 0) != 0]
    fill = cap - len(first)
    return first + (rng.sample(rest, fill) if len(rest) > fill else rest)

def tree_rank(predictor, window, pool, memory, plan_rng, *, available: Sequence[int], reliability: float, max_expand: int=16, max_depth: int=4, child_cap: int=32, explore_bonus: float=0.01):
    _p, preds = predictor.predict(window, pool)
    groups = group_by_prediction(preds)
    reps = [g[0] for g in groups]
    raw, _f, _n = memory.novelty(preds[reps])
    details: Dict = dict(method='tree', calibrated=memory.ready, calibration_n=len(memory.errors), error_radius=memory.radius, memory_states=len(memory.observed), fallback=False, pool_raw=len(pool), groups=len(groups), expanded=0, deepest=1, deep_best=0)
    if not memory.ready:
        details['fallback'] = True
        return (groups, np.zeros(len(reps)), preds, details)
    w = float(np.clip(reliability, 0.3, 0.95))
    scores = raw.astype(float).copy()
    best_depth = np.ones(len(reps), int)
    current = window.frames[-1]
    seen = {digest(current)}
    heap: list = []
    tie = 0
    for g, r in enumerate(reps):
        dg = digest(preds[r])
        if dg in seen:
            continue
        seen.add(dg)
        child = window.clone()
        child.push(pool[r], preds[r])
        heapq.heappush(heap, (-(raw[g] + explore_bonus), tie, g, 1, child))
        tie += 1
    while heap and details['expanded'] < max_expand:
        _neg, _t, g, d, node = heapq.heappop(heap)
        if d >= max_depth:
            continue
        frame = node.frames[-1]
        kids = _children(frame, available, child_cap, plan_rng)
        if not kids:
            continue
        _p, fut = predictor.predict(node, kids)
        details['expanded'] += 1
        fresh = []
        for a, f in zip(kids, fut):
            dg = digest(f)
            if dg in seen:
                continue
            seen.add(dg)
            fresh.append((a, f))
        if not fresh:
            continue
        nov, _f, _n = memory.novelty([f for _a, f in fresh], depth=d + 1)
        disc = w ** d
        for (a, f), n in zip(fresh, nov):
            v = disc * float(n)
            if v > scores[g]:
                scores[g] = v
                best_depth[g] = d + 1
            details['deepest'] = max(details['deepest'], d + 1)
            if d + 1 < max_depth:
                child = node.clone()
                child.push(a, f)
                heapq.heappush(heap, (-disc * (float(n) + explore_bonus), tie, g, d + 1, child))
                tie += 1
    details['deep_best'] = int((best_depth > 1).sum())
    if float(scores.max()) <= 1e-08:
        details['fallback'] = True
    if not np.isfinite(scores).all():
        raise FloatingPointError('non-finite tree scores')
    return (groups, scores, preds, details)
