from __future__ import annotations
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from object_abstraction.src.affordance import FrameObjects
from automask_agent.src.macro import MOVE_KEYS
MAX_AVATAR = 256
MAX_TARGET = 256
MATCH = 0.9

def _bbox(px: np.ndarray) -> Tuple[int, int, int, int]:
    y0, x0 = px.min(0)
    y1, x1 = px.max(0)
    return (int(y0), int(x0), int(y1), int(x1))

def rect_gap(a, b) -> int:
    dy = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    dx = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return max(dy, dx)

class AvatarModel:

    def __init__(self) -> None:
        self.votes: Counter = Counter()
        self.disp: Dict[int, Counter] = defaultdict(Counter)
        self.blocked = set()
        self.last_pos: Optional[Tuple[int, int]] = None

    def update(self, prev_objs: FrameObjects, nxt: np.ndarray, aid: int, mask: np.ndarray) -> None:
        if aid not in MOVE_KEYS:
            return
        prev = prev_objs.frame
        nxt = np.asarray(nxt, np.uint8)
        changed = (prev != nxt) & mask
        if not changed.any():
            here = self.locate(prev_objs)
            if here is not None:
                self.blocked.add(((here[0], here[1]), aid))
            return
        best = None
        for i, px in enumerate(prev_objs.pixels):
            if len(px) > MAX_AVATAR or not changed[px[:, 0], px[:, 1]].any():
                continue
            color = prev_objs.descs[i][1]
            moved = changed & (nxt == color)
            if not moved.any():
                continue
            ys, xs = np.nonzero(moved)
            y0, x0, y1, x1 = _bbox(px)
            cands = {(int(ys.min()) - y0, int(xs.min()) - x0), (int(ys.max()) - y1, int(xs.max()) - x1), (int(ys.min()) - y0, int(xs.max()) - x1), (int(ys.max()) - y1, int(xs.min()) - x0)}
            for dy, dx in cands:
                if (dy, dx) == (0, 0):
                    continue
                q = px + np.array([dy, dx])
                if q.min() < 0 or q.max() > 63:
                    continue
                match = float((nxt[q[:, 0], q[:, 1]] == color).mean())
                if match >= MATCH and (best is None or (match, len(px)) > best[0]):
                    best = ((match, len(px)), i, (dy, dx))
        if best is None:
            return
        _s, i, d = best
        self.votes[prev_objs.descs[i]] += 1
        self.disp[aid][d] += 1
        y0, x0, _y1, _x1 = _bbox(prev_objs.pixels[i])
        self.last_pos = (y0 + d[0], x0 + d[1])

    @property
    def desc(self):
        if not self.votes:
            return None
        d, n = self.votes.most_common(1)[0]
        return d if n >= 2 else None

    def moves(self) -> Dict[int, Tuple[int, int]]:
        out = {}
        for k, c in self.disp.items():
            (d, n), total = (c.most_common(1)[0], sum(c.values()))
            if n * 2 >= total:
                out[k] = d
        return out

    @property
    def ready(self) -> bool:
        return self.desc is not None and len(self.moves()) >= 2

    def locate(self, objs: FrameObjects) -> Optional[Tuple[int, int, int, int]]:
        d = self.desc
        if d is None:
            return None
        found = [_bbox(px) for desc, px in zip(objs.descs, objs.pixels) if desc == d]
        if not found:
            return None
        if self.last_pos is None or len(found) == 1:
            return found[0]
        ly, lx = self.last_pos
        return min(found, key=lambda b: abs(b[0] - ly) + abs(b[1] - lx))

def targets(objs: FrameObjects, avatar_desc, mask: np.ndarray) -> List[Tuple]:
    out = []
    for desc, px in zip(objs.descs, objs.pixels):
        if desc == avatar_desc or len(px) > MAX_TARGET:
            continue
        b = _bbox(px)
        cy, cx = ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        if not mask[cy, cx]:
            continue
        out.append(((desc, cy // 8, cx // 8), desc, b))
    return out

def bfs(start: Tuple[int, int], size: Tuple[int, int], moves: Dict[int, Tuple[int, int]], blocked, max_nodes: int=4096):
    h, w = size
    seen = {start: (0, -1)}
    queue = deque([start])
    while queue and len(seen) < max_nodes:
        p = queue.popleft()
        dist, first = seen[p]
        for k, (dy, dx) in sorted(moves.items()):
            if (p, k) in blocked:
                continue
            q = (p[0] + dy, p[1] + dx)
            if q in seen or q[0] < 0 or q[1] < 0 or (q[0] + h > 64) or (q[1] + w > 64):
                continue
            seen[q] = (dist + 1, k if first < 0 else first)
            queue.append(q)
    pos = np.array(list(seen.keys()), dtype=np.int64)
    dist = np.array([v[0] for v in seen.values()], dtype=np.int64)
    first = np.array([v[1] for v in seen.values()], dtype=np.int64)
    return (pos, dist, first)

def plan_to_targets(avatar_box, cand: List[Tuple], moves, blocked, touched, touched_descs):
    if not cand or not moves:
        return None
    y0, x0, y1, x1 = avatar_box
    h, w = (y1 - y0 + 1, x1 - x0 + 1)
    pos, dist, first = bfs((y0, x0), (h, w), moves, blocked)
    reach = max((max(abs(dy), abs(dx)) for dy, dx in moves.values()))
    best = None
    for tid, desc, b in cand:
        if tid in touched:
            continue
        gy = np.maximum(0, np.maximum(b[0] - (pos[:, 0] + h - 1) - 1, pos[:, 0] - b[2] - 1))
        gx = np.maximum(0, np.maximum(b[1] - (pos[:, 1] + w - 1) - 1, pos[:, 1] - b[3] - 1))
        ok = np.maximum(gy, gx) <= reach
        if not ok.any():
            continue
        j = int(np.flatnonzero(ok)[np.argmin(dist[ok])])
        rank = (desc in touched_descs, int(dist[j]))
        if best is None or rank < best[0]:
            best = (rank, tid, b, int(first[j]), int(dist[j]))
    if best is None:
        return None
    return (best[1], best[2], best[3], best[4])
