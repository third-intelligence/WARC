from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
ActionTriple = Tuple[int, int, int]

@dataclass
class Node:
    actions: List[ActionTriple]
    priorities: Dict[ActionTriple, int] = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    edges: Dict[ActionTriple, str] = field(default_factory=dict)
    unstable: Set[ActionTriple] = field(default_factory=set)

class Graph:

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.conflicts = 0

    def register(self, key: str, actions: Sequence[ActionTriple], priorities: Optional[Dict[ActionTriple, int]]=None) -> None:
        if key not in self.nodes:
            self.nodes[key] = Node(list(actions), dict(priorities or {}))
        else:
            self.nodes[key].actions = list(actions)
            self.nodes[key].priorities = dict(priorities or {})

    def options(self, key: str, available_actions: Sequence[ActionTriple]):
        node = self.nodes[key]
        fresh = [a for a in available_actions if node.counts[a] == 0]
        if fresh and min((node.priorities.get(a, 0) for a in fresh)) == 0:
            return ([a for a in fresh if node.priorities.get(a, 0) == 0], 'explore', None)
        queue = deque([(key, None, 0)])
        seen = {key}
        best = None
        while queue:
            current, first, distance = queue.popleft()
            nd = self.nodes[current]
            pool = available_actions if current == key else nd.actions
            possible = [a for a in pool if nd.counts[a] == 0]
            if possible:
                priority = min((nd.priorities.get(a, 0) for a in possible))
                rank = (priority, distance)
                if best is None or rank < best[0]:
                    best = (rank, first, [a for a in possible if nd.priorities.get(a, 0) == priority])
            for action, target in sorted(nd.edges.items()):
                if action in nd.unstable or target not in self.nodes or target in seen:
                    continue
                if action not in nd.actions:
                    continue
                if current == key and action not in available_actions:
                    continue
                seen.add(target)
                queue.append((target, action if first is None else first, distance + 1))
        if best is not None:
            _rank, first, opts = best
            if first is None:
                return (opts, 'explore', None)
            return ([first], 'return', node.edges[first])
        lowest = min((node.counts[a] for a in available_actions))
        return ([a for a in available_actions if node.counts[a] == lowest], 'reprobe', None)

    def record(self, source: str, action: ActionTriple, target: str, *, terminal: bool=False) -> bool:
        node = self.nodes[source]
        repeated = node.counts[action] > 0
        node.counts[action] += 1
        if action in node.edges and node.edges[action] != target:
            self.conflicts += 1
            node.unstable.add(action)
        if terminal:
            node.unstable.add(action)
        else:
            node.edges[action] = target
        return repeated

def choose(options: Sequence[ActionTriple], scores, rng, epsilon: float=0.2) -> ActionTriple:
    if scores is None or len(options) == 1 or rng.random() < epsilon:
        return options[rng.randrange(len(options))]
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(options),) or not np.isfinite(scores).all():
        raise ValueError('non-finite or misaligned candidate scores')
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-08))
    return options[int(best[rng.randrange(len(best))])]
