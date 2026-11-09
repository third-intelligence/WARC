"""Small, game-agnostic controls. No environment imports or game-ID lookup."""
import hashlib
import random
from collections import Counter

import numpy as np
from scipy import ndimage


def component_actions(frame):
    """One actual pixel nearest each 4-connected component's centroid.

    Exclude the dominant color, retaining uniform-coordinate fallback in agents.
    No HUD removal, game-specific thresholds, or semantic object inference.
    """
    grid = np.asarray(frame, dtype=np.uint8)
    background = int(np.bincount(grid.ravel(), minlength=16).argmax())
    actions = []
    for color in np.unique(grid):
        if color == background:
            continue
        labels, _ = ndimage.label(grid == color)
        for index, bounds in enumerate(ndimage.find_objects(labels), 1):
            if bounds is None:
                continue
            ys, xs = np.nonzero(labels[bounds] == index)
            nearest = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
            actions.append((6, int(xs[nearest] + bounds[1].start),
                            int(ys[nearest] + bounds[0].start)))
    return actions


def encode(action):
    kind, x, y = action
    return {"kind": kind, **({"x": x, "y": y} if kind == 6 else {})}


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, observation):
        kinds = [k for k in observation["available_actions"] if k != 0]
        kind = self.rng.choice(kinds)
        if kind == 6:
            return {"kind": 6, "x": self.rng.randrange(64), "y": self.rng.randrange(64)}
        return {"kind": kind}


class ComponentRandomAgent(RandomAgent):
    def act(self, observation):
        action = super().act(observation)
        if action["kind"] == 6 and self.rng.random() >= 0.1:
            candidates = component_actions(observation["frame"][-1])
            if candidates:
                return encode(self.rng.choice(candidates))
        return action


class CountAgent(RandomAgent):
    """Least-visited action per exact full-frame state; not Graph Explorer."""
    def __init__(self, seed=0):
        super().__init__(seed)
        self.counts = Counter()

    def act(self, observation):
        grid = np.asarray(observation["frame"][-1], dtype=np.uint8)
        state = (observation["levels_completed"], hashlib.sha256(grid.tobytes()).digest())
        candidates = [(k, None, None) for k in observation["available_actions"] if k not in (0, 6)]
        if 6 in observation["available_actions"]:
            clicks = component_actions(grid)
            if not clicks or self.rng.random() < 0.1:
                clicks = [(6, self.rng.randrange(64), self.rng.randrange(64))]
            candidates.extend(clicks)
        minimum = min(self.counts[state, a] for a in candidates)
        action = self.rng.choice([a for a in candidates if self.counts[state, a] == minimum])
        self.counts[state, action] += 1
        return encode(action)


BUILTINS = {"random": RandomAgent, "component_random": ComponentRandomAgent, "count": CountAgent}
