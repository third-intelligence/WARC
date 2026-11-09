from __future__ import annotations
from collections import deque
from typing import Dict, List, Sequence
import numpy as np

def feature(frame: np.ndarray, mask: np.ndarray, block: int=4) -> np.ndarray:
    frame = np.asarray(frame)
    assert frame.shape == mask.shape == (64, 64) and 64 % block == 0
    assert frame.min() >= 0 and frame.max() < 16 and mask.any()
    y, x = np.indices(frame.shape)
    ids = (y // block * (64 // block) + x // block) * 16 + frame
    return (np.bincount(ids[mask], minlength=(64 // block) ** 2 * 16) / mask.sum()).astype(np.float32)

def distance(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).sum(axis=-1) / 2

class Memory:

    def __init__(self, mask: np.ndarray, block: int=4, minimum: int=5, window: int=20) -> None:
        self.mask = mask.copy()
        self.block = block
        self.minimum = minimum
        self.observed: List[np.ndarray] = []
        self.errors: deque = deque(maxlen=window)

    def encode(self, frame: np.ndarray) -> np.ndarray:
        return feature(frame, self.mask, self.block)

    def add(self, frame: np.ndarray) -> None:
        f = self.encode(frame)
        if not any((np.array_equal(f, old) for old in self.observed)):
            self.observed.append(f)

    def observe_error(self, prediction: np.ndarray, actual: np.ndarray) -> None:
        self.errors.append(float(distance(self.encode(prediction), self.encode(actual))))

    @property
    def ready(self) -> bool:
        return len(self.errors) >= self.minimum

    @property
    def radius(self) -> float:
        return float(np.median(self.errors)) if self.errors else 0.0

    def novelty(self, frames: Sequence[np.ndarray], depth: int=1):
        features = np.stack([self.encode(f) for f in frames])
        memories = np.stack(self.observed)
        nearest = np.array([distance(memories, f).min() for f in features])
        return (np.maximum(0.0, nearest - depth * self.radius), features, nearest)

def rank(method: str, predictor, window, pool, memory: Memory, plan_rng, branch: int=4):
    if method not in {'novelty', 'rollout'}:
        raise ValueError(f'unsupported ranking method: {method}')
    probs, predictions = predictor.predict(window, pool)
    raw, features, nearest = memory.novelty(predictions)
    details: Dict = dict(method=method, calibrated=memory.ready, calibration_n=len(memory.errors), error_radius=memory.radius, nearest_distances=nearest.tolist(), one_step_novelty=raw.tolist(), memory_states=len(memory.observed), fallback=False, rollout_scored=0)
    endpoints = predictions.copy()
    if not memory.ready:
        scores = np.zeros(len(pool))
        details['fallback'] = True
    else:
        scores = raw.copy()
        if method == 'rollout' and len(pool) > 1:
            branches, terminal_scores = ([], [])
            for i, first in enumerate(pool):
                others = [a for a in pool if a != first]
                children = [first] + plan_rng.sample(others, min(branch - 1, len(others)))
                imagined = window.clone()
                imagined.push(first, predictions[i])
                _p, future = predictor.predict(imagined, children)
                future_scores, _f, _n = memory.novelty(future, depth=2)
                best = int(np.argmax(future_scores))
                endpoints[i] = future[best]
                scores[i] = max(raw[i], future_scores[best] / 2)
                branches.append([list(a) for a in children])
                terminal_scores.append(future_scores.tolist())
                details['rollout_scored'] += len(children)
            details.update(branches=branches, terminal_scores=terminal_scores)
        if float(scores.max()) <= 1e-08:
            details['fallback'] = True
    if not np.isfinite(scores).all():
        raise FloatingPointError('non-finite planning scores')
    return (probs, predictions, scores, details, endpoints)
