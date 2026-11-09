from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.graph import choose
from graph_agent.src.perception import candidates, window_key
from unified_wm.src.predictor import UnifiedMpcAgent
ARM_CONFIGS: Dict[str, Dict] = {'base_h2': dict(base_arm='native_h2', candidates=32, dedup=False, credit='none', topk=0, predict_batch=16), 'wide_h2': dict(base_arm='native_h2', candidates=256, dedup=False, credit='none', topk=32, predict_batch=64), 'wide_dedup_h2': dict(base_arm='native_h2', candidates=256, dedup=True, credit='verified', topk=32, predict_batch=64)}

def group_by_prediction(predictions: np.ndarray) -> List[List[int]]:
    groups: Dict[bytes, List[int]] = {}
    for i, frame in enumerate(predictions):
        groups.setdefault(np.ascontiguousarray(frame).tobytes(), []).append(i)
    return list(groups.values())

def rank_wide(predictor, window, pool, memory, plan_rng, *, branch: int=4, topk: int=0, dedup: bool=True, predictions=None):
    if predictions is None:
        _probs, predictions = predictor.predict(window, pool)
    groups = group_by_prediction(predictions) if dedup else [[i] for i in range(len(pool))]
    reps = [g[0] for g in groups]
    rep_preds = predictions[reps]
    raw, _features, nearest = memory.novelty(rep_preds)
    current = window.frames[-1]
    noop = [len(g) for g, r in zip(groups, reps) if np.array_equal(predictions[r], current)]
    details: Dict = dict(method='wide', calibrated=memory.ready, calibration_n=len(memory.errors), error_radius=memory.radius, memory_states=len(memory.observed), fallback=False, rollout_scored=0, pool_raw=len(pool), groups=len(groups), noop_group=noop[0] if noop else 0)
    endpoints = rep_preds.copy()
    if not memory.ready:
        details['fallback'] = True
        return (reps, np.zeros(len(reps)), predictions, endpoints, groups, details)
    scores = raw.copy()
    if len(reps) > 1:
        if topk and len(reps) > topk:
            jitter = np.array([plan_rng.random() for _ in reps])
            expand = sorted(np.lexsort((jitter, -raw))[:topk].tolist())
        else:
            expand = list(range(len(reps)))
        rep_actions = [pool[r] for r in reps]
        for j in expand:
            first = rep_actions[j]
            others = [a for a in rep_actions if a != first]
            children = [first] + plan_rng.sample(others, min(branch - 1, len(others)))
            imagined = window.clone()
            imagined.push(first, rep_preds[j])
            _p, future = predictor.predict(imagined, children)
            future_scores, _f, _n = memory.novelty(future, depth=2)
            best = int(np.argmax(future_scores))
            endpoints[j] = future[best]
            scores[j] = max(raw[j], future_scores[best] / 2)
            details['rollout_scored'] += len(children)
        details['expanded'] = len(expand)
    if float(scores.max()) <= 1e-08:
        details['fallback'] = True
    if not np.isfinite(scores).all():
        raise FloatingPointError('non-finite planning scores')
    return (reps, scores, predictions, endpoints, groups, details)

class WideMpcAgent(UnifiedMpcAgent):

    def __init__(self, *, dedup: bool=True, credit: str='verified', topk: int=32, predict_batch: int=64, **kw) -> None:
        if credit not in {'none', 'verified'}:
            raise ValueError(f'unknown credit mode: {credit}')
        super().__init__(**kw)
        self.dedup = bool(dedup)
        self.credit = credit if self.dedup else 'none'
        self.topk = int(topk)
        if self.predictor is not None:
            self.predictor.batch_size = int(predict_batch)
        self._siblings: List = []
        self.stats = dict(plans=0, pool_raw=0, groups=0, noop_group=0, expanded=0, credited=0, credit_events=0)

    def act(self, frame: np.ndarray, available: Sequence[int]):
        if self.predictor is None or self.method is None:
            self._siblings = []
            return super().act(frame, available)
        if self.pending is not None:
            raise RuntimeError('observe() was not called for the previous action')
        frame = np.asarray(frame, np.uint8)
        if self.window is None:
            self.reset_context(frame)
        if not np.array_equal(frame, self.window.frames[-1]):
            raise RuntimeError('observation/history drift')
        allowed, priorities = candidates(frame, available, with_groups=True)
        if not allowed:
            raise ValueError('no legal candidates')
        key = window_key(self.window, self.mask, available, use_history=False)
        self.graph.register(key, allowed, priorities)
        options, mode, expected = self.graph.options(key, allowed)
        pool_rng = random.Random(self.seed * 1000003 + self.steps + 38197)
        pool = pool_rng.sample(options, self.candidates_cap) if len(options) > self.candidates_cap else list(options)
        plan_rng = random.Random(self.seed * 1000003 + self.steps + 8371)
        reps, scores, preds, endpoints, groups, plan = rank_wide(self.predictor, self.window, pool, self.memory, plan_rng, branch=self.branch, topk=self.topk, dedup=self.dedup)
        rep_actions = [pool[r] for r in reps]
        action = choose(rep_actions, None if plan.get('fallback') else scores, self.rng, self.epsilon)
        j = rep_actions.index(action)
        self.predicted_frame = preds[reps[j]].copy()
        self.endpoint = endpoints[j].copy()
        self._siblings = [pool[i] for i in groups[j][1:]] if plan['calibrated'] else []
        self.last_plan = dict(plan, mode=mode, candidate_count=len(allowed), pool_size=len(pool), selected=j, group_size=len(groups[j]))
        s = self.stats
        s['plans'] += 1
        s['pool_raw'] += len(pool)
        s['groups'] += len(groups)
        s['noop_group'] += plan['noop_group']
        s['expanded'] += plan.get('expanded', 0)
        self.pending = (action, key, expected)
        return action

    def observe(self, action, next_frame, *, next_available, state: str, new_level: bool) -> None:
        key = self.pending[1] if self.pending is not None else None
        siblings, self._siblings = (self._siblings, [])
        super().observe(action, next_frame, next_available=next_available, state=state, new_level=new_level)
        ev = self.last_eval or {}
        if self.credit == 'verified' and key is not None and siblings and ev.get('valid') and (ev.get('exact') == 1) and (key in self.graph.nodes):
            node = self.graph.nodes[key]
            for a in siblings:
                node.counts[a] += 1
            self.stats['credited'] += len(siblings)
            self.stats['credit_events'] += 1

    def metrics(self) -> Dict:
        result = super().metrics()
        s = self.stats
        n = max(1, s['plans'])
        result.update(dedup=self.dedup, credit=self.credit, topk=self.topk, candidates_cap=self.candidates_cap, mean_pool_raw=s['pool_raw'] / n, mean_groups=s['groups'] / n, mean_noop_group=s['noop_group'] / n, mean_expanded=s['expanded'] / n, credited_actions=s['credited'], credit_events=s['credit_events'])
        return result
