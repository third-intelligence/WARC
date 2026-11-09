from __future__ import annotations
import math
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.graph import choose
from graph_agent.src.perception import candidates, window_key
from graph_agent.src.window import norm_action
from wide_agent.src.agent import WideMpcAgent, rank_wide
from object_abstraction.src.affordance import ObjectCache
from automask_agent.src.automask import FreshMaskMemory, apply_fresh_mask
from automask_agent.src.cells import CellArchive, cell_key
from automask_agent.src.macro import MOVE_KEYS, is_macro, macro_candidates, macro_should_stop, score_macros
from automask_agent.src.objnav import AvatarModel, plan_to_targets, rect_gap, targets
from automask_agent.src.tree import tree_rank
RESET = (0, -1, -1)
_WIDE = dict(base_arm='native_h2', candidates=256, dedup=False, credit='none', topk=32, predict_batch=64)
_OFF = dict(macro=False, restore=False, objnav=False, bandit=False, tree=False, automask='')
ARM_CONFIGS: Dict[str, Dict] = {'wide_h2': dict(_WIDE, **_OFF), 'macro_h2': dict(_WIDE, **dict(_OFF, macro=True)), 'restore_h2': dict(_WIDE, **dict(_OFF, restore=True)), 'objnav_h2': dict(_WIDE, **dict(_OFF, objnav=True)), 'bandit_h2': dict(_WIDE, **dict(_OFF, bandit=True)), 'tree_h2': dict(_WIDE, **dict(_OFF, tree=True)), 'automask_h2': dict(_WIDE, **dict(_OFF, automask='majority')), 'automask_strict_h2': dict(_WIDE, **dict(_OFF, automask='strict'))}
DEFAULT_PARAMS = dict(macro_len=8, macro_gamma=0.85, stall=30, force_every=60, restore_reserve=20, restore_cost=0.1, nav_timeout_mult=3, nav_timeout_add=10, nav_grace=10, episode=20, episode_after=10, ucb_c=0.5, bandit_discount=0.95, levelup_bonus=5.0, tree_expand=16, tree_depth=4, tree_child_cap=32, tree_bonus=0.01, automask_quorum=0.9)
STRATEGIES = ('novelty', 'macro', 'restore', 'objnav', 'random')

class AutomaskAgent(WideMpcAgent):

    def __init__(self, *, macro: bool=False, restore: bool=False, objnav: bool=False, bandit: bool=False, tree: bool=False, automask: str='', run_budget: int=1000, params: Optional[Dict]=None, **kw) -> None:
        if tree and (macro or restore or objnav or bandit):
            raise ValueError('tree is defined as a standalone arm')
        if automask not in ('', 'majority', 'strict'):
            raise ValueError(f'unknown automask mode: {automask}')
        if automask and (macro or restore or objnav or bandit or tree):
            raise ValueError('automask is defined as a standalone arm')
        self.automask = automask
        self.base_mask: Optional[np.ndarray] = None
        self.use_macro, self.use_restore, self.use_objnav = (bool(macro), bool(restore), bool(objnav))
        self.use_bandit, self.use_tree = (bool(bandit), bool(tree))
        self.run_budget = int(run_budget)
        unknown = set(params or {}) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f'unknown params: {sorted(unknown)}')
        self.p = dict(DEFAULT_PARAMS, **params or {})
        self.archive = CellArchive()
        self.level_start_cell: Optional[str] = None
        self.path: list = []
        self.traj: list = []
        self.last_new_step = 0
        self.last_special_step = 0
        self.queue: deque = deque()
        self.restore_target: Optional[str] = None
        self.restore_credit_until = -1
        self.restore_ok = True
        self.macro_state: Optional[Dict] = None
        self.nav: Optional[Dict] = None
        self.touched, self.touched_descs = (set(), set())
        self.avatar = AvatarModel() if objnav or bandit else None
        self.objects = ObjectCache()
        self.recent_exact: deque = deque(maxlen=20)
        self.episode: Optional[Dict] = None
        self.ucb_n = {s: 0.0 for s in STRATEGIES}
        self.ucb_r = {s: 0.0 for s in STRATEGIES}
        self.pulls: Counter = Counter()
        self.rewards: Dict[str, list] = {s: [] for s in STRATEGIES}
        self._mode: Optional[str] = None
        self.x: Counter = Counter()
        super().__init__(**kw)
        self.xrng = random.Random(self.seed * 7919 + 36)

    def _cell(self, frame) -> str:
        return cell_key(frame, self.base_mask)

    def reset_context(self, frame: np.ndarray, *, new_level: bool=False) -> None:
        fresh = self.memory is None or new_level
        super().reset_context(frame, new_level=new_level)
        if fresh:
            self.base_mask = self.mask.copy()
            if self.automask:
                self.memory = FreshMaskMemory(self.mask, 4, 5, 20)
                self.memory.add(np.asarray(frame, np.uint8))
            self.archive = CellArchive()
            self.level_start_cell = self._cell(frame)
            self.archive.visit(self.level_start_cell, (), ())
            self.last_new_step = self.last_special_step = self.steps
            self.queue.clear()
            self.restore_target = None
            self.macro_state = self.nav = None
            self.touched, self.touched_descs = (set(), set())
            if self.avatar is not None:
                self.avatar.blocked = set()
        self.path, self.traj = ([], [])

    def _stalled(self) -> bool:
        return self.steps - self.last_new_step >= self.p['stall']

    def _due(self) -> bool:
        f = self.p['force_every']
        return self._stalled() or (f > 0 and self.steps - self.last_special_step >= f)

    def _emit(self, frame, available, action, mode: str, predicted=None):
        allowed, priorities = candidates(frame, available, with_groups=True)
        key = window_key(self.window, self.mask, available, use_history=False)
        self.graph.register(key, allowed, priorities)
        self.predicted_frame = None if predicted is None else predicted.copy()
        self.endpoint = self.predicted_frame
        self._siblings = []
        self.last_plan = dict(mode=mode, fallback=False, calibrated=self.memory.ready, error_radius=self.memory.radius, memory_states=len(self.memory.observed), candidate_count=len(allowed))
        self.pending = (tuple(action), key, None)
        self._mode = mode
        return tuple(action)

    def act(self, frame: np.ndarray, available: Sequence[int]):
        if self.predictor is None or self.method is None:
            return super().act(frame, available)
        if self.pending is not None:
            raise RuntimeError('observe() was not called for the previous action')
        frame = np.asarray(frame, np.uint8)
        if self.window is None:
            self.reset_context(frame)
        if not np.array_equal(frame, self.window.frames[-1]):
            raise RuntimeError('observation/history drift')
        avail = [int(a) for a in available]
        if self.queue:
            return self._emit(frame, avail, self.queue[0]['action'], 'restore')
        if self.macro_state is not None:
            if self.macro_state['aid'] in avail:
                self.macro_state['remaining'] -= 1
                return self._emit(frame, avail, (self.macro_state['aid'], -1, -1), 'macro')
            self.x['macro_stop_unavailable'] += 1
            self.x['macro_steps'] += self.macro_state['steps']
            self.macro_state = None
        strategy = self._strategy(frame, avail) if self.use_bandit else None
        if self.use_restore and self._due() and self._start_restore():
            return self._emit(frame, avail, self.queue[0]['action'], 'restore')
        if strategy == 'restore' and self.episode.get('fresh') and self._start_restore():
            self.episode['fresh'] = False
            return self._emit(frame, avail, self.queue[0]['action'], 'restore')
        nav_on = self.use_objnav and (self.nav is not None or self._due()) or (strategy == 'objnav' and (self.nav is not None or self.episode.get('fresh')))
        if nav_on and self.avatar.ready:
            act = self._nav_step(frame, avail)
            if self.episode is not None:
                self.episode['fresh'] = False
            if act is not None:
                return self._emit(frame, avail, act, 'nav')
            if self.nav is None:
                self.last_special_step = self.steps
        if strategy == 'random':
            allowed, _pri = candidates(frame, avail, with_groups=True)
            return self._emit(frame, avail, allowed[self.xrng.randrange(len(allowed))], 'random')
        if self.use_tree:
            return self._plan_tree(frame, avail)
        if self.use_macro or strategy == 'macro':
            return self._plan_wide(frame, avail, macros=True)
        if self.automask:
            return self._plan_wide(frame, avail, macros=False)
        self._mode = 'plan'
        return super().act(frame, avail)

    def _plan_wide(self, frame, available, *, macros: bool):
        allowed, priorities = candidates(frame, available, with_groups=True)
        key = window_key(self.window, self.mask, available, use_history=False)
        self.graph.register(key, allowed, priorities)
        options, mode, expected = self.graph.options(key, allowed)
        pool_rng = random.Random(self.seed * 1000003 + self.steps + 38197)
        pool = pool_rng.sample(options, self.candidates_cap) if len(options) > self.candidates_cap else list(options)
        plan_rng = random.Random(self.seed * 1000003 + self.steps + 8371)
        preds0 = None
        if self.automask:
            _p, preds0 = self.predictor.predict(self.window, pool)
            apply_fresh_mask(self.memory, self.base_mask, frame, preds0, self._quorum(), self.x)
        reps, scores, preds, endpoints, groups, plan = rank_wide(self.predictor, self.window, pool, self.memory, plan_rng, branch=self.branch, topk=self.topk, dedup=self.dedup, predictions=preds0)
        rep_actions = [pool[r] for r in reps]
        ms = macro_candidates(available) if macros else []
        mscores = [0.0] * len(ms)
        if ms and self.memory.ready:
            res = score_macros(self.predictor, self.window, ms, self.memory, length=self.p['macro_len'], gamma=self.p['macro_gamma'])
            mscores = [r[0] for r in res]
            self.x['macro_scored'] += len(ms)
        combined = rep_actions + ms
        cs = np.concatenate([np.asarray(scores, float), np.asarray(mscores, float)])
        use = self.memory.ready and float(cs.max()) > 1e-08
        action = choose(combined, cs if use else None, self.rng, self.epsilon)
        if is_macro(action):
            a = (int(action[1]), -1, -1)
            self.macro_state = dict(aid=a[0], remaining=self.p['macro_len'] - 1, first=None, steps=0)
            self.x['macro_started'] += 1
            self.predicted_frame = preds[pool.index(a)].copy() if a in pool else None
            self.endpoint = self.predicted_frame
            mode, self._mode = ('macro', 'macro')
        else:
            j = rep_actions.index(action)
            a = action
            self.predicted_frame = preds[reps[j]].copy()
            self.endpoint = endpoints[j].copy()
            self._mode = 'plan'
        self._siblings = []
        self.last_plan = dict(plan, mode=mode, candidate_count=len(allowed), pool_size=len(pool), fallback=not use, macro_candidates=len(ms))
        s = self.stats
        s['plans'] += 1
        s['pool_raw'] += len(pool)
        s['groups'] += len(groups)
        s['noop_group'] += plan['noop_group']
        s['expanded'] += plan.get('expanded', 0)
        self.pending = (a, key, expected)
        return a

    def _quorum(self) -> float:
        return 1.0 if self.automask == 'strict' else float(self.p['automask_quorum'])

    def _plan_tree(self, frame, available):
        allowed, priorities = candidates(frame, available, with_groups=True)
        key = window_key(self.window, self.mask, available, use_history=False)
        self.graph.register(key, allowed, priorities)
        node = self.graph.nodes[key]
        options, mode, expected = self.graph.options(key, allowed)
        pool_rng = random.Random(self.seed * 1000003 + self.steps + 38197)
        pool = pool_rng.sample(options, self.candidates_cap) if len(options) > self.candidates_cap else list(options)
        plan_rng = random.Random(self.seed * 1000003 + self.steps + 8371)
        rel = float(np.mean(self.recent_exact)) if len(self.recent_exact) >= 5 else 0.5
        groups, scores, preds, plan = tree_rank(self.predictor, self.window, pool, self.memory, plan_rng, available=available, reliability=rel, max_expand=self.p['tree_expand'], max_depth=self.p['tree_depth'], child_cap=self.p['tree_child_cap'], explore_bonus=self.p['tree_bonus'])
        if plan['fallback'] or len(pool) == 1 or self.rng.random() < self.epsilon:
            idx = self.rng.randrange(len(pool))
        else:
            best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-08))
            g = int(best[self.rng.randrange(len(best))])
            idx = min(groups[g], key=lambda i: (node.counts[pool[i]], i))
        action = pool[idx]
        self.predicted_frame = preds[idx].copy()
        self.endpoint = self.predicted_frame
        self._siblings = []
        self.last_plan = dict(plan, mode=mode, candidate_count=len(allowed), pool_size=len(pool), reliability=rel)
        s = self.stats
        s['plans'] += 1
        s['pool_raw'] += len(pool)
        s['groups'] += len(groups)
        s['expanded'] += plan['expanded']
        self.x['tree_deep_best'] += plan['deep_best']
        self.x['tree_deepest_sum'] += plan['deepest']
        self._mode = 'plan'
        self.pending = (action, key, expected)
        return action

    def _start_restore(self) -> bool:
        if not self.restore_ok:
            return False
        remaining = self.run_budget - self.steps
        cur = self._cell(self.window.frames[-1])
        target = self.archive.select(cur, remaining, self.xrng, reserve=self.p['restore_reserve'], cost_weight=self.p['restore_cost'])
        if target is None:
            return False
        cell = self.archive.cells[target]
        cell.chosen += 1
        self.queue = deque([dict(action=RESET, expect=self.level_start_cell)] + [dict(action=tuple(a), expect=c) for a, c in zip(cell.path, cell.traj)])
        self.restore_target = target
        self.last_new_step = self.last_special_step = self.steps
        self.x['restore_started'] += 1
        self.x['restore_len'] += len(cell.path) + 1
        return True

    def _nav_step(self, frame, available):
        objs = self.objects.get(frame)
        box = self.avatar.locate(objs)
        if box is None:
            self.nav = None
            return None
        cand = self._stable_targets(objs)
        moves = {k: d for k, d in self.avatar.moves().items() if k in available}
        if not moves:
            return None
        reach = max((max(abs(dy), abs(dx)) for dy, dx in moves.values()))
        blocked = self.avatar.blocked
        for _ in range(8):
            if self.nav is None:
                res = plan_to_targets(box, cand, moves, blocked, self.touched, self.touched_descs)
                if res is None:
                    return None
                tid, b, first, dist = res
                self.nav = dict(tid=tid, steps=0, limit=self.p['nav_timeout_mult'] * dist + self.p['nav_timeout_add'])
                self.x['nav_started'] += 1
                self.last_special_step = self.steps
            tid = self.nav['tid']
            here = [c for c in cand if c[0] == tid]
            if not here:
                self._nav_finish('changed')
                continue
            if min((rect_gap(box, c[2]) for c in here)) <= reach:
                self._nav_finish('reached')
                continue
            if self.nav['steps'] >= self.nav['limit']:
                self._nav_finish('timeout')
                continue
            res = plan_to_targets(box, here, moves, blocked, set(), set())
            if res is None:
                self._nav_finish('unreachable')
                continue
            if res[2] < 0:
                self._nav_finish('reached')
                continue
            return (int(res[2]), -1, -1)
        return None

    def _stable_targets(self, objs):
        cand = targets(objs, self.avatar.desc, self.mask)
        if len(self.window.frames) < 2:
            return cand
        before = {c[0] for c in targets(self.objects.get(self.window.frames[-2]), self.avatar.desc, self.mask)}
        return [c for c in cand if c[0] in before]

    def _nav_finish(self, why: str) -> None:
        tid = self.nav['tid']
        self.touched.add(tid)
        if why == 'reached':
            self.touched_descs.add(tid[0])
        self.x[f'nav_{why}'] += 1
        self.nav = None
        self.last_special_step = self.steps
        self.last_new_step = min(self.last_new_step, self.steps - self.p['stall'] + self.p['nav_grace'])

    def _strategy(self, frame, available) -> str:
        ep = self.episode
        if ep is not None:
            busy = self.queue or self.macro_state is not None or self.nav is not None
            s = ep['strategy']
            need = self.p['episode_after'] if s in ('restore', 'objnav') else self.p['episode']
            done = ep['free'] >= need if s in ('restore', 'objnav') else ep['steps'] >= need
            if busy or not done:
                return s
            self._close_episode()
        avail = []
        for s in STRATEGIES:
            if s == 'macro' and (not macro_candidates(available)):
                continue
            if s == 'restore' and (not self.restore_ok or self.archive.select(self._cell(frame), self.run_budget - self.steps, self.xrng, reserve=self.p['restore_reserve'], cost_weight=self.p['restore_cost']) is None):
                continue
            if s == 'objnav' and (not self._nav_available(frame, available)):
                continue
            avail.append(s)
        untried = [s for s in avail if self.pulls[s] == 0]
        if untried:
            s = untried[0]
        else:
            total = sum((self.ucb_n[t] for t in avail))
            s = max(avail, key=lambda t: self.ucb_r[t] / max(self.ucb_n[t], 1e-09) + self.p['ucb_c'] * math.sqrt(math.log(max(total, 1.0) + 1.0) / max(self.ucb_n[t], 1e-09)) + 1e-09 * self.xrng.random())
        self.pulls[s] += 1
        self.episode = dict(strategy=s, steps=0, free=0, new=0, levels=0, fresh=True)
        return s

    def _nav_available(self, frame, available) -> bool:
        if self.avatar is None or not self.avatar.ready:
            return False
        objs = self.objects.get(frame)
        box = self.avatar.locate(objs)
        if box is None:
            return False
        moves = {k: d for k, d in self.avatar.moves().items() if k in available}
        cand = self._stable_targets(objs)
        return plan_to_targets(box, cand, moves, self.avatar.blocked, self.touched, self.touched_descs) is not None

    def _close_episode(self) -> None:
        ep, self.episode = (self.episode, None)
        if ep is None or ep['steps'] == 0:
            return
        r = (ep['new'] + self.p['levelup_bonus'] * ep['levels']) / ep['steps']
        g = self.p['bandit_discount']
        for t in STRATEGIES:
            self.ucb_n[t] *= g
            self.ucb_r[t] *= g
        self.ucb_n[ep['strategy']] += 1.0
        self.ucb_r[ep['strategy']] += r
        self.rewards[ep['strategy']].append(r)

    def observe(self, action, next_frame, *, next_available, state: str, new_level: bool) -> None:
        if self.predictor is None or self.method is None:
            return super().observe(action, next_frame, next_available=next_available, state=state, new_level=new_level)
        mode, self._mode = (self._mode, None)
        if self.pending is None:
            mode = 'runner_reset'
        prev = self.window.frames[-1].copy() if self.window is not None else None
        if mode == 'restore':
            self.predictor.observe = lambda *a, **k: None
        try:
            super().observe(action, next_frame, next_available=next_available, state=state, new_level=new_level)
        finally:
            if mode == 'restore':
                del self.predictor.observe
        nxt = np.asarray(next_frame, np.uint8)
        aid = int(action[0])
        valid = aid != 0 and state == 'NOT_FINISHED' and (not new_level)
        ev = self.last_eval or {}
        if ev.get('valid') and 'exact' in ev:
            self.recent_exact.append(int(ev['exact']))
        self.x[f'mode_{mode}'] += 1
        cur = self._cell(nxt)
        if new_level:
            self.x['level_ups'] += 1
        elif valid:
            self.path.append(tuple(norm_action(*action)))
            self.traj.append(cur)
            if self.archive.visit(cur, tuple(self.path), tuple(self.traj)):
                self.last_new_step = self.steps
                if self.episode is not None:
                    self.episode['new'] += 1
                if self.restore_target in self.archive.cells and self.steps <= self.restore_credit_until:
                    self.archive.cells[self.restore_target].gains += 1
        elif aid == 0 and cur == self.level_start_cell:
            self.archive.visit(cur, (), ())
        elif aid == 0:
            self.x['reset_mismatch'] += 1
            self.restore_ok = False
            self.queue.clear()
        if mode == 'restore' and self.queue:
            entry = self.queue.popleft()
            if entry['expect'] is not None and cur != entry['expect']:
                self.queue.clear()
                self.x['restore_fail'] += 1
                if self.restore_target in self.archive.cells:
                    self.archive.cells[self.restore_target].fails += 1
            elif not self.queue:
                self.x['restore_done'] += 1
                self.restore_credit_until = self.steps + self.p['stall']
        if state == 'GAME_OVER' or new_level:
            self.queue.clear()
            self.macro_state = None
            self.nav = None
        if mode == 'macro' and self.macro_state is not None and (prev is not None):
            ms = self.macro_state
            ms['steps'] += 1
            stop, why, n = macro_should_stop(prev, nxt, self.mask, valid=valid, first_change=ms['first'], remaining=ms['remaining'])
            if ms['first'] is None:
                ms['first'] = n
            if stop:
                self.x[f'macro_stop_{why}'] += 1
                self.x['macro_steps'] += ms['steps']
                self.macro_state = None
        if self.avatar is not None and prev is not None and (aid in MOVE_KEYS) and (state == 'NOT_FINISHED') and (not new_level):
            self.avatar.update(self.objects.get(prev), nxt, aid, self.mask)
        if mode == 'nav' and self.nav is not None:
            self.nav['steps'] += 1
        if self.episode is not None:
            self.episode['steps'] += 1
            if mode not in ('restore', 'nav'):
                self.episode['free'] += 1
            if new_level:
                self.episode['levels'] += 1
                self._close_episode()

    def metrics(self) -> Dict:
        result = super().metrics()
        x = self.x
        result.update(macro=self.use_macro, restore=self.use_restore, objnav=self.use_objnav, bandit=self.use_bandit, tree=self.use_tree, automask=self.automask, automask_params=self.p, cells=len(self.archive), automask_counters=dict(sorted(x.items())), mean_macro_len=x['macro_steps'] / max(1, x['macro_started']), mean_restore_len=x['restore_len'] / max(1, x['restore_started']))
        if self.use_tree:
            n = max(1, self.stats['plans'])
            result.update(mean_tree_expanded=self.stats['expanded'] / n, mean_tree_deepest=x['tree_deepest_sum'] / n, tree_deep_best_rate=x['tree_deep_best'] / max(1, self.stats['groups']))
        if self.avatar is not None:
            result.update(avatar_ready=self.avatar.ready, avatar_votes=sum(self.avatar.votes.values()), avatar_moves={str(k): list(v) for k, v in self.avatar.moves().items()}, avatar_blocked=len(self.avatar.blocked))
        if self.automask:
            n = max(1, x['automask_checked'])
            result.update(automask_quorum=self._quorum(), memory_remasks=self.memory.remasks, automask_rate=x['automask_steps'] / n, mean_masked_px=x['automask_px'] / max(1, x['automask_steps']), masked_px_max=x['masked_px_max'])
        if self.use_bandit:
            result.update(bandit_pulls=dict(self.pulls), bandit_mean_reward={s: float(np.mean(v)) for s, v in self.rewards.items() if v})
        return result
