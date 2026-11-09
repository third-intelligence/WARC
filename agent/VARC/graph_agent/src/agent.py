from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from graph_agent.src.graph import Graph, choose
from graph_agent.src.novelty import Memory, rank
from graph_agent.src.perception import candidates, initial_mask, window_key
from graph_agent.src.window import PlanWindow, norm_action
ActionTriple = Tuple[int, int, int]
RESET_ACTION_ID = 0
ARMS = {'graph': dict(predictor=None, method=None), 'frozen_h1': dict(predictor='frozen', method='novelty'), 'native_h1': dict(predictor='ttt', method='novelty'), 'frozen_h2': dict(predictor='frozen', method='rollout'), 'native_h2': dict(predictor='ttt', method='rollout')}

class MpcAgent:

    def __init__(self, *, arm: str, game: str, ckpt: str='', device: str='cpu', seed: int=0, budget: int=4000, candidates_cap: int=32, branch: int=4, epsilon: float=0.2, update_steps: int=100, mask_mode: str='border_bar', hist_n: int=3, task_token_mode: str='mean_game', predictor=None) -> None:
        if arm not in ARMS:
            raise ValueError(f'unknown arm: {arm}')
        self.arm = arm
        self.game = game
        self.seed = seed
        self.rng = random.Random(seed)
        self.budget = int(budget)
        self.candidates_cap = int(candidates_cap)
        self.branch = int(branch)
        self.epsilon = float(epsilon)
        self.mask_mode = mask_mode
        self.hist_n = int(hist_n)
        self.steps = self.real_transitions = self.boundaries = 0
        self.window: Optional[PlanWindow] = None
        self.memory: Optional[Memory] = None
        self.mask: Optional[np.ndarray] = None
        self.pending = None
        self.graph = Graph()
        self.last_plan: Dict = {}
        self.last_eval: Dict = {}
        self.last_train: Dict = {}
        self.predicted_frame: Optional[np.ndarray] = None
        self.endpoint: Optional[np.ndarray] = None
        spec = ARMS[arm]
        self.method = spec['method']
        self.predictor = predictor
        if spec['predictor'] is not None and self.predictor is None:
            import torch
            from graph_agent.src.predictor import HistPredictor
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.predictor = HistPredictor(ckpt, game, spec['predictor'], device, seed, update_steps=update_steps, budget=budget, task_token_mode=task_token_mode)
            self.hist_n = self.predictor.hist_n
        self.rng = random.Random(seed)

    def reset_context(self, frame: np.ndarray, *, new_level: bool=False) -> None:
        self.window = PlanWindow(self.hist_n)
        self.window.reset(frame)
        if self.memory is None or new_level:
            self.mask = initial_mask(np.asarray(frame, np.uint8), self.mask_mode)
            self.memory = Memory(self.mask, 4, 5, 20)
            self.graph = Graph()
        self.memory.add(np.asarray(frame, np.uint8))

    def act(self, frame: np.ndarray, available: Sequence[int]) -> ActionTriple:
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
        scores = preds = endpoints = None
        plan: Dict = dict(method='graph', fallback=True)
        if self.predictor is not None and self.method is not None:
            plan_rng = random.Random(self.seed * 1000003 + self.steps + 8371)
            _probs, preds, scores, plan, endpoints = rank(self.method, self.predictor, self.window, pool, self.memory, plan_rng, self.branch)
        action = choose(pool, None if plan.get('fallback') else scores, self.rng, self.epsilon)
        index = pool.index(action)
        self.predicted_frame = None if preds is None else preds[index].copy()
        self.endpoint = None if endpoints is None else endpoints[index].copy()
        self.last_plan = dict(plan, mode=mode, candidate_count=len(allowed), pool_size=len(pool), selected=index, scores=None if scores is None else [float(s) for s in scores])
        self.pending = (action, key, expected)
        return action

    def observe(self, action: ActionTriple, next_frame: np.ndarray, *, next_available: Sequence[int], state: str, new_level: bool) -> None:
        self.steps += 1
        self.last_eval, self.last_train = ({}, {})
        nxt = np.asarray(next_frame, np.uint8)
        if self.pending is None:
            if int(action[0]) != RESET_ACTION_ID:
                raise RuntimeError('unmatched transition without a pending plan')
            self.reset_context(nxt, new_level=new_level)
            self.predicted_frame = self.endpoint = None
            return
        expected_action, key, _expected_target = self.pending
        if tuple(norm_action(*action)) != tuple(expected_action):
            raise RuntimeError('executed action differs from the selected action')
        self.pending = None
        valid = int(action[0]) != RESET_ACTION_ID and state == 'NOT_FINISHED' and (not new_level)
        following = self.window.clone()
        following.push(expected_action, nxt)
        next_key = window_key(following, self.mask, next_available, use_history=False)
        self.graph.record(key, expected_action, next_key, terminal=not valid)
        self.real_transitions += 1
        self.boundaries += int(not valid)
        if self.predicted_frame is not None:
            changed = self.window.frames[-1] != nxt
            correct = self.predicted_frame == nxt
            self.last_eval = dict(exact=int(bool(correct.all())), changed_cells=int(changed.sum()), changed_correct=int((correct & changed).sum()), pixel_acc=float(correct.mean()), valid=bool(valid))
            if valid:
                self.memory.observe_error(self.predicted_frame, nxt)
        self.memory.add(nxt)
        if self.predictor is not None:
            self.last_train = self.predictor.observe(self.window, expected_action, nxt, valid_transition=valid, update=valid and self.steps < self.budget) or {}
        if valid:
            self.window = following
        else:
            self.reset_context(nxt, new_level=new_level)

    def metrics(self) -> Dict:
        result = dict(arm=self.arm, env_steps=self.steps, real_transitions=self.real_transitions, boundaries=self.boundaries, graph_states=len(self.graph.nodes), graph_conflicts=self.graph.conflicts)
        if self.predictor is not None:
            p = self.predictor
            result.update(gradient_steps=p.gradients, prediction_seconds=p.prediction_seconds, update_seconds=p.update_seconds, scored_candidates=p.scored_candidates, task_id=p.task_id, schedule_passes=p.schedule_passes)
        return result
