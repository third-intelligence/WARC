"""The explorer of the Kaggle notebook (the graph explorer with WARC), behind an act / observe protocol.

The notebook's episode loop answers GAME_OVER with RESET without asking the agent and reports every executed
operation, the RESET included. This adapter only translates:
  - observation dict -> the agent's obs: the last frame of `frame`, zero-padded or cropped to 64 x 64, int64;
    available = `available_actions`
  - agent action (aid, x, y) -> {'kind', 'x', 'y'} (x, y only for ACTION6) and back
  - new_level = levels_completed changed between the previous and the new observation; a full reset is treated as a
    new level instead of an error
Agent settings: schedule budget 4000, branch 4, epsilon 0.2, mask border_bar, task token mean_game, the arm's
candidate / dedup / credit / topk / predict_batch / plan flags, run_budget = the operation cap. With
predictor='final' the world model is trained by the TTT recipe of final_ttt.FinalPredictor, injected where
UnifiedMpcAgent constructs its UnifiedPredictor.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import numpy as np

RESET_KIND, CLICK_KIND = 0, 6


def frame_of(observation: Dict[str, Any], image_size: int = 64) -> np.ndarray:
    grid = np.asarray(observation['frame'][-1])
    if grid.ndim == 3 and grid.shape[0] == 1:
        grid = grid[0]
    if grid.shape != (image_size, image_size):
        tmp = np.zeros((image_size, image_size), dtype=np.int64)
        h, w = min(grid.shape[0], image_size), min(grid.shape[1], image_size)
        tmp[:h, :w] = grid[:h, :w]
        return tmp
    return grid.astype(np.int64)


def triple_of(action: Dict[str, int]):
    kind = int(action['kind'])
    if kind == CLICK_KIND:
        return (kind, int(action['x']), int(action['y']))
    return (kind, -1, -1)


@contextlib.contextmanager
def injected_predictor(factory):
    """Make UnifiedMpcAgent build `factory(...)` instead of UnifiedPredictor while the agent is constructed."""
    import unified_wm.src.predictor as p33
    original = p33.UnifiedPredictor
    p33.UnifiedPredictor = factory
    try:
        yield
    finally:
        p33.UnifiedPredictor = original


class AgentPlayer:
    """act(observation) -> action dict; observe(previous, action, observation)."""

    def __init__(self, seed: int, *, game: str, ckpt: str, arm: str = 'automask_strict_h2', device: str = 'cuda',
                 predictor: str = 'final', run_budget: int = 1000, strict_split: bool = False,
                 params: Optional[Dict] = None, schedule_budget: int = 4000, update_steps: int = 100):
        from automask_agent.src.agent import ARM_CONFIGS, AutomaskAgent
        cfg = dict(ARM_CONFIGS[arm])
        kw = dict(arm=cfg['base_arm'], game=game, ckpt=ckpt, device=device, seed=seed, budget=schedule_budget,
                  candidates_cap=cfg['candidates'], branch=4, epsilon=0.2, update_steps=update_steps,
                  mask_mode='border_bar', task_token_mode='mean_game',
                  dedup=cfg['dedup'], credit=cfg['credit'], topk=cfg['topk'], predict_batch=cfg['predict_batch'],
                  macro=cfg['macro'], restore=cfg['restore'], objnav=cfg['objnav'], bandit=cfg['bandit'],
                  tree=cfg['tree'], automask=cfg['automask'], run_budget=run_budget, params=dict(params or {}))
        if predictor == 'final':
            from .final_ttt import FinalPredictor

            def factory(ckpt_, game_, arm_, device_, seed_, *, update_steps=100, budget=4000):
                return FinalPredictor(ckpt_, game_, arm_, device_, seed_, update_steps=update_steps,
                                     budget=budget, strict_split=strict_split)
            with injected_predictor(factory):
                self.agent = AutomaskAgent(**kw)
        else:
            raise ValueError(f'unknown predictor {predictor!r}')
        self.arm, self.predictor_kind = arm, predictor
        self.started = False

    # the notebook's per-game summary reads these
    @property
    def predictor(self):
        return self.agent.predictor

    @property
    def counters(self):
        return dict(self.agent.x) if hasattr(self.agent, 'x') else None

    def _start(self, observation):
        if not self.started:
            self.agent.reset_context(frame_of(observation))
            self.started = True

    def act(self, observation: Dict[str, Any]) -> Dict[str, int]:
        self._start(observation)
        available = [int(a) for a in observation['available_actions']]
        aid, x, y = self.agent.act(frame_of(observation), available)
        if int(aid) == CLICK_KIND:
            return {'kind': CLICK_KIND, 'x': int(x), 'y': int(y)}
        return {'kind': int(aid)}

    def observe(self, previous: Dict[str, Any], action: Dict[str, int], observation: Dict[str, Any]) -> None:
        self._start(previous)
        level_before = int(previous['levels_completed'])
        level_after = int(observation['levels_completed'])
        new_level = level_after != level_before or bool(observation.get('full_reset', False))
        self.agent.observe(triple_of(action), frame_of(observation),
                           next_available=[int(a) for a in observation['available_actions']],
                           state=observation['state'], new_level=new_level)

    def metrics(self) -> Dict[str, Any]:
        return self.agent.metrics()
