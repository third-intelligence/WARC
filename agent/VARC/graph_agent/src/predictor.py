from __future__ import annotations
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from agent_base.src.augment import GEOMETRIC_TRANSFORMS_NO_FLIP
from wm_core.src.models import freeze_for_ttt
from hist_wm.eval.rollout_ttt_eval import load_model_auto
from hist_wm.eval.ttt_eval import WindowItemReplay, _reinject_mean_tokens
from hist_wm.src.hist_data import stack_batch_hist
from graph_agent.src.ttt import build_optimizer, capped_update
ActionTriple = Tuple[int, int, int]
_ZERO_TARGET = np.zeros((64, 64), np.uint8)

class HistPredictor:

    def __init__(self, ckpt: str, game: str, arm: str, device, seed: int, *, update_steps: int=100, budget: int=4000, batch_size: int=16, task_token_mode: str='mean_game') -> None:
        if arm not in {'frozen', 'ttt'}:
            raise ValueError(f'unknown predictor arm: {arm}')
        self.device = torch.device(device)
        self.arm = arm
        self.batch_size = int(batch_size)
        self.model, self.meta = load_model_auto(ckpt, device=self.device)
        split = self.meta.get('split') or {}
        if game in (split.get('train_games') or []):
            raise ValueError(f'train/test leakage: {game} is a pretraining game of this fold')
        if split.get('test_games') and game not in split['test_games']:
            raise ValueError(f'{game} is not a held-out game of this checkpoint')
        if task_token_mode == 'mean_game':
            _reinject_mean_tokens(self.model, split)
            self.task_id = int((split.get('task_lookup') or {}).get(game, 0))
        elif task_token_mode == 'policy_v1':
            self.task_id = 0
        else:
            raise ValueError(f'unknown task token mode: {task_token_mode}')
        self.game = game
        self.hist_n = int(getattr(self.model, 'hist_n', 1))
        self.model.eval()
        self.model.requires_grad_(False)
        self.replay = WindowItemReplay()
        self.rng = random.Random(seed + 9137)
        self.gradients = 0
        self.prediction_seconds = 0.0
        self.update_seconds = 0.0
        self.scored_candidates = 0
        self.model_calls = 0
        self.update_steps = int(update_steps)
        self.schedule_passes: Optional[int] = None
        self.optimizer = self.scheduler = self.trainable = None
        if arm == 'ttt':
            self.model.requires_grad_(True)
            self.trainable = freeze_for_ttt(self.model, 'policy')
            self.optimizer, self.scheduler, self.schedule_passes = build_optimizer(self.trainable, max_steps=budget, cap=self.update_steps)

    @torch.no_grad()
    def predict(self, window, actions: Sequence[ActionTriple]):
        self.model.eval()
        started = time.perf_counter()
        probabilities: List[np.ndarray] = []
        predictions: List[np.ndarray] = []
        for i in range(0, len(actions), self.batch_size):
            chunk = actions[i:i + self.batch_size]
            items = [window.window_item(a, _ZERO_TARGET, 0) for a in chunk]
            obs, aid, ax, ay, _nxt, ho, ha, hx, hy = stack_batch_hist(items, self.hist_n, self.device)
            tasks = torch.full((len(chunk),), self.task_id, dtype=torch.long, device=self.device)
            logits = self.model.forward_dynamics(obs, tasks, aid, ax, ay, hist_obs=ho, hist_action_ids=ha, hist_action_xs=hx, hist_action_ys=hy).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError('non-finite world-model prediction')
            probabilities.append((1 - logits.softmax(1).gather(1, obs[:, None]).squeeze(1)).cpu().numpy())
            predictions.append(logits.argmax(1).to(torch.uint8).cpu().numpy())
            self.model_calls += len(chunk)
        self.scored_candidates += len(actions)
        self.prediction_seconds += time.perf_counter() - started
        return (np.concatenate(probabilities), np.concatenate(predictions))

    def observe(self, window, action: ActionTriple, nxt: np.ndarray, *, valid_transition: bool=True, update: bool=True) -> Optional[Dict]:
        if self.arm == 'frozen' or not valid_transition:
            return None
        self.replay.add(window.window_item(action, nxt, 0))
        if len(self.replay) % 5 or not update:
            return None
        started = time.perf_counter()
        result = capped_update(self.model, self.hist_n, self.optimizer, self.replay, chunk_size=5, replay_ratio=2.0, minibatch_size=8, task_id=self.task_id, cap=self.update_steps, num_color_permute=9, geometric_transforms=GEOMETRIC_TRANSFORMS_NO_FLIP, device=self.device, rng=self.rng, scheduler=self.scheduler, trainable=self.trainable)
        if not np.isfinite(result['dyn_loss']):
            raise FloatingPointError('non-finite TTT loss')
        self.gradients += result['n_steps']
        self.update_seconds += time.perf_counter() - started
        return dict(result, gradient_steps=self.gradients, update_seconds=self.update_seconds, task_id=self.task_id, replay_size=len(self.replay))
