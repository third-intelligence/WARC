from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from hist_wm.eval.ttt_eval import WindowItemReplay
from graph_agent.src.agent import ARMS, MpcAgent
from graph_agent.src.predictor import HistPredictor
from graph_agent.src.ttt import build_optimizer
from unified_wm.src.model import build_from_unified_checkpoint

class UnifiedPredictor(HistPredictor):

    def __init__(self, ckpt: str, game: str, arm: str, device, seed: int, *, update_steps: int=100, budget: int=4000, batch_size: int=16) -> None:
        if arm not in {'frozen', 'ttt'}:
            raise ValueError(f'unknown predictor arm: {arm}')
        self.device = torch.device(device)
        self.arm = arm
        self.batch_size = int(batch_size)
        self.model, self.meta = build_from_unified_checkpoint(ckpt, device=self.device)
        split = self.meta.get('split') or {}
        if game in (split.get('train_games') or []):
            raise ValueError(f'train/test leakage: {game} is a pretraining game of this fold')
        if split.get('test_games') and game not in split['test_games']:
            raise ValueError(f'{game} is not a held-out game of this checkpoint')
        self.task_id = 0
        self.game = game
        self.hist_n = int(self.model.hist_n)
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
            self.trainable = list(self.model.parameters())
            self.optimizer, self.scheduler, self.schedule_passes = build_optimizer(self.trainable, max_steps=budget, cap=self.update_steps)

class UnifiedMpcAgent(MpcAgent):

    def __init__(self, *, arm: str, game: str, ckpt: str='', device: str='cpu', seed: int=0, budget: int=4000, update_steps: int=100, **kw) -> None:
        kw.pop('task_token_mode', None)
        predictor = None
        if ARMS[arm]['predictor'] is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            predictor = UnifiedPredictor(ckpt, game, ARMS[arm]['predictor'], device, seed, update_steps=update_steps, budget=budget)
        super().__init__(arm=arm, game=game, ckpt=ckpt, device=device, seed=seed, budget=budget, update_steps=update_steps, predictor=predictor, **kw)
        if predictor is not None:
            self.hist_n = predictor.hist_n
