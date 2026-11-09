from __future__ import annotations
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from policy_wm.src.losses import ffp_loss
from hist_wm.eval.ttt_eval import _forward_dynamics
from hist_wm.src.hist_data import expand_window_items

def n_views(*, augment: bool=True, num_color_permute: int=9, aug_geom: str='no_flip') -> int:
    if not augment:
        return 1
    n_geo = 3 if aug_geom == 'no_flip' else 5
    return 1 + n_geo * (1 + num_color_permute)

def projected_passes(max_steps: int, *, cap: int=100, chunk_size: int=5, replay_ratio: float=2.0, minibatch: int=8, num_color_permute: int=9, aug_geom: str='no_flip') -> int:
    views = n_views(num_color_permute=num_color_permute, aug_geom=aug_geom)
    total = 0
    for seen in range(chunk_size, max_steps + 1, chunk_size):
        older = max(0, seen - chunk_size)
        replay_n = min(int(replay_ratio * chunk_size), older)
        raw = chunk_size + replay_n
        batches = math.ceil(raw * views / minibatch)
        total += math.ceil(cap / batches)
    return total

def lr_multiplier(step: int, total: int, warmup: int=10) -> float:
    if total <= 0 or not 0 <= warmup < total:
        raise ValueError('bad schedule horizon')
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return 0.5 * (1 + math.cos(math.pi * progress))

def build_optimizer(trainable, *, lr: float=0.0001, weight_decay: float=0.01, max_steps: int=4000, cap: int=100):
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    total = projected_passes(max_steps, cap=cap)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: lr_multiplier(k, total, min(10, total // 5)))
    return (optimizer, scheduler, total)

def capped_update(model, hist_n, optimizer, replay, *, chunk_size: int, replay_ratio: float, minibatch_size: int, task_id: int, cap: int, max_grad_norm: float=1.0, augment: bool=True, num_color_permute: int=9, geometric_transforms=None, device, rng, scheduler=None, trainable) -> Dict:
    if len(replay) < chunk_size:
        return {'dyn_loss': float('nan'), 'n_steps': 0, 'scheduler_steps': 0, 'lr': optimizer.param_groups[0]['lr']}
    new = replay.tail(chunk_size)
    n_rep = int(replay_ratio * chunk_size)
    rep = replay.sample(n_rep, exclude_tail=chunk_size, rng=rng)
    raw = new + rep
    data: List[tuple] = expand_window_items(raw, num_color_permute=num_color_permute, rng=rng, geometric_transforms=geometric_transforms) if augment else list(raw)
    model.train()
    sum_loss, n_steps, scheduler_steps = (0.0, 0, 0)
    while n_steps < cap:
        rng.shuffle(data)
        for i in range(0, len(data), minibatch_size):
            if n_steps >= cap:
                break
            mb = data[i:i + minibatch_size]
            optimizer.zero_grad(set_to_none=True)
            ffp_logits, nxt = _forward_dynamics(model, hist_n, mb, task_id, device)
            loss = ffp_loss(ffp_logits, nxt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            sum_loss += float(loss.detach())
            n_steps += 1
        if scheduler is not None:
            scheduler.step()
            scheduler_steps += 1
    model.eval()
    return {'dyn_loss': sum_loss / max(n_steps, 1), 'n_steps': n_steps, 'scheduler_steps': scheduler_steps, 'lr': optimizer.param_groups[0]['lr']}
