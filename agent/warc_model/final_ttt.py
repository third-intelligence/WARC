"""Test-time training recipe for the WARC world model, as an agent predictor.

After every valid transition: 20 gradient steps of minibatch 8 drawn from the newest 10 transitions + 20 older ones
(replay.sample), each expanded to its symmetry slots and then reduced to the rotation views (rotation_slot: the same
rotation without the colour permutation, RNG use unchanged); bf16 autocast for the training forward (loss in FP32),
gradient clip 1, all parameters trainable; Muon (momentum 0.95, nesterov, adjust_lr 'match_rms_adamw') on the 2-D
`.weight`s of `blocks.*` except ada_mod, AdamW on the rest, both lr 3e-4 and weight decay 0.01, constant learning
rate.

NaN guard: a step whose loss or gradient norm is non-finite is skipped (no optimizer step; it still counts toward
the 20); after each update every parameter is checked, and if any is non-finite the weights of the last finite state
are restored and TTT stays off for the rest of the game. An update whose 20 steps were all skipped also turns TTT
off. Without a non-finite value the arithmetic is unchanged.

Prediction is HistPredictor.predict under bf16 autocast on CUDA (logits cast to FP32 before softmax / argmax). A
non-finite bf16 prediction is redone in FP32 and the rest of the game predicts in FP32. WARC_PREDICT_BF16=0 predicts
in FP32 from the start.
"""
from __future__ import annotations

import math
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch

from hist_wm.eval import ttt_eval as base
from graph_agent.src.predictor import HistPredictor

from .model import build_from_checkpoint
from .varc2_optim import Muon

NEWEST, OLDER, STEPS, MINIBATCH, LR, WEIGHT_DECAY = 10, 20, 20, 8, 3e-4, 0.01
NUM_COLOR_PERMUTE = 9
VIEW_COUNT = 1 + 3 * (1 + NUM_COLOR_PERMUTE)   # 31 slots per transition (expand_window_items)
STACK = 'blocks'
RECIPE = 'muon_lr3e-4_const_b10-20_rot'
PREDICT_BF16 = os.environ.get('WARC_PREDICT_BF16', '1') != '0'


def rotation_slot(j):
    """Slot j of a transition's 31 -> the slot with the same rotation and no colour permutation."""
    return 0 if j == 0 else 1 + 10 * ((j - 1) // 10)


class _Optimizers:
    """Several optimizers stepped as one (bank_update calls zero_grad / step and reads param_groups[0]['lr'])."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()


def split_params(model):
    """(muon, adamw) parameter lists in named_parameters order: the 2-D weights of the transformer blocks except the
    adaLN modulation go to Muon, everything else to AdamW."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            is_muon = (p.ndim == 2 and name.endswith('.weight') and name.split('.', 1)[0] == STACK
                       and 'ada_mod' not in name)
            (muon if is_muon else adamw).append(p)
    return muon, adamw


def bank_update(p, *, newest=NEWEST, older=OLDER):
    """One update on the replay: the newest 10 transitions plus 20 older ones, each in its rotation views."""
    replay, rng = p.replay, p.rng
    chunk = min(newest, len(replay))
    raw = replay.tail(chunk) + replay.sample(int(older / newest * chunk), exclude_tail=chunk, rng=rng)
    data = base.expand_window_items(raw, num_color_permute=NUM_COLOR_PERMUTE, rng=rng,
                                    geometric_transforms=base.GEOMETRIC_TRANSFORMS_NO_FLIP)
    data = [data[(k // VIEW_COUNT) * VIEW_COUNT + rotation_slot(k % VIEW_COUNT)] for k in range(len(data))]
    p.model.train()
    total, steps, passes, skipped = 0., 0, 0, 0
    while steps < p.update_steps:
        rng.shuffle(data)
        for i in range(0, len(data), MINIBATCH):
            if steps >= p.update_steps:
                break
            mb = data[i:i + MINIBATCH]
            p.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(p.device.type, dtype=torch.bfloat16):
                logits, target = base._forward_dynamics(p.model, 3, mb, p.task_id, p.device)
            loss = base.ffp_loss(logits.float(), target)
            value = float(loss.detach())
            if not math.isfinite(value):
                p.optimizer.zero_grad(set_to_none=True)
                skipped += 1
                steps += 1
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(p.trainable, 1.)
            if not math.isfinite(float(norm)):
                p.optimizer.zero_grad(set_to_none=True)
                skipped += 1
                steps += 1
                continue
            p.optimizer.step()
            total += value
            steps += 1
        passes += 1
    done = steps - skipped
    return dict(dyn_loss=total / done if done else float('nan'), n_steps=steps, skipped_steps=skipped,
                scheduler_steps=passes, lr=p.optimizer.param_groups[0]['lr'], n_raw=len(raw), n_augmented=len(data))


class FinalPredictor(HistPredictor):
    """WARC world model + the TTT recipe, with UnifiedPredictor's constructor signature.

    `update_steps` and `budget` are accepted for the agent's call and ignored (20 steps after every valid
    transition, constant LR). `strict_split`: refuse a game the checkpoint was pretrained on or does not hold out
    (for fold models); off for the model pretrained on all public games."""

    def __init__(self, ckpt: str, game: str, arm: str, device, seed: int, *, update_steps: int = 100,
                 budget: int = 4000, batch_size: int = 16, strict_split: bool = True) -> None:
        if arm not in {"frozen", "ttt"}:
            raise ValueError(f"unknown predictor arm: {arm}")
        self.device = torch.device(device)
        self.arm = arm
        self.batch_size = int(batch_size)   # the agent sets 64 after construction
        self.model, self.meta = build_from_checkpoint(ckpt, device=self.device)
        split = self.meta.get("split") or {}
        stem = game.split("-", 1)[0]
        self.pretrained_on_game = stem in (split.get("train_games") or [])
        if strict_split:
            if self.pretrained_on_game:
                raise ValueError(f"train/test leakage: {stem} is a pretraining game of this checkpoint")
            if split.get("test_games") and stem not in split["test_games"]:
                raise ValueError(f"{stem} is not a held-out game of this checkpoint")
        self.task_id = 0
        self.game = game
        self.hist_n = int(self.model.hist_n)
        self.model.eval()
        self.model.requires_grad_(False)
        self.replay = base.WindowItemReplay()
        self.rng = random.Random(seed + 9137)
        self.gradients = 0
        self.prediction_seconds = 0.0
        self.update_seconds = 0.0
        self.scored_candidates = 0
        self.model_calls = 0
        self.update_steps = STEPS
        self.schedule_passes: Optional[int] = None
        self.optimizer = self.scheduler = self.trainable = None
        self.skipped_steps = self.nan_restores = self.all_skipped_updates = self.bf16_fallbacks = 0
        self.predict_fp32 = not PREDICT_BF16
        self.ttt_disabled = False
        self.recipe = dict(name=RECIPE, newest=NEWEST, older=OLDER, steps=STEPS, minibatch=MINIBATCH, lr=LR,
                           weight_decay=WEIGHT_DECAY, views='rot', schedule='const', precision='bf16_autocast',
                           predict_precision='bf16_autocast' if PREDICT_BF16 else 'fp32')
        if arm == "ttt":
            self.model.requires_grad_(True)
            self.trainable = list(self.model.parameters())
            muon, adamw = split_params(self.model)
            if not muon:
                raise ValueError('muon group is empty')
            self.optimizer = _Optimizers(
                [Muon(muon, lr=LR, weight_decay=WEIGHT_DECAY, momentum=.95, adjust_lr_fn='match_rms_adamw'),
                 torch.optim.AdamW(adamw, lr=LR, weight_decay=WEIGHT_DECAY)])
            self._finite_copy = [q.detach().clone() for q in self.trainable]
            self.recipe.update(muon_tensors=len(muon), adamw_tensors=len(adamw),
                               muon_parameters=sum(q.numel() for q in muon),
                               adamw_parameters=sum(q.numel() for q in adamw))

    def predict(self, window, actions):
        """HistPredictor.predict under bf16 autocast on CUDA; FP32 after a non-finite bf16 prediction (which is redone
        in FP32) or with WARC_PREDICT_BF16=0."""
        if self.predict_fp32 or self.device.type != 'cuda':
            return super().predict(window, actions)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return super().predict(window, actions)
        except FloatingPointError:
            self.predict_fp32 = True
            self.bf16_fallbacks += 1
            return super().predict(window, actions)

    def observe(self, window, action, nxt: np.ndarray, *, valid_transition: bool = True,
                update: bool = True) -> Optional[Dict]:
        """Add a valid transition to the replay and run one bank update after it (every valid transition), unless
        the NaN guard turned TTT off."""
        if self.arm == "frozen" or not valid_transition:
            return None
        self.replay.add(window.window_item(action, nxt, 0))
        if not update or self.ttt_disabled:
            return None
        started = time.perf_counter()
        result = bank_update(self)
        self.skipped_steps += result["skipped_steps"]
        if result["skipped_steps"] >= result["n_steps"]:   # every step non-finite: stop paying for TTT
            self.all_skipped_updates += 1
            self.ttt_disabled = True
        with torch.no_grad():
            finite = bool(torch.stack([torch.isfinite(q).all() for q in self.trainable]).all())   # one sync
            if finite:
                for keep, q in zip(self._finite_copy, self.trainable):
                    keep.copy_(q)
            else:
                for keep, q in zip(self._finite_copy, self.trainable):
                    q.copy_(keep)
                self.nan_restores += 1
                self.ttt_disabled = True
        self.model.eval()
        self.gradients += result["n_steps"] - result["skipped_steps"]
        self.update_seconds += time.perf_counter() - started
        return dict(result, gradient_steps=self.gradients, update_seconds=self.update_seconds,
                    task_id=self.task_id, replay_size=len(self.replay), skipped_total=self.skipped_steps,
                    nan_restores=self.nan_restores, ttt_disabled=self.ttt_disabled)
