from __future__ import annotations
import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_WM_CORE_SRC = _PROJ_ROOT / 'wm_core' / 'src'
for _p in (str(_PROJ_ROOT), str(_WM_CORE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
import torch.nn.functional as F
from agent_base.src.augment import GEOMETRIC_TRANSFORMS_NO_FLIP
from policy_wm.src.losses import ffp_loss
from episode_data.src.data import find_recordings
from utils.lr_scheduler import get_cosine_schedule_with_warmup
from varc2.data import parse_recording
from wm_core.src.models import freeze_for_ttt
from hist_wm.src.hist_data import expand_window_items, stack_batch_hist, steps_to_window_items
from hist_wm.src.hist_model import build_from_hist_wm_checkpoint

class WindowItemReplay:

    def __init__(self, capacity: int=50000):
        self.buf: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def add(self, item: tuple) -> None:
        self.buf.append(item)

    def tail(self, n: int) -> List[tuple]:
        n = min(n, len(self.buf))
        return list(self.buf)[-n:]

    def sample(self, n: int, exclude_tail: int=0, rng: Optional[random.Random]=None) -> List[tuple]:
        rng = rng or random
        pool_size = max(0, len(self.buf) - exclude_tail)
        if pool_size <= 0 or n <= 0:
            return []
        pool = list(self.buf)[:pool_size]
        return rng.sample(pool, min(n, pool_size))

def load_game_stream_with_levels(game: str, data_root: Path, *, eval_holdout: int, max_train_steps: int, eval_cap: int, seed: int, hist_n: int) -> Tuple[List[tuple], List[tuple], List[int], Dict]:
    rng = random.Random(seed)
    recs = find_recordings(game, data_root)
    if not recs:
        raise FileNotFoundError(f'no recordings for game={game!r} under {data_root}')
    steps_by_rec = {r: parse_recording(r) for r in recs}
    parsed = [r for r in recs if steps_by_rec[r]]
    if not parsed:
        raise ValueError(f'no valid steps in any recording for game={game!r}')
    reached = {r: max((s.levels + (1 if s.completed else 0) for s in steps_by_rec[r])) for r in parsed}
    gmax = max(reached.values())
    reaching = [r for r in parsed if reached[r] == gmax]
    if eval_holdout > 0:
        n_hold = min(eval_holdout, len(reaching))
        eval_recs = reaching[-n_hold:]
        train_recs = [r for r in parsed if r not in set(eval_recs)]
        if not train_recs:
            train_recs = list(parsed)
    else:
        eval_recs = reaching[-1:]
        train_recs = list(parsed)
    train_stream: List[tuple] = []
    for r in train_recs:
        train_stream.extend(steps_to_window_items(steps_by_rec[r], hist_n))
        if max_train_steps and len(train_stream) >= max_train_steps:
            break
    if max_train_steps:
        train_stream = train_stream[:max_train_steps]
    eval_all: List[tuple] = []
    for r in eval_recs:
        eval_all.extend(steps_to_window_items(steps_by_rec[r], hist_n))
    lvl_all = [it[4] for it in eval_all]
    if eval_cap > 0 and len(eval_all) > eval_cap:
        idx = sorted(rng.sample(range(len(eval_all)), eval_cap))
        eval_all = [eval_all[i] for i in idx]
        lvl_all = [lvl_all[i] for i in idx]
    info = {'game': game, 'n_recordings': len(recs), 'n_train_recordings': len(train_recs), 'n_eval_recordings': len(eval_recs), 'n_train_transitions': len(train_stream), 'n_eval_transitions': len(eval_all), 'n_train_full_hist': sum((1 for it in train_stream if len(it[0]) == hist_n)), 'eval_recordings': [r.name for r in eval_recs], 'game_max_level': int(gmax), 'reached_levels': {r.name: int(reached[r]) for r in parsed}}
    return (train_stream, eval_all, lvl_all, info)

def _forward_dynamics(model, hist_n, mb, task_id, device):
    obs, aid, x, y, nxt, h_obs, h_aid, h_ax, h_ay = stack_batch_hist(mb, hist_n, device)
    task_ids = torch.full((obs.size(0),), task_id, dtype=torch.long, device=device)
    logits = model.forward_dynamics(obs, task_ids, aid, x, y, hist_obs=h_obs, hist_action_ids=h_aid, hist_action_xs=h_ax, hist_action_ys=h_ay)
    return (logits, nxt)

@torch.no_grad()
def evaluate_wm_per_level(model, hist_n, eval_set: List[tuple], eval_levels: List[int], task_id: int, device, *, batch_size: int=64) -> Dict:
    if not eval_set:
        return {'ce': float('nan'), 'exact_acc': float('nan'), 'pixel_acc': float('nan'), 'n': 0, 'per_level': {}}
    model.eval()
    lv_stats: Dict[int, list] = {}
    for i in range(0, len(eval_set), batch_size):
        mb = eval_set[i:i + batch_size]
        logits, nxt = _forward_dynamics(model, hist_n, mb, task_id, device)
        logits = logits.float()
        B = logits.size(0)
        num_colors = logits.size(1)
        ce = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, num_colors), nxt.reshape(-1), reduction='none').view(B, -1).mean(dim=1)
        pred = logits.argmax(dim=1)
        exact = (pred == nxt).flatten(1).all(dim=1)
        pixel = (pred == nxt).flatten(1).double().mean(dim=1)
        ce_l, ex_l, px_l = (ce.cpu().tolist(), exact.cpu().tolist(), pixel.cpu().tolist())
        for j in range(B):
            st = lv_stats.setdefault(eval_levels[i + j], [0, 0.0, 0, 0.0])
            st[0] += 1
            st[1] += ce_l[j]
            st[2] += int(ex_l[j])
            st[3] += px_l[j]
    n = sum((st[0] for st in lv_stats.values()))
    return {'ce': sum((st[1] for st in lv_stats.values())) / n, 'exact_acc': sum((st[2] for st in lv_stats.values())) / n, 'pixel_acc': sum((st[3] for st in lv_stats.values())) / n, 'n': n, 'per_level': {str(lvl): {'ce': st[1] / st[0], 'exact_acc': st[2] / st[0], 'n': st[0]} for lvl, st in sorted(lv_stats.items())}}

def _incremental_update_scheduled(model, hist_n, optimizer, replay: WindowItemReplay, *, chunk_size, replay_ratio, epochs_per_round, minibatch_size, task_id, max_grad_norm=1.0, augment=True, num_color_permute=9, geometric_transforms=None, device, rng, scheduler=None, trainable) -> dict:
    if len(replay) < chunk_size:
        return {'dyn_loss': float('nan'), 'n_steps': 0, 'lr': optimizer.param_groups[0]['lr']}
    new = replay.tail(chunk_size)
    n_rep = int(replay_ratio * chunk_size)
    rep = replay.sample(n_rep, exclude_tail=chunk_size, rng=rng)
    raw = new + rep
    data = expand_window_items(raw, num_color_permute=num_color_permute, rng=rng, geometric_transforms=geometric_transforms) if augment else list(raw)
    model.train()
    sum_loss, n_steps = (0.0, 0)
    for _ in range(epochs_per_round):
        rng.shuffle(data)
        for i in range(0, len(data), minibatch_size):
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
    return {'dyn_loss': sum_loss / max(n_steps, 1), 'n_steps': n_steps, 'lr': optimizer.param_groups[0]['lr']}

@torch.no_grad()
def _reinject_mean_tokens(model, split_rec: Dict) -> None:
    lookup = split_rec.get('task_lookup') or {}
    train_games = split_rec.get('train_games') or []
    test_games = split_rec.get('test_games') or []
    if not (lookup and train_games and test_games):
        return
    w = model.task_token_embed.weight
    train_ids = torch.tensor([lookup[g] for g in train_games], device=w.device)
    test_ids = torch.tensor([lookup[g] for g in test_games], device=w.device)
    w[test_ids] = w[train_ids].mean(dim=0, keepdim=True)

def run_game(args, game: str, device) -> Dict:
    model, meta = build_from_hist_wm_checkpoint(args.ckpt, device=device)
    split_rec = meta.get('split') or {}
    _reinject_mean_tokens(model, split_rec)
    lookup = split_rec.get('task_lookup') or {}
    task_id = int(lookup.get(game, 0)) if args.task_id_mode == 'game' else 0
    cfg_meta = meta.get('config') or {}
    hist_n = args.hist_n if args.hist_n > 0 else int(cfg_meta.get('hist_n', 2) or 2)
    assert hist_n == model.hist_n, f'hist_n mismatch: cli/ckpt {hist_n} vs model {model.hist_n}'
    trainable = freeze_for_ttt(model, 'policy')
    optimizer = torch.optim.AdamW(trainable, lr=args.ttt_lr)
    rng = random.Random(args.seed)
    stream, eval_set, eval_levels, info = load_game_stream_with_levels(game, Path(args.data_root) / 'data', eval_holdout=args.eval_holdout, max_train_steps=args.max_steps, eval_cap=args.eval_cap, seed=args.seed, hist_n=hist_n)
    print(f'[hist_wm.ttt] game={game} task_id={task_id} hist_n={hist_n} train_transitions={info['n_train_transitions']} (full-hist {info['n_train_full_hist']}) eval_transitions={info['n_eval_transitions']} (eval recs={info['eval_recordings']})', flush=True)
    n_rounds = max(0, len(stream) // max(1, args.chunk_size))
    total_epochs = n_rounds * args.epochs_per_round
    scheduler = None
    if args.lr_scheduler == 'cosine' and total_epochs > 0:
        warmup = min(total_epochs // 5, 10)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_epochs)
        print(f'[hist_wm.ttt] LR=cosine(peak={args.ttt_lr}, warmup={warmup}, total_epochs={total_epochs})', flush=True)
    geo = GEOMETRIC_TRANSFORMS_NO_FLIP if args.aug_geom == 'no_flip' else None
    replay = WindowItemReplay()
    curve: List[Dict] = []
    z = evaluate_wm_per_level(model, hist_n, eval_set, eval_levels, task_id, device, batch_size=args.eval_batch_size)
    curve.append({'env_step': 0, 'dyn_loss': None, 'lr': optimizer.param_groups[0]['lr'], **z})
    chunk_idx = 0
    for step, item in enumerate(stream, 1):
        replay.add(item)
        if step % args.chunk_size == 0 and len(replay) >= args.chunk_size:
            chunk_idx += 1
            st = _incremental_update_scheduled(model, hist_n, optimizer, replay, chunk_size=args.chunk_size, replay_ratio=args.replay_ratio, epochs_per_round=args.epochs_per_round, minibatch_size=args.ttt_minibatch, task_id=task_id, augment=not args.no_augment, num_color_permute=args.num_color_permute, geometric_transforms=geo, device=device, rng=rng, scheduler=scheduler, trainable=trainable)
            if chunk_idx % max(1, args.eval_every) == 0:
                m = evaluate_wm_per_level(model, hist_n, eval_set, eval_levels, task_id, device, batch_size=args.eval_batch_size)
                curve.append({'env_step': step, 'dyn_loss': st['dyn_loss'], 'lr': st['lr'], **m})
    if curve and curve[-1]['env_step'] != len(stream):
        m = evaluate_wm_per_level(model, hist_n, eval_set, eval_levels, task_id, device, batch_size=args.eval_batch_size)
        curve.append({'env_step': len(stream), 'dyn_loss': None, 'lr': optimizer.param_groups[0]['lr'], **m})
    ckpt_tag = Path(args.ckpt).parent.name or 'ckpt'
    tag = f'_{args.out_tag}' if args.out_tag else ''
    out_dir = Path(args.out_dir) if args.out_dir else _PROJ_ROOT / 'hist_wm' / 'outputs' / 'ttt' / ckpt_tag / f'{game}_s{args.seed}{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'curve.json').open('w') as fh:
        json.dump(curve, fh, indent=2)
    zero, final = (curve[0], curve[-1])

    def reach(thr):
        for c in curve:
            if c['exact_acc'] is not None and (not np.isnan(c['exact_acc'])) and (c['exact_acc'] >= thr):
                return c['env_step']
        return None
    summary = {'game': game, 'arch': meta.get('arch'), 'ckpt': args.ckpt, 'fold': cfg_meta.get('fold'), 'hist_n': hist_n, 'task_id_mode': args.task_id_mode, 'task_id': task_id, 'seed': args.seed, 'ttt_lr': args.ttt_lr, 'lr_scheduler': args.lr_scheduler, 'epochs_per_round': args.epochs_per_round, 'chunk_size': args.chunk_size, 'replay_ratio': args.replay_ratio, 'aug_geom': args.aug_geom, 'num_color_permute': args.num_color_permute, 'max_steps': args.max_steps, **{f'info_{k}': v for k, v in info.items()}, 'zero_shot': {k: zero[k] for k in ('ce', 'exact_acc', 'pixel_acc')}, 'final': {k: final[k] for k in ('ce', 'exact_acc', 'pixel_acc')}, 'zero_shot_per_level': zero['per_level'], 'final_per_level': final['per_level'], 'delta_exact': final['exact_acc'] - zero['exact_acc'], 'delta_ce': final['ce'] - zero['ce'], 'step_reach_exact_0.5': reach(0.5), 'step_reach_exact_0.9': reach(0.9), 'n_curve_points': len(curve)}
    with (out_dir / 'summary.json').open('w') as fh:
        json.dump(summary, fh, indent=2)
    if not args.no_save_adapted:
        torch.save({'model_state_dict': model.state_dict(), 'env_steps': len(stream), 'game': game, 'seed': args.seed, 'src_ckpt': str(args.ckpt)}, out_dir / 'adapted_model.pt')
    print(f'[hist_wm.ttt] DONE {game}: zero exact={zero['exact_acc']:.3f} ce={zero['ce']:.3f} -> final exact={final['exact_acc']:.3f} ce={final['ce']:.3f} (dExact={summary['delta_exact']:+.3f}) -> {out_dir}', flush=True)
    return summary

def parse_args():
    p = argparse.ArgumentParser(description='hist_wm incremental TTT eval (single GPU)')
    p.add_argument('--ckpt', required=True, help='hist_wm offline checkpoint (.pt)')
    p.add_argument('--games', default='', help='comma list; empty = all test games of the checkpoint fold')
    p.add_argument('--data-root', default='raw_data/ARC-AGI-3')
    p.add_argument('--task-id-mode', default='game', choices=['game', 'zero'])
    p.add_argument('--hist-n', type=int, default=0, help='0 (default) = inherit hist_n from the checkpoint config')
    p.add_argument('--chunk-size', type=int, default=5)
    p.add_argument('--replay-ratio', type=float, default=2.0)
    p.add_argument('--epochs-per-round', type=int, default=15)
    p.add_argument('--ttt-minibatch', type=int, default=8)
    p.add_argument('--ttt-lr', type=float, default=0.0001)
    p.add_argument('--lr-scheduler', default='cosine', choices=['cosine', 'none'])
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--num-color-permute', type=int, default=9)
    p.add_argument('--aug-geom', default='no_flip', choices=['full', 'no_flip'])
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--eval-holdout', type=int, default=1)
    p.add_argument('--eval-cap', type=int, default=1024)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='')
    p.add_argument('--out-tag', default='')
    p.add_argument('--no-save-adapted', action='store_true', help='do not save the adapted weights (adapted_model.pt) after TTT')
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.games:
        games = [g.strip() for g in args.games.split(',') if g.strip()]
    else:
        meta = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        games = list((meta.get('split') or {}).get('test_games') or [])
        del meta
        if not games:
            raise SystemExit('the checkpoint has no test_games; pass --games')
    print(f'[hist_wm.ttt] ckpt={args.ckpt} games={games} device={device}', flush=True)
    summaries = [run_game(args, g, device) for g in games]
    print('[hist_wm.ttt] ALL DONE: ' + ' '.join((f'{s['game']}:dExact={s['delta_exact']:+.3f}' for s in summaries)), flush=True)
if __name__ == '__main__':
    main()
