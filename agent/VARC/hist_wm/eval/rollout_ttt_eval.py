from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_WM_CORE_SRC = _PROJ_ROOT / 'wm_core' / 'src'
for _p in (str(_PROJ_ROOT), str(_WM_CORE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from agent_base.src.augment import GEOMETRIC_TRANSFORMS_NO_FLIP
from episode_data.src.data import find_recordings
from utils.lr_scheduler import get_cosine_schedule_with_warmup
from varc2.data import Step, parse_recording
from wm_core.src.models import POLICY_DEFAULTS, freeze_for_ttt
from hist_wm.src.hist_data import PAD_ID, history_depths, steps_to_window_items
from hist_wm.src.hist_model import ARCViTPolicyWMHist, build_from_hist_wm_checkpoint
from hist_wm.eval.ttt_eval import WindowItemReplay, _incremental_update_scheduled, _reinject_mean_tokens, evaluate_wm_per_level

def transplant_policy_state(hist_model: ARCViTPolicyWMHist, state: Dict) -> None:
    hs = hist_model.state_dict()
    used = set()
    for name, p in state.items():
        if name == 'patch_embed.proj.weight':
            hs['patch_embed3d.weight'].copy_(p.unsqueeze(2))
            used.add('patch_embed3d.weight')
        elif name == 'patch_embed.proj.bias':
            hs['patch_embed3d.bias'].copy_(p)
            used.add('patch_embed3d.bias')
        elif name == 'color_embed.weight':
            hs['color_embed.weight'][:16].copy_(p)
            used.add('color_embed.weight')
        else:
            hs[name].copy_(p)
            used.add(name)
    missing = set(hs) - used
    assert not missing, f'transplant left keys uninitialized: {sorted(missing)}'

def build_from_wm_core_policy_checkpoint(ckpt_path: str, *, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    mc = dict(config['model_config'])
    arch = mc.pop('arch', config.get('arch'))
    if arch != 'policy':
        raise ValueError(f'not an policy checkpoint (arch={arch!r})')
    num_tasks = mc.pop('num_tasks')
    patch_size = mc.pop('patch_size')
    model = ARCViTPolicyWMHist(num_tasks=num_tasks, patch_size=patch_size, hist_n=1, **{k: mc[k] for k in POLICY_DEFAULTS if k in mc})
    state = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt['model_state'].items()}
    transplant_policy_state(model, state)
    model = model.to(device)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != 'model_state'}
    meta['arch'] = arch
    return (model, meta)

def load_model_auto(ckpt_path: str, *, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch = ckpt['config']['model_config'].get('arch')
    del ckpt
    if arch == 'policy':
        return build_from_wm_core_policy_checkpoint(ckpt_path, device=device)
    if arch == 'policy_histtok':
        from histtok.src.tok_model import build_from_histtok_checkpoint
        return build_from_histtok_checkpoint(ckpt_path, device=device)
    return build_from_hist_wm_checkpoint(ckpt_path, device=device)

def build_rollout_starts(steps: List[Step], hist_n: int, horizon: int) -> List[tuple]:
    if horizon < 1:
        raise ValueError(f'horizon must be >= 1, got {horizon}')
    depths = history_depths(steps, hist_n)
    adj = [bool(np.array_equal(steps[j].obs_t1, steps[j + 1].obs_t)) for j in range(len(steps) - 1)]
    starts: List[tuple] = []
    for i in range(len(steps) - horizon + 1):
        if not all(adj[i:i + horizon - 1]):
            continue
        m = depths[i]
        frames = [steps[i - k].obs_t for k in range(m - 1, -1, -1)]
        inter = [(steps[i - k].action_id, steps[i - k].action_x, steps[i - k].action_y) for k in range(m - 1, 0, -1)]
        actions = [(steps[i + j].action_id, steps[i + j].action_x, steps[i + j].action_y) for j in range(horizon)]
        targets = [steps[i + j].obs_t1 for j in range(horizon)]
        starts.append((frames, inter, actions, targets, int(steps[i].levels)))
    return starts

def load_game_stream_with_rollout(game: str, data_root: Path, *, eval_holdout: int, max_train_steps: int, eval_cap: int, rollout_cap: int, seed: int, hist_n: int, horizon: int) -> Tuple[List[tuple], List[tuple], List[int], List[tuple], Dict]:
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
    rollout_starts: List[tuple] = []
    for r in eval_recs:
        rollout_starts.extend(build_rollout_starts(steps_by_rec[r], hist_n, horizon))
    n_starts_all = len(rollout_starts)
    if rollout_cap > 0 and len(rollout_starts) > rollout_cap:
        idx = sorted(rng.sample(range(len(rollout_starts)), rollout_cap))
        rollout_starts = [rollout_starts[i] for i in idx]
    info = {'game': game, 'n_recordings': len(recs), 'n_train_recordings': len(train_recs), 'n_eval_recordings': len(eval_recs), 'n_train_transitions': len(train_stream), 'n_eval_transitions': len(eval_all), 'n_rollout_starts': len(rollout_starts), 'n_rollout_starts_all': n_starts_all, 'eval_recordings': [r.name for r in eval_recs], 'game_max_level': int(gmax)}
    return (train_stream, eval_all, lvl_all, rollout_starts, info)

@torch.no_grad()
def evaluate_rollout(model, hist_n: int, starts: List[tuple], task_id: int, device, *, horizon: int, batch_size: int=64, return_frames: bool=False) -> Dict:
    if not starts:
        res = {'exact_acc_by_h': [float('nan')] * horizon, 'n': 0, 'per_level': {}}
        if return_frames:
            res['frames'] = np.zeros((0, horizon, 64, 64), dtype=np.uint8)
        return res
    model.eval()
    frames_chunks: List[np.ndarray] = []
    k = max(hist_n - 1, 0)
    n_ok = np.zeros(horizon, dtype=np.int64)
    lv_n: Dict[int, int] = {}
    lv_ok: Dict[int, np.ndarray] = {}
    for i0 in range(0, len(starts), batch_size):
        chunk = starts[i0:i0 + batch_size]
        B = len(chunk)
        win_obs = torch.full((B, hist_n, 64, 64), PAD_ID, dtype=torch.long, device=device)
        win_aid = torch.zeros((B, k), dtype=torch.long, device=device)
        win_ax = torch.full((B, k), -1, dtype=torch.long, device=device)
        win_ay = torch.full((B, k), -1, dtype=torch.long, device=device)
        aid = torch.zeros((B, horizon), dtype=torch.long, device=device)
        ax = torch.zeros((B, horizon), dtype=torch.long, device=device)
        ay = torch.zeros((B, horizon), dtype=torch.long, device=device)
        tgt = torch.zeros((B, horizon, 64, 64), dtype=torch.long, device=device)
        levels = []
        for b, (frames, inter, actions, targets, lvl) in enumerate(chunk):
            fs = frames[-hist_n:]
            win_obs[b, hist_n - len(fs):] = torch.from_numpy(np.stack(fs).astype(np.int64)).to(device)
            it = inter[-k:] if k > 0 else []
            for j, (a, x, y) in enumerate(it):
                slot = k - len(it) + j
                win_aid[b, slot], win_ax[b, slot], win_ay[b, slot] = (a, x, y)
            for j, (a, x, y) in enumerate(actions):
                aid[b, j], ax[b, j], ay[b, j] = (a, x, y)
            tgt[b] = torch.from_numpy(np.stack(targets).astype(np.int64)).to(device)
            levels.append(lvl)
        cum = torch.ones(B, dtype=torch.bool, device=device)
        cum_by_h = torch.zeros((B, horizon), dtype=torch.bool, device=device)
        preds_h: List[torch.Tensor] = []
        for h in range(horizon):
            obs = win_obs[:, -1]
            logits = model.forward_dynamics(obs, torch.full((B,), task_id, dtype=torch.long, device=device), aid[:, h], ax[:, h], ay[:, h], hist_obs=win_obs, hist_action_ids=win_aid, hist_action_xs=win_ax, hist_action_ys=win_ay)
            pred = logits.float().argmax(dim=1)
            if return_frames:
                preds_h.append(pred.to(torch.uint8))
            cum = cum & (pred == tgt[:, h]).flatten(1).all(dim=1)
            cum_by_h[:, h] = cum
            if hist_n > 1:
                win_obs = torch.cat([win_obs[:, 1:], pred.unsqueeze(1)], dim=1)
                if k > 0:
                    win_aid = torch.cat([win_aid[:, 1:], aid[:, h:h + 1]], dim=1)
                    win_ax = torch.cat([win_ax[:, 1:], ax[:, h:h + 1]], dim=1)
                    win_ay = torch.cat([win_ay[:, 1:], ay[:, h:h + 1]], dim=1)
            else:
                win_obs = pred.unsqueeze(1)
        if return_frames:
            frames_chunks.append(torch.stack(preds_h, dim=1).cpu().numpy())
        ok = cum_by_h.cpu().numpy()
        n_ok += ok.sum(axis=0)
        for b, lvl in enumerate(levels):
            lv_n[lvl] = lv_n.get(lvl, 0) + 1
            lv_ok[lvl] = lv_ok.get(lvl, np.zeros(horizon, dtype=np.int64)) + ok[b]
    N = len(starts)
    res = {'exact_acc_by_h': (n_ok / N).tolist(), 'n': N, 'per_level': {str(lvl): {'exact_acc_by_h': (lv_ok[lvl] / lv_n[lvl]).tolist(), 'n': lv_n[lvl]} for lvl in sorted(lv_n)}}
    if return_frames:
        res['frames'] = np.concatenate(frames_chunks, axis=0)
    return res

def run_game(args, game: str, device) -> Dict:
    model, meta = load_model_auto(args.ckpt, device=device)
    split_rec = meta.get('split') or {}
    _reinject_mean_tokens(model, split_rec)
    lookup = split_rec.get('task_lookup') or {}
    task_id = int(lookup.get(game, 0)) if args.task_id_mode == 'game' else 0
    hist_n = model.hist_n
    trainable = freeze_for_ttt(model, 'policy')
    optimizer = torch.optim.AdamW(trainable, lr=args.ttt_lr)
    rng = random.Random(args.seed)
    stream, eval_set, eval_levels, rollout_starts, info = load_game_stream_with_rollout(game, Path(args.data_root) / 'data', eval_holdout=args.eval_holdout, max_train_steps=args.max_steps, eval_cap=args.eval_cap, rollout_cap=args.rollout_cap, seed=args.seed, hist_n=hist_n, horizon=args.horizon)
    print(f'[hist_wm.rollout_ttt] game={game} task_id={task_id} hist_n={hist_n} H={args.horizon} train_transitions={info['n_train_transitions']} eval_transitions={info['n_eval_transitions']} rollout_starts={info['n_rollout_starts']}/{info['n_rollout_starts_all']} (eval recs={info['eval_recordings']})', flush=True)
    n_rounds = max(0, len(stream) // max(1, args.chunk_size))
    total_epochs = n_rounds * args.epochs_per_round
    scheduler = None
    if args.lr_scheduler == 'cosine' and total_epochs > 0:
        warmup = min(total_epochs // 5, 10)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_epochs)
    geo = GEOMETRIC_TRANSFORMS_NO_FLIP if args.aug_geom == 'no_flip' else None
    replay = WindowItemReplay()
    curve: List[Dict] = []

    def eval_point(env_step: int, st: Dict | None) -> None:
        m1 = evaluate_wm_per_level(model, hist_n, eval_set, eval_levels, task_id, device, batch_size=args.eval_batch_size)
        ro = evaluate_rollout(model, hist_n, rollout_starts, task_id, device, horizon=args.horizon, batch_size=args.eval_batch_size)
        curve.append({'env_step': env_step, 'dyn_loss': None if st is None else st['dyn_loss'], 'lr': optimizer.param_groups[0]['lr'], **m1, 'rollout': ro})
        print(f'[hist_wm.rollout_ttt] {game} env_step={env_step} 1step={m1['exact_acc']:.3f} rollout h1={ro['exact_acc_by_h'][0]:.3f} h5={ro['exact_acc_by_h'][min(4, args.horizon - 1)]:.3f} h{args.horizon}={ro['exact_acc_by_h'][-1]:.3f}', flush=True)
    eval_point(0, None)
    chunk_idx = 0
    for step, item in enumerate(stream, 1):
        replay.add(item)
        if step % args.chunk_size == 0 and len(replay) >= args.chunk_size:
            chunk_idx += 1
            st = _incremental_update_scheduled(model, hist_n, optimizer, replay, chunk_size=args.chunk_size, replay_ratio=args.replay_ratio, epochs_per_round=args.epochs_per_round, minibatch_size=args.ttt_minibatch, task_id=task_id, augment=not args.no_augment, num_color_permute=args.num_color_permute, geometric_transforms=geo, device=device, rng=rng, scheduler=scheduler, trainable=trainable)
            if chunk_idx % max(1, args.eval_every) == 0:
                eval_point(step, st)
    if curve and curve[-1]['env_step'] != len(stream):
        eval_point(len(stream), None)
    ckpt_tag = Path(args.ckpt).parent.name or 'ckpt'
    tag = f'_{args.out_tag}' if args.out_tag else ''
    out_dir = Path(args.out_dir) if args.out_dir else _PROJ_ROOT / 'hist_wm' / 'outputs' / 'rollout_ttt' / ckpt_tag / f'{game}_s{args.seed}{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'curve.json').open('w') as fh:
        json.dump(curve, fh, indent=2)
    zero, final = (curve[0], curve[-1])
    summary = {'game': game, 'arch': meta.get('arch'), 'ckpt': args.ckpt, 'hist_n': hist_n, 'horizon': args.horizon, 'rollout_acc_def': 'cumulative', 'rollout_history_fill': 'self', 'task_id_mode': args.task_id_mode, 'task_id': task_id, 'seed': args.seed, 'ttt_lr': args.ttt_lr, 'lr_scheduler': args.lr_scheduler, 'epochs_per_round': args.epochs_per_round, 'chunk_size': args.chunk_size, 'replay_ratio': args.replay_ratio, 'aug_geom': args.aug_geom, 'num_color_permute': args.num_color_permute, 'max_steps': args.max_steps, 'eval_every': args.eval_every, 'rollout_cap': args.rollout_cap, **{f'info_{k}': v for k, v in info.items()}, 'zero_shot_1step_exact': zero['exact_acc'], 'final_1step_exact': final['exact_acc'], 'zero_shot_rollout': zero['rollout']['exact_acc_by_h'], 'final_rollout': final['rollout']['exact_acc_by_h'], 'n_rollout_starts': zero['rollout']['n'], 'n_curve_points': len(curve)}
    with (out_dir / 'summary.json').open('w') as fh:
        json.dump(summary, fh, indent=2)
    if args.save_adapted:
        torch.save({'model_state_dict': model.state_dict(), 'env_steps': len(stream), 'game': game, 'seed': args.seed, 'src_ckpt': str(args.ckpt)}, out_dir / 'adapted_model.pt')
    print(f'[hist_wm.rollout_ttt] DONE {game}: 1step {zero['exact_acc']:.3f}->{final['exact_acc']:.3f} | rollout h{args.horizon} {zero['rollout']['exact_acc_by_h'][-1]:.3f}->{final['rollout']['exact_acc_by_h'][-1]:.3f} -> {out_dir}', flush=True)
    return summary

def parse_args():
    p = argparse.ArgumentParser(description='hist_wm TTT + autoregressive rollout eval (single GPU)')
    p.add_argument('--ckpt', required=True, help='hist_wm (policy_hist) or wm_core (policy) checkpoint (.pt)')
    p.add_argument('--games', default='', help='comma list; empty = all test games of the checkpoint fold')
    p.add_argument('--data-root', default='raw_data/ARC-AGI-3')
    p.add_argument('--task-id-mode', default='game', choices=['game', 'zero'])
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--rollout-cap', type=int, default=1024, help='cap on common start points (deterministic subsampling above it)')
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
    p.add_argument('--eval-every', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='')
    p.add_argument('--out-tag', default='')
    p.add_argument('--save-adapted', action='store_true', help='also save the adapted weights after TTT (default off)')
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
    print(f'[hist_wm.rollout_ttt] ckpt={args.ckpt} games={games} H={args.horizon} device={device}', flush=True)
    summaries = [run_game(args, g, device) for g in games]
    print('[hist_wm.rollout_ttt] ALL DONE: ' + ' '.join((f'{s['game']}:h{args.horizon} final={s['final_rollout'][-1]:.3f}' for s in summaries)), flush=True)
if __name__ == '__main__':
    main()
