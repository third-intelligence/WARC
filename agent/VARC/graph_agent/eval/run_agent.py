from __future__ import annotations
import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List
import numpy as np
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='graph explorer with a world model on one ARC-AGI-3 game')
    p.add_argument('--game', required=True)
    p.add_argument('--arm', default='native_h2', choices=['graph', 'frozen_h1', 'native_h1', 'frozen_h2', 'native_h2'])
    p.add_argument('--ckpt', default='', help='world-model checkpoint (not needed for arm=graph)')
    p.add_argument('--budget', type=int, default=1000, help='number of environment operations, RESET included')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='')
    p.add_argument('--operation-mode', default='offline')
    p.add_argument('--environments-dir', default='', help='directory with the game environment files (games/<game>/<version>/metadata.json)')
    p.add_argument('--candidates', type=int, default=32, help='candidate cap at the root')
    p.add_argument('--branch', type=int, default=4, help='children per candidate in the second step, itself included')
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--update-steps', type=int, default=100, help='gradient steps per update')
    p.add_argument('--schedule-budget', type=int, default=4000, help='horizon of the update schedule')
    p.add_argument('--task-token-mode', default='mean_game', choices=['mean_game', 'policy_v1'])
    p.add_argument('--mask-mode', default='border_bar', choices=['border_bar', 'none'])
    p.add_argument('--mode', default='full', choices=['full', 'l1'], help='l1: stop when level 1 is cleared')
    p.add_argument('--out-dir', default='')
    p.add_argument('--out-tag', default='')
    p.add_argument('--save-adapted', action='store_true')
    p.add_argument('--log-every', type=int, default=50)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    os.environ['ONLY_RESET_LEVELS'] = 'false'
    os.environ.setdefault('OPERATION_MODE', 'OFFLINE')
    from arc_env.src.env import Action, MiniArcEnv
    from graph_agent.src.agent import MpcAgent
    device = args.device
    if not device:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()
    env = MiniArcEnv(args.game, image_size=64, max_episode_steps=args.budget + 10, operation_mode=args.operation_mode, environments_dir=args.environments_dir)
    obs, info = env.reset()
    os.environ['ONLY_RESET_LEVELS'] = 'true'
    if int(info.get('levels_completed', 0)) != 0 or info.get('state') != 'NOT_FINISHED':
        raise RuntimeError('run must start at the unsolved first level')
    agent = MpcAgent(arm=args.arm, game=args.game, ckpt=args.ckpt, device=device, seed=args.seed, budget=args.schedule_budget, candidates_cap=args.candidates, branch=args.branch, epsilon=args.epsilon, update_steps=args.update_steps, mask_mode=args.mask_mode, task_token_mode=args.task_token_mode)
    agent.reset_context(obs)
    ckpt_tag = Path(args.ckpt).parent.name if args.ckpt else 'no_model'
    tag = f'_{args.out_tag}' if args.out_tag else ''
    out_dir = Path(args.out_dir) if args.out_dir else _PROJ_ROOT / 'graph_agent' / 'outputs' / args.arm / ckpt_tag / f'{args.game}_s{args.seed}{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[agent] game={args.game} arm={args.arm} device={device} budget={args.budget} ckpt={ckpt_tag} task_id={getattr(agent.predictor, 'task_id', None)} -> {out_dir}', flush=True)
    steps_log: List[Dict] = []
    action_counter: Counter = Counter()
    mode_counter: Counter = Counter()
    max_level = 0
    level_first_step: Dict[int, int] = {}
    win_levels = int(info.get('win_levels', 0))
    status = 'budget_exhaustion'
    resets = 0
    plan_time = 0.0
    for step in range(1, args.budget + 1):
        state = info.get('state', 'NOT_FINISHED')
        available = [int(a) for a in info.get('available_actions', [])]
        level_before = int(info.get('levels_completed', 0))
        if state == 'GAME_OVER':
            action = (0, -1, -1)
            mode_counter['reset'] += 1
        else:
            _t = time.time()
            action = agent.act(obs, available)
            plan_time += time.time() - _t
            mode_counter[agent.last_plan.get('mode', '?')] += 1
            if agent.last_plan.get('fallback'):
                mode_counter['fallback'] += 1
        aid, ax, ay = (int(action[0]), int(action[1]), int(action[2]))
        next_obs, info_next = env.step(Action(action_id=aid, x=ax, y=ay))
        action_counter[aid] += 1
        resets += int(aid == 0)
        level_after = int(info_next.get('levels_completed', 0))
        next_state = info_next.get('state', 'NOT_FINISHED')
        if bool(info_next.get('full_reset', False)) or level_after < level_before:
            raise RuntimeError('full reset / level regression violates the protocol')
        agent.observe(action, next_obs, next_available=[int(a) for a in info_next.get('available_actions', [])], state=next_state, new_level=level_after != level_before)
        row = dict(step=step, action=[aid, ax, ay], mode=mode_counter and agent.last_plan.get('mode'), levels_completed=level_after, level_up=int(level_after > level_before), state=next_state, fallback=bool(agent.last_plan.get('fallback')), calibrated=bool(agent.last_plan.get('calibrated', False)), error_radius=float(agent.last_plan.get('error_radius', 0.0)), memory_states=int(agent.last_plan.get('memory_states', 0)), candidate_count=int(agent.last_plan.get('candidate_count', 0)))
        row.update({f'pred_{k}': v for k, v in (agent.last_eval or {}).items()})
        if agent.last_train:
            row['ttt_loss'] = agent.last_train.get('dyn_loss')
            row['ttt_lr'] = agent.last_train.get('lr')
            row['gradient_steps'] = agent.last_train.get('gradient_steps')
        steps_log.append(row)
        if level_after > max_level:
            for lv in range(max_level + 1, level_after + 1):
                level_first_step.setdefault(lv, step)
            max_level = level_after
        win_levels = max(win_levels, int(info_next.get('win_levels', 0)))
        obs, info = (next_obs, info_next)
        if next_state == 'WIN':
            status = 'win'
            break
        if args.mode == 'l1' and level_after >= 1:
            status = 'l1_complete'
            break
        if args.log_every and step % args.log_every == 0:
            m = agent.metrics()
            print(f'[agent]   step {step}/{args.budget} level={max_level} resets={resets} modes={dict(mode_counter)} grads={m.get('gradient_steps', 0)} plan={plan_time:.0f}s ttt={m.get('update_seconds', 0):.0f}s', flush=True)
    elapsed = time.time() - t_start
    metrics = agent.metrics()
    n_steps = len(steps_log)
    preq = [r for r in steps_log if 'pred_exact' in r]
    summary = {'game': args.game, 'arm': args.arm, 'seed': args.seed, 'ckpt': args.ckpt, 'ckpt_dir': ckpt_tag, 'budget': args.budget, 'status': status, 'candidates': args.candidates, 'branch': args.branch, 'epsilon': args.epsilon, 'update_steps': args.update_steps, 'schedule_budget': args.schedule_budget, 'task_token_mode': args.task_token_mode, 'mask_mode': args.mask_mode, 'operation_mode': args.operation_mode, 'device': str(device), 'max_level': max_level, 'win_levels': win_levels, 'won': status == 'win', 'first_clear_step': level_first_step.get(1), 'level_first_step': {str(k): v for k, v in sorted(level_first_step.items())}, 'success_at': {str(b): int(level_first_step.get(1, 10 ** 9) <= b) for b in (100, 200, 500, 1000, 2000, 4000) if b <= args.budget}, 'n_steps': n_steps, 'n_resets': resets, 'action_dist': dict(sorted(action_counter.items())), 'plan_modes': dict(mode_counter), 'fallback_rate': float(np.mean([r['fallback'] for r in steps_log])) if steps_log else None, 'pred_exact_rate': float(np.mean([r['pred_exact'] for r in preq])) if preq else None, 'pred_exact_rate_last50': float(np.mean([r['pred_exact'] for r in preq[-50:]])) if preq else None, 'noop_rate': float(np.mean([r['pred_changed_cells'] == 0 for r in preq])) if preq else None, 'elapsed_sec': elapsed, 'plan_time_sec': plan_time}
    summary.update({k: v for k, v in metrics.items() if k not in summary})
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out_dir / 'steps.json').write_text(json.dumps(steps_log))
    if args.save_adapted and agent.predictor is not None:
        import torch
        torch.save({'model_state_dict': agent.predictor.model.state_dict(), 'env_steps': n_steps, 'game': args.game, 'seed': args.seed, 'src_ckpt': args.ckpt}, out_dir / 'adapted_model.pt')
    print(f'[agent] DONE {args.game} arm={args.arm} seed={args.seed}: max_level={max_level}/{win_levels} status={status} first_clear={level_first_step.get(1)} steps={n_steps} resets={resets} elapsed={elapsed / 60:.1f}min -> {out_dir}', flush=True)
if __name__ == '__main__':
    main()
