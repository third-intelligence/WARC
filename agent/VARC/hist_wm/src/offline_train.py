from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_WM_CORE_SRC = _PROJ_ROOT / 'wm_core' / 'src'
for _p in (str(_PROJ_ROOT), str(_WM_CORE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.args import resolve_global_batch_size
from utils.distribution import init_distributed_mode
from utils.lr_scheduler import get_cosine_schedule_with_warmup
from varc2.constants import IGNORE_INDEX, NUM_ACTIONS
from varc2.losses import bc_joint_loss, ffp_loss
from wm_core.src.data import build_fold_split
from wm_core.src.models import build_optimizers, split_param_groups
from wm_core.src.offline_train import _atomic_save, _format_eta, inject_ood_task_tokens, set_seed
from hist_wm.src.hist_data import build_fold_loaders_hist
from hist_wm.src.hist_model import ARCH_NAME, build_hist_model

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='hist_wm offline joint training (policy_hist)')
    p.add_argument('--fold', type=int, default=5, choices=[1, 2, 3, 4, 5])
    p.add_argument('--data-root', type=str, default='raw_data/ARC-AGI-3')
    p.add_argument('--save-dir', type=str, default=None, help='default: saves/hist_wm_policy_hist_fold{fold}')
    p.add_argument('--patch-size', type=int, default=2)
    p.add_argument('--embed-dim', type=int, default=None)
    p.add_argument('--enc-layers', type=int, default=None)
    p.add_argument('--dyn-layers', type=int, default=None)
    p.add_argument('--dec-layers', type=int, default=None)
    p.add_argument('--num-heads', type=int, default=None)
    p.add_argument('--mlp-ratio', type=float, default=None)
    p.add_argument('--hist-n', type=int, default=2, help='observation window length n (o_{t-n+1..t}; n-1 intermediate actions)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--global-batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=0.0003)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--lr-scheduler', type=str, default='cosine', choices=('none', 'cosine'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--optimizer', type=str, default='muon', choices=('muon', 'adamw'))
    p.add_argument('--muon-lr', type=float, default=0.0)
    p.add_argument('--muon-momentum', type=float, default=0.95)
    p.add_argument('--muon-adjust-lr', type=str, default='match_rms_adamw', choices=('match_rms_adamw', 'original'))
    p.add_argument('--lambda-policy', type=float, default=1.0)
    p.add_argument('--lambda-dyn', type=float, default=1.0)
    p.add_argument('--lambda-value', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=0.95)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default='bf16', choices=('fp16', 'bf16'))
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--compile-mode', type=str, default='reduce-overhead', choices=('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'))
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--disable-augment', action='store_true')
    p.add_argument('--num-color-permute', type=int, default=9)
    p.add_argument('--eval-batch-size', type=int, default=None)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--resume-checkpoint', type=str, default=None)
    p.add_argument('--use-wandb', dest='use_wandb', action='store_true', default=True)
    p.add_argument('--no-wandb', dest='use_wandb', action='store_false')
    p.add_argument('--wandb-project', type=str, default='warc')
    p.add_argument('--wandb-run-name', type=str, default='')
    p.add_argument('--wandb-resume-id', type=str, default='')
    args = p.parse_args()
    args.arch = ARCH_NAME
    args.ffp_horizon = 1
    args.lambda_recon = 0.0
    if args.save_dir is None:
        args.save_dir = f'saves/hist_wm_policy_hist_fold{args.fold}'
    if not args.wandb_run_name:
        args.wandb_run_name = f'hist_wm_policy_hist_fold{args.fold}'
    return args

def model_overrides(args) -> Dict[str, Any]:
    return {k: getattr(args, k) for k in ('embed_dim', 'enc_layers', 'dyn_layers', 'dec_layers', 'num_heads', 'mlp_ratio')}

def _hist_to_device(batch, device):
    return (batch['hist_obs'].to(device, non_blocking=True), batch['hist_action_id'].to(device, non_blocking=True), batch['hist_action_x'].to(device, non_blocking=True), batch['hist_action_y'].to(device, non_blocking=True))

@torch.no_grad()
def evaluate_per_level(model, loader, meta, device, *, amp_enabled: bool, amp_dtype) -> Dict[str, Any]:
    model.eval()
    games, trajs, levels = meta
    tl_stats: Dict[tuple, list] = {}
    traj_game: Dict[str, str] = {}
    pos = 0
    autocast_device = device.type if device.type in {'cuda', 'cpu', 'mps'} else 'cuda'
    for batch in loader:
        inputs = batch['inputs'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        task_ids = batch['task_ids'].to(device, non_blocking=True)
        action_ids = batch['action_id'].to(device, non_blocking=True)
        action_xs = batch['action_x'].to(device, non_blocking=True)
        action_ys = batch['action_y'].to(device, non_blocking=True)
        h_obs, h_aid, h_ax, h_ay = _hist_to_device(batch, device)
        B = inputs.size(0)
        with autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(inputs, task_ids, action_ids, action_xs, action_ys, hist_obs=h_obs, hist_action_ids=h_aid, hist_action_xs=h_ax, hist_action_ys=h_ay)['ffp_logits']
        logits = logits.float()
        num_colors = logits.size(1)
        ce = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, num_colors), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction='none').view(B, -1).mean(dim=1)
        exact = (logits.argmax(dim=1) == targets).flatten(1).all(dim=1)
        ce_l = ce.cpu().tolist()
        ex_l = exact.cpu().tolist()
        for i in range(B):
            tid, lvl = (trajs[pos + i], levels[pos + i])
            traj_game[tid] = games[pos + i]
            st = tl_stats.setdefault((tid, lvl), [0, 0, 0.0])
            st[0] += 1
            st[1] += int(ex_l[i])
            st[2] += ce_l[i]
        pos += B
    model.train()
    overall_n = sum((st[0] for st in tl_stats.values()))
    overall = {'exact_acc': sum((st[1] for st in tl_stats.values())) / max(overall_n, 1), 'pixel_ce': sum((st[2] for st in tl_stats.values())) / max(overall_n, 1), 'n_samples': overall_n}
    per_level: Dict[str, Dict[str, Dict[str, float]]] = {}
    gl_accs: Dict[tuple, list] = {}
    for (tid, lvl), (n, nex, ces) in tl_stats.items():
        gl_accs.setdefault((traj_game[tid], lvl), []).append([nex / n, ces / n, float(n)])
    for (game, lvl), rows in sorted(gl_accs.items()):
        d = per_level.setdefault(game, {})
        d[str(lvl)] = {'exact_acc': sum((r[0] for r in rows)) / len(rows), 'pixel_ce': sum((r[1] for r in rows)) / len(rows), 'n_trajs': len(rows), 'n_samples': int(sum((r[2] for r in rows)))}
    traj_tot: Dict[str, list] = {}
    for (tid, _lvl), (n, nex, ces) in tl_stats.items():
        st = traj_tot.setdefault(tid, [0, 0, 0.0])
        st[0] += n
        st[1] += nex
        st[2] += ces
    per_game: Dict[str, Dict[str, float]] = {}
    game_rows: Dict[str, list] = {}
    for tid, (n, nex, ces) in traj_tot.items():
        game_rows.setdefault(traj_game[tid], []).append([nex / n, ces / n, float(n)])
    for game, rows in sorted(game_rows.items()):
        per_game[game] = {'exact_acc': sum((r[0] for r in rows)) / len(rows), 'pixel_ce': sum((r[1] for r in rows)) / len(rows), 'n_trajs': len(rows), 'n_samples': int(sum((r[2] for r in rows)))}
    return {'overall': overall, 'per_game': per_game, 'per_level': per_level}

def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    resolve_global_batch_size(args, world_size, distributed=distributed, rank=rank)
    set_seed(args.seed + (rank if distributed else 0))
    is_main = not distributed or rank == 0
    split = build_fold_split(args.data_root, args.fold, quiet=not is_main)
    train_ds, train_loader, train_sampler, val_ds, val_loader, test_ds, test_loader = build_fold_loaders_hist(split, hist_n=args.hist_n, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers, augment=not args.disable_augment, num_color_permute=args.num_color_permute, gamma=args.gamma, distributed=distributed, rank=rank, world_size=world_size)
    val_meta = val_ds.sample_meta()
    test_meta = test_ds.sample_meta()
    if is_main:
        print(f'Total training examples: {len(train_ds)} (val {len(val_ds)}, test {len(test_ds)}; hist_n={args.hist_n})')
    core_model, resolved_model_cfg = build_hist_model(len(split.task_lookup), patch_size=args.patch_size, hist_n=args.hist_n, device=device, **model_overrides(args))
    dyn_params, head_params = split_param_groups(core_model, 'policy')
    if is_main:
        n_params = sum((p.numel() for n, p in core_model.named_parameters() if 'task_token' not in n))
        print(f'Parameter count: {n_params / 1000000.0:.2f}M (excluding task tokens); arch={ARCH_NAME} {resolved_model_cfg}')
        print(f'param groups: dynamics {sum((p.numel() for p in dyn_params)) / 1000000.0:.2f}M / heads {sum((p.numel() for p in head_params)) / 1000000.0:.2f}M')
    model = core_model
    if not args.no_compile and hasattr(torch, 'compile'):
        if is_main:
            print(f'Applying torch.compile (mode={args.compile_mode}) before DDP...')
        model = torch.compile(model, mode=args.compile_mode)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None)
    optimizers = build_optimizers(core_model, 'policy', optimizer=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay, muon_lr=args.muon_lr, muon_momentum=args.muon_momentum, muon_adjust_lr=args.muon_adjust_lr)
    if is_main:
        names = [type(o).__name__ for o in optimizers]
        sizes = [sum((p.numel() for g in o.param_groups for p in g['params'])) / 1000000.0 for o in optimizers]
        print('optimizers: ' + ', '.join((f'{n}({s:.2f}M)' for n, s in zip(names, sizes))))
    amp_enabled = device.type == 'cuda' and (not args.no_amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = GradScaler(enabled=amp_enabled and args.amp_dtype == 'fp16')
    if is_main:
        print('AMP disabled (CPU or --no-amp)' if not amp_enabled else f'Using AMP (dtype={args.amp_dtype}, GradScaler={('on' if scaler.is_enabled() else 'off')})')
    schedulers = []
    if args.lr_scheduler == 'cosine':
        warmup = min(args.epochs // 5, 10)
        schedulers = [get_cosine_schedule_with_warmup(o, warmup, args.epochs) for o in optimizers]
    save_dir = Path(args.save_dir)
    split_record = split.describe()
    config = dict(vars(args))
    config['model_config'] = resolved_model_cfg
    config['ood_task_token'] = 'mean'
    config['world_size'] = world_size
    config['grad_clip_groups'] = 'dynamics/heads separate (isolation)'
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'split.json').write_text(json.dumps(split_record, indent=2))
        (save_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))

    def ckpt_payload(epoch: int, best_acc: float, eval_snapshot, wandb_id: str, with_optim: bool) -> Dict[str, Any]:
        payload = {'model_state': core_model.state_dict(), 'config': config, 'split': split_record, 'epoch': epoch, 'best_eval_accuracy': best_acc, 'eval': eval_snapshot, 'wandb_run_id': wandb_id}
        if with_optim:
            payload['optimizer_states'] = [o.state_dict() for o in optimizers]
            payload['scheduler_states'] = [s.state_dict() for s in schedulers]
            if scaler.is_enabled():
                payload['scaler_state'] = scaler.state_dict()
        return payload
    start_epoch = 1
    best_eval_acc = float('-inf')
    resumed_wandb_id = ''
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        state = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt['model_state'].items()}
        core_model.load_state_dict(state)
        for o, sd in zip(optimizers, ckpt.get('optimizer_states', [])):
            o.load_state_dict(sd)
        for s, sd in zip(schedulers, ckpt.get('scheduler_states', [])):
            s.load_state_dict(sd)
        if scaler.is_enabled() and 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_eval_acc = float(ckpt.get('best_eval_accuracy', float('-inf')))
        resumed_wandb_id = str(ckpt.get('wandb_run_id', '') or '')
        if is_main:
            print(f'Resumed from {args.resume_checkpoint} (epoch {start_epoch}, best {best_eval_acc:.4f})')
            prev = ckpt.get('config', {}) or {}
            drift = {k: (prev.get(k), getattr(args, k)) for k in ('arch', 'fold', 'patch_size', 'optimizer', 'hist_n', 'lambda_policy', 'lambda_dyn', 'lambda_value', 'learning_rate', 'global_batch_size', 'epochs', 'num_color_permute') if k in prev and prev.get(k) != getattr(args, k)}
            if drift:
                print('[hist_wm] WARNING: resume config drift (ckpt -> now): ' + ', '.join((f'{k}: {a!r} -> {b!r}' for k, (a, b) in drift.items())))
            if start_epoch > args.epochs:
                print(f'[hist_wm] already completed ({start_epoch - 1}/{args.epochs} epochs); nothing to train.')
    wandb_run = None
    if args.use_wandb and is_main:
        try:
            import wandb
            kwargs: Dict[str, Any] = {'project': args.wandb_project, 'name': args.wandb_run_name, 'config': {**config, 'split_counts': split_record['counts']}}
            resume_id = (args.wandb_resume_id or resumed_wandb_id or '').strip()
            if resume_id:
                kwargs.update(id=resume_id, resume='allow')
            wandb_run = wandb.init(**kwargs)
            print(f'[hist_wm] wandb: project={args.wandb_project} run={args.wandb_run_name} id={wandb_run.id}')
        except Exception as exc:
            print(f'[hist_wm] WARNING: wandb init failed ({exc!r}); continuing WITHOUT wandb logging')
            wandb_run = None
    wandb_id = wandb_run.id if wandb_run is not None else ''
    global_start = time.time()
    prev_steps = 0
    latest_eval: Optional[Dict[str, Any]] = None
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            total_batches = len(train_loader)
            epoch_start = time.time()
            sums = torch.zeros(6, dtype=torch.float64, device=device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for step, batch in enumerate(train_loader, 1):
                inputs = batch['inputs'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                task_ids = batch['task_ids'].to(device, non_blocking=True)
                action_ids = batch['action_id'].to(device, non_blocking=True)
                action_xs = batch['action_x'].to(device, non_blocking=True)
                action_ys = batch['action_y'].to(device, non_blocking=True)
                value_target = batch['value_target'].to(device, non_blocking=True)
                h_obs, h_aid, h_ax, h_ay = _hist_to_device(batch, device)
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                autocast_device = device.type if device.type in {'cuda', 'cpu', 'mps'} else 'cuda'
                with autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
                    out = model(inputs, task_ids, action_ids, action_xs, action_ys, hist_obs=h_obs, hist_action_ids=h_aid, hist_action_xs=h_ax, hist_action_ys=h_ay)
                    l_dyn = ffp_loss(out['ffp_logits'], targets, ignore_index=IGNORE_INDEX)
                    l_pol, _bc_parts = bc_joint_loss(out['type_logits'], out['click_logits'], action_ids, action_xs, action_ys, num_actions=NUM_ACTIONS)
                    l_val = F.mse_loss(out['value'].float(), value_target.float())
                    loss = args.lambda_policy * l_pol + args.lambda_dyn * l_dyn + args.lambda_value * l_val
                scaler.scale(loss).backward()
                for opt in optimizers:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(dyn_params, args.max_grad_norm)
                if head_params:
                    torch.nn.utils.clip_grad_norm_(head_params, args.max_grad_norm)
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
                B = inputs.size(0)
                with torch.no_grad():
                    exact = (out['ffp_logits'].argmax(dim=1) == targets).flatten(1).all(dim=1)
                    sums += torch.stack([loss.detach().double() * B, l_dyn.detach().double() * B, l_pol.detach().double() * B, l_val.detach().double() * B, exact.double().sum(), torch.tensor(float(B), dtype=torch.float64, device=device)])
                if is_main and step % 10 == 0 and (total_batches > 0):
                    done = prev_steps + step
                    total = total_batches * args.epochs
                    eta = (total - done) * (time.time() - global_start) / max(done, 1)
                    bar = '#' * int(30 * done / total) + '-' * (30 - int(30 * done / total))
                    sys.stdout.write(f'\rEpoch {epoch} [{bar}] {100 * done / total:5.1f}% ETA {_format_eta(eta)}')
                    sys.stdout.flush()
            if is_main and total_batches > 0:
                sys.stdout.write('\n')
            prev_steps += total_batches
            epoch_time = time.time() - epoch_start
            if distributed and dist.is_initialized():
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            t_loss, t_dyn, t_bc, t_val, t_exact, t_n = sums.tolist()
            n = max(t_n, 1)
            train_loss, dyn_loss_avg = (t_loss / n, t_dyn / n)
            bc_loss_avg, value_loss_avg = (t_bc / n, t_val / n)
            train_acc = t_exact / n
            current_lr = optimizers[-1].param_groups[0]['lr']
            run_eval = is_main and (epoch % args.eval_every == 0 or epoch == args.epochs)
            log_parts = [f'epoch={epoch}', f'train_loss={train_loss:.4f}', f'ffp={dyn_loss_avg:.4f}', f'bc={bc_loss_avg:.4f}', f'value={value_loss_avg:.4f}', f'train_acc={train_acc:.4f}', f'lr={current_lr:.6f}', f'epoch_time={epoch_time:.1f}s']
            if device.type == 'cuda':
                log_parts.append(f'gpu_peak_alloc={torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f}GiB')
            train_metrics = {'epoch': epoch, 'train/loss': train_loss, 'train/ffp_loss': dyn_loss_avg, 'train/bc_loss': bc_loss_avg, 'train/value_loss': value_loss_avg, 'train/accuracy': train_acc, 'train/lr': current_lr, 'train/epoch_time': epoch_time}
            if len(optimizers) > 1:
                train_metrics['train/lr_muon'] = optimizers[0].param_groups[0]['lr']
            if run_eval:
                inject_ood_task_tokens(core_model, split)
                val_metrics = evaluate_per_level(core_model, val_loader, val_meta, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
                test_metrics = evaluate_per_level(core_model, test_loader, test_meta, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
                latest_eval = {'epoch': epoch, 'val': val_metrics, 'test': test_metrics}
                with (save_dir / 'eval_history.jsonl').open('a') as fh:
                    fh.write(json.dumps(latest_eval) + '\n')
                va = val_metrics['overall']
                log_parts.append(f'val_exact={va['exact_acc']:.4f}')
                log_parts.append(f'val_ce={va['pixel_ce']:.4f}')
                print(' | '.join(log_parts))
                for game in split.test_games:
                    lv = test_metrics['per_level'].get(game, {})
                    lv_txt = ' '.join((f'lv{k}={v['exact_acc']:.3f}({v['n_trajs']}t)' for k, v in sorted(lv.items(), key=lambda kv: int(kv[0]))))
                    og = test_metrics['per_game'].get(game, {})
                    print(f'  [test {game}] exact={og.get('exact_acc', 0.0):.4f} {lv_txt}')
                if wandb_run is not None:
                    m = dict(train_metrics)
                    m.update({'val/exact_acc': va['exact_acc'], 'val/pixel_ce': va['pixel_ce']})
                    for game, d in val_metrics['per_game'].items():
                        m[f'val/{game}/exact_acc'] = d['exact_acc']
                        m[f'val/{game}/pixel_ce'] = d['pixel_ce']
                    for game, d in test_metrics['per_game'].items():
                        m[f'test/{game}/exact_acc'] = d['exact_acc']
                        m[f'test/{game}/pixel_ce'] = d['pixel_ce']
                    for game, lvls in test_metrics['per_level'].items():
                        for lvl, d in lvls.items():
                            m[f'test/{game}/lv{lvl}/exact_acc'] = d['exact_acc']
                            m[f'test/{game}/lv{lvl}/pixel_ce'] = d['pixel_ce']
                    if best_eval_acc > float('-inf'):
                        m['val/best_exact_acc'] = max(best_eval_acc, va['exact_acc'])
                    wandb_run.log(m, step=epoch)
                if va['exact_acc'] > best_eval_acc:
                    best_eval_acc = va['exact_acc']
                    _atomic_save(ckpt_payload(epoch, best_eval_acc, latest_eval, wandb_id, with_optim=False), save_dir / 'checkpoint_best.pt')
            elif is_main:
                print(' | '.join(log_parts))
                if wandb_run is not None:
                    wandb_run.log(train_metrics, step=epoch)
            for s in schedulers:
                s.step()
            if is_main:
                _atomic_save(ckpt_payload(epoch, best_eval_acc, latest_eval, wandb_id, with_optim=True), save_dir / 'checkpoint_last.pt')
            if distributed and dist.is_initialized():
                dist.barrier()
        ran_epochs = start_epoch <= args.epochs
        if is_main and (ran_epochs or not (save_dir / 'checkpoint_final.pt').exists()):
            _atomic_save(ckpt_payload(args.epochs, best_eval_acc, latest_eval, wandb_id, with_optim=False), save_dir / 'checkpoint_final.pt')
        if is_main:
            print(f'Done. best val exact_acc={best_eval_acc:.4f} -> {save_dir}')
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
if __name__ == '__main__':
    train(parse_args())
