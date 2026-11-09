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
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.args import resolve_global_batch_size
from utils.distribution import init_distributed_mode
from utils.lr_scheduler import get_cosine_schedule_with_warmup
from varc2.constants import IGNORE_INDEX
from varc2.losses import ffp_loss
from varc2.optim import Muon
from wm_core.src.offline_train import _atomic_save, _format_eta, set_seed
from hist_wm.src.hist_data import build_fold_loaders_hist
from hist_wm.src.offline_train import _hist_to_device, evaluate_per_level
from unified_wm.src.folds import FOLDS, SPLIT_SEED, build_fold3_split, make_folds
from unified_wm.src.model import ARCH_NAME, STACK_NAME, build_unified_model

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='unified_wm offline training (unified_hist, FFP only)')
    p.add_argument('--fold', type=int, default=1, choices=list(FOLDS))
    p.add_argument('--split-seed', type=int, default=SPLIT_SEED, help='seed that selects the 3-fold split (unified_wm/src/folds.py)')
    p.add_argument('--data-root', type=str, default='raw_data/ARC-AGI-3')
    p.add_argument('--save-dir', type=str, default=None, help='default: saves/unified_wm_unified_3fold_s{split_seed}_fold{fold}_n{hist_n}_ep{epochs}')
    p.add_argument('--patch-size', type=int, default=2)
    p.add_argument('--embed-dim', type=int, default=None)
    p.add_argument('--depth', type=int, default=None)
    p.add_argument('--num-heads', type=int, default=None)
    p.add_argument('--mlp-dim', type=int, default=None)
    p.add_argument('--hist-n', type=int, default=3)
    p.add_argument('--epochs', type=int, default=30)
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
    p.add_argument('--gamma', type=float, default=0.95, help='for the value targets built by the data loader (not used by the loss)')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--amp-dtype', type=str, default='bf16', choices=('fp16', 'bf16'))
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--compile-mode', type=str, default='reduce-overhead', choices=('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'))
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--disable-augment', action='store_true')
    p.add_argument('--num-color-permute', type=int, default=9)
    p.add_argument('--eval-batch-size', type=int, default=None)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--max-train-batches', type=int, default=0, help='>0 cuts every epoch to this many batches (smoke tests)')
    p.add_argument('--resume-checkpoint', type=str, default=None)
    p.add_argument('--use-wandb', dest='use_wandb', action='store_true', default=True)
    p.add_argument('--no-wandb', dest='use_wandb', action='store_false')
    p.add_argument('--wandb-project', type=str, default='warc')
    p.add_argument('--wandb-run-name', type=str, default='')
    p.add_argument('--wandb-resume-id', type=str, default='')
    args = p.parse_args(argv)
    args.arch = ARCH_NAME
    if args.save_dir is None:
        args.save_dir = f'saves/unified_wm_unified_3fold_s{args.split_seed}_fold{args.fold}_n{args.hist_n}_ep{args.epochs}'
    if not args.wandb_run_name:
        args.wandb_run_name = Path(args.save_dir).name
    return args

def build_optimizers(model, *, optimizer: str, lr: float, weight_decay: float, muon_lr: float, muon_momentum: float, muon_adjust_lr: str):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if optimizer == 'adamw':
        return [torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=weight_decay)]
    muon, adamw = ([], [])
    for name, p in trainable:
        is_muon = p.ndim == 2 and name.endswith('.weight') and (name.split('.', 1)[0] == STACK_NAME) and ('ada_mod' not in name)
        (muon if is_muon else adamw).append(p)
    assert muon, 'muon group is empty -- stack naming changed?'
    return [Muon(muon, lr=muon_lr if muon_lr > 0 else lr, weight_decay=weight_decay, momentum=muon_momentum, adjust_lr_fn=muon_adjust_lr), torch.optim.AdamW(adamw, lr=lr, weight_decay=weight_decay)]

def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    resolve_global_batch_size(args, world_size, distributed=distributed, rank=rank)
    set_seed(args.seed + (rank if distributed else 0))
    is_main = not distributed or rank == 0
    split = build_fold3_split(args.data_root, args.fold, seed=args.split_seed, quiet=not is_main)
    assert sorted(split.test_games) == make_folds(args.split_seed)[args.fold]
    train_ds, train_loader, train_sampler, val_ds, val_loader, test_ds, test_loader = build_fold_loaders_hist(split, hist_n=args.hist_n, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers, augment=not args.disable_augment, num_color_permute=args.num_color_permute, gamma=args.gamma, distributed=distributed, rank=rank, world_size=world_size)
    val_meta = val_ds.sample_meta()
    test_meta = test_ds.sample_meta()
    if is_main:
        print(f'Total training examples: {len(train_ds)} (val {len(val_ds)}, test {len(test_ds)}; hist_n={args.hist_n})')
    core_model, resolved_model_cfg = build_unified_model(hist_n=args.hist_n, patch_size=args.patch_size, device=device, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_dim=args.mlp_dim)
    params = [p for p in core_model.parameters() if p.requires_grad]
    if is_main:
        print(f'Parameter count: {sum((p.numel() for p in params)) / 1000000.0:.2f}M; arch={ARCH_NAME} {resolved_model_cfg}')
    model = core_model
    if not args.no_compile and hasattr(torch, 'compile'):
        if is_main:
            print(f'Applying torch.compile (mode={args.compile_mode}) before DDP...')
        model = torch.compile(model, mode=args.compile_mode)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, output_device=local_rank if device.type == 'cuda' else None)
    optimizers = build_optimizers(core_model, optimizer=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay, muon_lr=args.muon_lr, muon_momentum=args.muon_momentum, muon_adjust_lr=args.muon_adjust_lr)
    if is_main:
        sizes = [sum((p.numel() for g in o.param_groups for p in g['params'])) / 1000000.0 for o in optimizers]
        print('optimizers: ' + ', '.join((f'{type(o).__name__}({s:.2f}M)' for o, s in zip(optimizers, sizes))))
    amp_enabled = device.type == 'cuda' and (not args.no_amp)
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    scaler = GradScaler(enabled=amp_enabled and args.amp_dtype == 'fp16')
    schedulers = []
    if args.lr_scheduler == 'cosine':
        warmup = min(args.epochs // 5, 10)
        schedulers = [get_cosine_schedule_with_warmup(o, warmup, args.epochs) for o in optimizers]
    save_dir = Path(args.save_dir)
    split_record = split.describe()
    config = dict(vars(args))
    config['model_config'] = resolved_model_cfg
    config['world_size'] = world_size
    config['loss'] = 'ffp_only'
    config['folds'] = {'num_folds': len(FOLDS), 'split_seed': args.split_seed, 'test_games_by_fold': make_folds(args.split_seed)}
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
        core_model.load_state_dict({k.replace('_orig_mod.', '', 1): v for k, v in ckpt['model_state'].items()})
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
            drift = {k: (prev.get(k), getattr(args, k)) for k in ('arch', 'fold', 'split_seed', 'patch_size', 'optimizer', 'hist_n', 'learning_rate', 'global_batch_size', 'epochs', 'num_color_permute') if k in prev and prev.get(k) != getattr(args, k)}
            if drift:
                print('[unified_wm] WARNING: resume config drift (ckpt -> now): ' + ', '.join((f'{k}: {a!r} -> {b!r}' for k, (a, b) in drift.items())))
    wandb_run = None
    if args.use_wandb and is_main:
        try:
            import wandb
            kwargs: Dict[str, Any] = {'project': args.wandb_project, 'name': args.wandb_run_name, 'config': {**config, 'split_counts': split_record['counts']}}
            resume_id = (args.wandb_resume_id or resumed_wandb_id or '').strip()
            if resume_id:
                kwargs.update(id=resume_id, resume='allow')
            wandb_run = wandb.init(**kwargs)
        except Exception as exc:
            print(f'[unified_wm] WARNING: wandb init failed ({exc!r}); continuing WITHOUT wandb')
            wandb_run = None
    wandb_id = wandb_run.id if wandb_run is not None else ''
    global_start = time.time()
    prev_steps = 0
    latest_eval: Optional[Dict[str, Any]] = None
    autocast_device = device.type if device.type in {'cuda', 'cpu', 'mps'} else 'cuda'
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            total_batches = len(train_loader)
            if args.max_train_batches > 0:
                total_batches = min(total_batches, args.max_train_batches)
            epoch_start = time.time()
            sums = torch.zeros(3, dtype=torch.float64, device=device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for step, batch in enumerate(train_loader, 1):
                if step > total_batches:
                    break
                inputs = batch['inputs'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                action_ids = batch['action_id'].to(device, non_blocking=True)
                action_xs = batch['action_x'].to(device, non_blocking=True)
                action_ys = batch['action_y'].to(device, non_blocking=True)
                h_obs, h_aid, h_ax, h_ay = _hist_to_device(batch, device)
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                with autocast(device_type=autocast_device, enabled=amp_enabled, dtype=amp_dtype):
                    out = model(inputs, None, action_ids, action_xs, action_ys, hist_obs=h_obs, hist_action_ids=h_aid, hist_action_xs=h_ax, hist_action_ys=h_ay)
                    loss = ffp_loss(out['ffp_logits'], targets, ignore_index=IGNORE_INDEX)
                scaler.scale(loss).backward()
                for opt in optimizers:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
                B = inputs.size(0)
                with torch.no_grad():
                    exact = (out['ffp_logits'].argmax(dim=1) == targets).flatten(1).all(dim=1)
                    sums += torch.stack([loss.detach().double() * B, exact.double().sum(), torch.tensor(float(B), dtype=torch.float64, device=device)])
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
            t_ffp, t_exact, t_n = sums.tolist()
            n = max(t_n, 1)
            current_lr = optimizers[-1].param_groups[0]['lr']
            log_parts = [f'epoch={epoch}', f'ffp={t_ffp / n:.4f}', f'train_acc={t_exact / n:.4f}', f'lr={current_lr:.6f}', f'epoch_time={epoch_time:.1f}s']
            if device.type == 'cuda':
                log_parts.append(f'gpu_peak_alloc={torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f}GiB')
            train_metrics = {'epoch': epoch, 'train/loss': t_ffp / n, 'train/ffp_loss': t_ffp / n, 'train/accuracy': t_exact / n, 'train/lr': current_lr, 'train/epoch_time': epoch_time}
            if len(optimizers) > 1:
                train_metrics['train/lr_muon'] = optimizers[0].param_groups[0]['lr']
            run_eval = is_main and (epoch % args.eval_every == 0 or epoch == args.epochs)
            if run_eval:
                val_metrics = evaluate_per_level(core_model, val_loader, val_meta, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
                test_metrics = evaluate_per_level(core_model, test_loader, test_meta, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
                latest_eval = {'epoch': epoch, 'val': val_metrics, 'test': test_metrics}
                with (save_dir / 'eval_history.jsonl').open('a') as fh:
                    fh.write(json.dumps(latest_eval) + '\n')
                va = val_metrics['overall']
                log_parts += [f'val_exact={va['exact_acc']:.4f}', f'val_ce={va['pixel_ce']:.4f}']
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
