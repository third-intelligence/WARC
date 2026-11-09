"""Pretraining of the WARC world model on the human recordings of the public ARC-AGI-3 games (fold models).

Runs the vendored trainer (agent/VARC/unified_wm/src/offline_train.py) with its model
builder replaced by warc_model.model.build_model, so a checkpoint written here has the format and arch
(unified_hist_cm) of the released weights and loads in the agent without conversion. Everything else is the
trainer's recipe: the 3-fold split (unified_wm/src/folds.py, split seed 0), a window of hist_n 3 frames, patch 2,
30 epochs, global batch 516 (6 GPUs x 86), Muon + AdamW with lr 3e-4 and weight decay 0, 6 warmup epochs then
cosine, gradient clip 1, bf16 autocast, torch.compile (reduce-overhead), 9 colour permutations, seed 42, and
checkpoint_best selected by validation exact-frame accuracy. The click vector (click_embed, (4, 512), outside
`blocks`) is updated by AdamW, like the action and colour embeddings.

    cd warc
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PYTHONDONTWRITEBYTECODE=1 python -B -m torch.distributed.run \\
        --nproc_per_node=6 --master_port=29606 train.py --fold 2 --data-root /path/to/ARC-AGI-3 \\
        --distributed --global-batch-size 516 --epochs 30 --num-workers 8 --no-wandb --save-dir runs/fold2

    python train.py --fold 2 --dry-run      # CPU: resolved arguments, model config and optimizer groups; no data

Resume after an interruption: the same command plus `--resume-checkpoint <save-dir>/checkpoint_last.pt`.
--save-dir defaults to runs/fold<f> under this directory. --eval-max-batches N (smoke tests only) cuts each
evaluation to N batches; 0 (the default) leaves the trainer's evaluation unchanged.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _paths  # noqa: E402
_paths.use_training_paths()

import torch  # noqa: E402
from warc_model import model as wm  # noqa: E402
from unified_wm.src import offline_train as ot  # noqa: E402


def patch(eval_max_batches=0):
    ot.build_unified_model = wm.build_model        # the model: WarcWM, arch unified_hist_cm
    ot.ARCH_NAME = wm.ARCH_NAME                     # args.arch and the parameter-count line
    if eval_max_batches > 0:
        evaluate = ot.evaluate_per_level

        def cut(model, loader, meta, device, **kw):
            return evaluate(model, itertools.islice(loader, eval_max_batches), meta, device, **kw)
        ot.evaluate_per_level = cut


def split_args(argv, default_save=None):
    """(our flags, the trainer's argv with this script's defaults filled in)."""
    ours, rest = dict(dry_run=False, eval_max_batches=0), []
    it = iter(argv)
    for a in it:
        if a == '--dry-run':
            ours['dry_run'] = True
        elif a == '--eval-max-batches':
            ours['eval_max_batches'] = int(next(it))
        else:
            rest.append(a)
    if '--save-dir' not in rest:
        if default_save is None:
            fold = rest[rest.index('--fold') + 1] if '--fold' in rest else '1'
            default_save = f'runs/fold{fold}'
        rest += ['--save-dir', str(HERE / default_save)]
    if '--data-root' not in rest and not ours['dry_run']:
        raise SystemExit('--data-root <dir with data/training and data/evaluation> is required (see data/README.md)')
    return ours, rest


def optimizer_groups(model, args):
    """{optimizer: parameter names} as the trainer's build_optimizers routes them."""
    opts = ot.build_optimizers(model, optimizer=args.optimizer, lr=args.learning_rate,
                               weight_decay=args.weight_decay, muon_lr=args.muon_lr,
                               muon_momentum=args.muon_momentum, muon_adjust_lr=args.muon_adjust_lr)
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {type(o).__name__: [by_id[id(p)] for g in o.param_groups for p in g['params']] for o in opts}


def dry_run(args):
    model, cfg = ot.build_unified_model(hist_n=args.hist_n, patch_size=args.patch_size, embed_dim=args.embed_dim,
                                        depth=args.depth, num_heads=args.num_heads, mlp_dim=args.mlp_dim)
    groups = optimizer_groups(model, args)
    n = sum(p.numel() for p in model.parameters())
    print('args:', json.dumps(vars(args), indent=1, default=str))
    print('model_config:', json.dumps(cfg, indent=1))
    print(f'parameters: {n / 1e6:.3f}M ({n} = backbone {n - model.click_embed.numel()} + click vector '
          f'{model.click_embed.numel()}); click_embed {tuple(model.click_embed.shape)} all zero: '
          f'{bool((model.click_embed == 0).all())}')
    for name, names in groups.items():
        print(f'{name}: {len(names)} tensors' + (' (includes click_embed)' if 'click_embed' in names else ''))
    assert 'click_embed' in groups.get('AdamW', []), groups


def write_provenance(argv, args, extra=None):
    """provenance.json in the save directory: the command, the data root and the SHA-256 of the code that ran."""
    if os.environ.get('RANK', '0') != '0':
        return
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    import wm_core.src.data as data_mod
    import hist_wm.src.hist_data as hist_mod
    import unified_wm.src.model as backbone_mod
    files = {'warc/train.py': Path(__file__), 'warc_model/model.py': Path(wm.__file__),
             'unified_wm/src/offline_train.py': Path(ot.__file__),
             'unified_wm/src/folds.py': Path(ot.__file__).with_name('folds.py'),
             'unified_wm/src/model.py': Path(backbone_mod.__file__), 'hist_wm/src/hist_data.py': Path(hist_mod.__file__),
             'wm_core/src/data.py': Path(data_mod.__file__)}
    prov = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), argv=argv, data_root=args.data_root, arch=args.arch,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), world_size=os.environ.get('WORLD_SIZE'),
                torch=torch.__version__,
                sha256={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()})
    if extra:
        prov.update(extra)
    name = 'provenance.json' if not args.resume_checkpoint else f'provenance_resumed_{time.strftime("%Y%m%d_%H%M%S")}.json'
    (save_dir / name).write_text(json.dumps(prov, indent=1) + '\n')


if __name__ == '__main__':
    ours, argv = split_args(sys.argv[1:])
    patch(ours['eval_max_batches'])
    args = ot.parse_args(argv)
    if ours['dry_run']:
        dry_run(args)
        sys.exit(0)
    write_provenance(argv, args)
    ot.train(args)      # the trainer's own entry point: train(parse_args())
