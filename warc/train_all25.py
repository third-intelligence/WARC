"""Pretraining of the submission model on all 25 public games.

train.py's recipe (the model builder replaced by warc_model.model.build_model) with the split of all25_split.py
(fold 0 = all 25 games train, validation = one recording per game). The two patches touch different names of the
same trainer module.

    cd warc
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PYTHONDONTWRITEBYTECODE=1 python -B -m torch.distributed.run \\
        --nproc_per_node=6 --master_port=29615 train_all25.py --data-root /path/to/ARC-AGI-3 \\
        --distributed --global-batch-size 516 --epochs 30 --num-workers 8 --no-wandb --save-dir runs/all25_s0

    python train_all25.py --data-root /path/to/ARC-AGI-3 --dry-run     # CPU: the split, the loaders, the model

--save-dir defaults to runs/all25_s0 under this directory. Resume: add `--resume-checkpoint <save-dir>/checkpoint_last.pt`.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as t  # noqa: E402  (sys.path, the trainer module, the model patch, split_args)
import all25_split as s  # noqa: E402
from train import ot  # noqa: E402


def main(argv):
    if '--fold' in argv:
        raise SystemExit('train_all25.py has no folds (fold 0 = all 25 games); drop --fold')
    ours, rest = t.split_args(argv, default_save='runs/all25_s0')
    t.patch(ours['eval_max_batches'])      # the model
    s.patch()                              # the split
    argv2 = s.default_args(rest)           # --fold 0
    args = ot.parse_args(argv2)
    if ours['dry_run']:
        s.dry_run(args.data_root)
        t.dry_run(args)
        return
    t.write_provenance(argv2, args, extra=dict(split='all 25 games train; validation = one recording per game'))
    ot.train(args)


if __name__ == '__main__':
    main(sys.argv[1:])
