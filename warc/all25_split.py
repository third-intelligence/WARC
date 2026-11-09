"""The split of the submission model: all 25 public games are training games.

The fold models hold out 8 or 9 games each. The submitted model must have seen every public game, so the trainer's
fold split is replaced from the outside by "fold 0 = no test games": the training games are all 25 and, as in the
fold models, exactly one recording per game (the last by file name among the recordings that reach the game's top
level) is the validation set that selects checkpoint_best. Nothing else changes. The data module decides the
validation recording by file name, so the recordings must keep the names listed in data/SHA256SUMS.
"""
import time

import _paths
_paths.use_training_paths()

import wm_core.src.data as data_mod  # noqa: E402
from wm_core.src.data import build_fold_split  # noqa: E402
from unified_wm.src import offline_train as ot  # noqa: E402

ALL_FOLD = 0                       # the one "fold": no test games


def make_folds_all(seed=0):
    return {ALL_FOLD: []}


def build_all_split(data_root, fold, *, seed=0, quiet=False):
    """build_fold_split with FOLDS = {0: []}: train = all 25 games, val = one recording per game."""
    assert fold == ALL_FOLD, fold
    original = data_mod.FOLDS
    data_mod.FOLDS = make_folds_all(seed)
    try:
        return build_fold_split(data_root, ALL_FOLD, quiet=quiet)
    finally:
        data_mod.FOLDS = original


def patch():
    ot.FOLDS = (ALL_FOLD,)                 # parse_args' --fold choices
    ot.make_folds = make_folds_all         # the assert on test games and split.json's test_games_by_fold
    ot.build_fold3_split = build_all_split


def default_args(argv):
    out = list(argv)
    if '--fold' not in out:
        out += ['--fold', str(ALL_FOLD)]
    return out


def dry_run(data_root):
    """Build the split and the loaders on CPU and print their sizes."""
    from hist_wm.src.hist_data import build_fold_loaders_hist
    t0 = time.time()
    split = build_all_split(data_root, ALL_FOLD)
    n_tr = sum(len(v) for v in split.train_trajs.values())
    n_va = sum(len(v) for v in split.val_trajs.values())
    print(f'games: {len(split.train_games)} train, {len(split.test_games)} test; recordings: {n_tr} train, '
          f'{n_va} val, {sum(len(v) for v in split.test_trajs.values())} test ({time.time() - t0:.0f} s)')
    print('val recordings:', {g: [t.split("/")[1][:8] for t in v] for g, v in sorted(split.val_trajs.items())})
    (train_ds, train_loader, _, val_ds, val_loader, test_ds, test_loader) = build_fold_loaders_hist(
        split, hist_n=3, batch_size=172, eval_batch_size=None, num_workers=0, augment=True,
        num_color_permute=9, gamma=0.95, distributed=False, rank=0, world_size=1)
    games, trajs, levels = test_ds.sample_meta()
    print(f'examples: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)} (meta {len(games)}); '
          f'train batches per epoch at 516: {len(train_ds) // 516}')
    batch = next(iter(val_loader))
    print('val batch keys:', sorted(batch.keys()), 'inputs', tuple(batch['inputs'].shape))
