from __future__ import annotations
import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PROJ_ROOT), str(_PROJ_ROOT / 'wm_core' / 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from varc2.constants import FOLDS as FOLDS_5
ALL_GAMES = sorted((g for games in FOLDS_5.values() for g in games))
NO_CLICK_GAMES = ('g50t', 'ls20', 're86', 'tr87', 'tu93', 'wa30')
FOCUS_GAMES = ('bp35', 'cn04', 'g50t', 'lf52', 'sp80')
NUM_FOLDS = 3
FOLD_SIZES = (9, 8, 8)
SPLIT_SEED = 0
FOLDS = tuple(range(1, NUM_FOLDS + 1))
CKPT_TEMPLATE = 'saves/unified_3fold_s{seed}_fold{fold}_n3_ep30/checkpoint_best.pt'

def _ok(folds: List[List[str]]) -> bool:
    no_click = [sum((g in NO_CLICK_GAMES for g in f)) for f in folds]
    focus = sorted((sum((g in FOCUS_GAMES for g in f)) for f in folds))
    return no_click == [2, 2, 2] and focus == [1, 2, 2]

def make_folds(seed: int=SPLIT_SEED) -> Dict[int, List[str]]:
    assert len(ALL_GAMES) == sum(FOLD_SIZES) == 25
    rng = random.Random(seed)
    for _ in range(100000):
        perm = rng.sample(ALL_GAMES, len(ALL_GAMES))
        folds, i = ([], 0)
        for n in FOLD_SIZES:
            folds.append(sorted(perm[i:i + n]))
            i += n
        if _ok(folds):
            return {k + 1: f for k, f in enumerate(folds)}
    raise RuntimeError('no split satisfies the constraints')

@contextlib.contextmanager
def patched_folds(seed: int=SPLIT_SEED):
    import wm_core.src.data as data_mod
    original = data_mod.FOLDS
    data_mod.FOLDS = make_folds(seed)
    try:
        yield data_mod.FOLDS
    finally:
        data_mod.FOLDS = original

def build_fold3_split(data_root, fold: int, *, seed: int=SPLIT_SEED, quiet: bool=False):
    from wm_core.src.data import build_fold_split
    with patched_folds(seed):
        return build_fold_split(data_root, fold, quiet=quiet)

def ckpt_of(fold: int, seed: int=SPLIT_SEED, root: Path=_PROJ_ROOT) -> Path:
    return root / CKPT_TEMPLATE.format(seed=seed, fold=fold)

def game_to_fold(root: Path=_PROJ_ROOT, seed: int=SPLIT_SEED) -> Dict[str, int]:
    expected = make_folds(seed)
    mapping: Dict[str, int] = {}
    for fold in FOLDS:
        split_path = ckpt_of(fold, seed, root).parent / 'split.json'
        if split_path.exists():
            saved = sorted(json.loads(split_path.read_text())['test_games'])
            if saved != expected[fold]:
                raise ValueError(f'test_games in {split_path} differ from the seed={seed} split: {saved}')
        for game in expected[fold]:
            mapping[game] = fold
    return mapping

def binding(game: str, root: Path=_PROJ_ROOT, seed: int=SPLIT_SEED) -> Tuple[int, Path]:
    mapping = game_to_fold(root, seed)
    if game not in mapping:
        raise ValueError(f'unknown game: {game}')
    return (mapping[game], ckpt_of(mapping[game], seed, root))

def click_games_from_recordings(data_root: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for game_dir in sorted((p for p in data_root.glob('data/*/*') if p.is_dir())):
        click = False
        for f in game_dir.glob('*.jsonl'):
            with f.open() as fh:
                for line in fh:
                    if 6 in (json.loads(line).get('data', {}).get('available_actions') or []):
                        click = True
                        break
            if click:
                break
        out[game_dir.name] = click
    return out

def main() -> None:
    p = argparse.ArgumentParser(description='unified_wm 3-fold split')
    p.add_argument('--seed', type=int, default=SPLIT_SEED)
    p.add_argument('--games-of', type=int, default=0, help='print the test games of this fold, space-separated')
    p.add_argument('--verify-click', action='store_true')
    p.add_argument('--data-root', default=str(_PROJ_ROOT / 'raw_data' / 'ARC-AGI-3'))
    a = p.parse_args()
    folds = make_folds(a.seed)
    if a.games_of:
        print(' '.join(folds[a.games_of]))
        return
    if a.verify_click:
        rec = click_games_from_recordings(Path(a.data_root))
        derived = tuple(sorted((g for g, c in rec.items() if not c)))
        print(f'from recordings: no click = {derived}')
        assert derived == NO_CLICK_GAMES, f'NO_CLICK_GAMES {NO_CLICK_GAMES} mismatch'
        print('NO_CLICK_GAMES matches')
    for k, games in folds.items():
        nc = [g for g in games if g in NO_CLICK_GAMES]
        fc = [g for g in games if g in FOCUS_GAMES]
        print(f'fold {k} ({len(games)} game): {' '.join(games)}\n    no click {len(nc)}: {' '.join(nc)} / focus {len(fc)}: {' '.join(fc)}')
if __name__ == '__main__':
    main()
