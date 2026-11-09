from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from src.ARC_loader_AA3 import parse_jsonl_trajectory
Transition = Tuple
FOLD_GAMES: Dict[int, List[str]] = {1: ['ls20', 'lp85', 'bp35', 'su15', 'tr87'], 2: ['cn04', 'ft09', 'ka59', 'tn36', 'tu93'], 3: ['cd82', 'dc22', 'lf52', 's5i5', 'wa30'], 4: ['m0r0', 're86', 'sb26', 'sc25', 'vc33'], 5: ['ar25', 'g50t', 'r11l', 'sk48', 'sp80']}
GAME_FOLD: Dict[str, int] = {g: k for k, gs in FOLD_GAMES.items() for g in gs}
GAME_ORDER: List[str] = [g for k in sorted(FOLD_GAMES) for g in FOLD_GAMES[k]]

def fold_ckpt(fold: int, saves_root: str) -> Path:
    name = 'policy_value' if fold == 1 else f'policy_value_fold{fold}'
    return Path(saves_root) / name / 'checkpoint_best.pt'

def find_recordings(game: str, data_root: Path) -> List[Path]:
    recs: List[Path] = []
    for split in ('training', 'evaluation'):
        d = data_root / split / game
        if d.is_dir():
            recs.extend(sorted(d.glob('*.recording.jsonl')))
    return sorted(recs)

def _subsample(items: List, cap: int, rng: random.Random) -> List:
    if cap <= 0 or len(items) <= cap:
        return list(items)
    idx = sorted(rng.sample(range(len(items)), cap))
    return [items[i] for i in idx]

def load_game_transitions(game: str, *, data_root: Path, eval_holdout: int=2, max_train_steps: int=1000, eval_cap: int=1024, seed: int=0) -> Tuple[List[Transition], List[Transition], Dict]:
    rng = random.Random(seed)
    recs = find_recordings(game, data_root)
    if not recs:
        raise FileNotFoundError(f'no recordings for game={game!r} under {data_root}')
    n_hold = min(max(0, eval_holdout), max(0, len(recs) - 1))
    train_recs = recs[:len(recs) - n_hold] if n_hold > 0 else recs
    eval_recs = recs[len(recs) - n_hold:] if n_hold > 0 else recs[-1:]
    train_stream: List[Transition] = []
    for r in train_recs:
        train_stream.extend(parse_jsonl_trajectory(r))
        if max_train_steps and len(train_stream) >= max_train_steps:
            break
    if max_train_steps:
        train_stream = train_stream[:max_train_steps]
    eval_all: List[Transition] = []
    for r in eval_recs:
        eval_all.extend(parse_jsonl_trajectory(r))
    eval_set = _subsample(eval_all, eval_cap, rng)
    info = {'game': game, 'fold': GAME_FOLD.get(game), 'n_recordings': len(recs), 'n_train_recordings': len(train_recs), 'n_eval_recordings': len(eval_recs), 'n_train_transitions': len(train_stream), 'n_eval_transitions': len(eval_set), 'n_eval_transitions_raw': len(eval_all)}
    return (train_stream, eval_set, info)
