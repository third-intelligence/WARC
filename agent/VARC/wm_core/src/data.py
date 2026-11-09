from __future__ import annotations
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = Path(__file__).resolve().parent
for _p in (str(_PROJ_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from torch.utils.data import DataLoader
from varc2.constants import FOLDS, GEOMETRIC_TRANSFORMS, NUM_ACTIONS, NUM_COLORS
from varc2.data import AA3Dataset, Step, apply_geometric_transform, available_list_to_mask, collate_fn, parse_recording, transform_action6_coord, value_targets_from_steps
from varc2.constants import ACTION_REMAP
SPLIT_DIRS = ('training', 'evaluation')
ALL_GAMES = sorted({g for games in FOLDS.values() for g in games})

def discover_games(data_root) -> Dict[str, List[Path]]:
    root = Path(data_root)
    found: Dict[str, List[Path]] = {}
    for split in SPLIT_DIRS:
        split_dir = root / 'data' / split
        if not split_dir.is_dir():
            continue
        for game in ALL_GAMES:
            gdir = split_dir / game
            if gdir.is_dir():
                files = sorted(gdir.glob('*.recording.jsonl'))
                if files:
                    found.setdefault(game, []).extend(files)
    missing = sorted(set(ALL_GAMES) - set(found))
    if missing:
        raise FileNotFoundError(f'fold games missing under {root}/data/{{{','.join(SPLIT_DIRS)}}}: {missing}')
    return found

def traj_max_level(steps: List[Step]) -> int:
    return max((s.levels + (1 if s.completed else 0) for s in steps))

@dataclass
class FoldSplit:
    fold: int
    train_games: List[str]
    test_games: List[str]
    task_lookup: Dict[str, int]
    game_max_level: Dict[str, int]
    train_trajs: Dict[str, List[str]]
    val_trajs: Dict[str, List[str]]
    test_trajs: Dict[str, List[str]]
    steps_by_traj: Dict[str, List[Step]] = field(repr=False)
    traj_files: Dict[str, str] = field(repr=False)

    def traj_to_task(self) -> Dict[str, str]:
        return {t: t.split('/', 1)[0] for t in self.steps_by_traj}

    def describe(self) -> Dict:
        return {'fold': self.fold, 'train_games': self.train_games, 'test_games': self.test_games, 'task_lookup': self.task_lookup, 'game_max_level': self.game_max_level, 'train_trajectories': self.train_trajs, 'val_trajectories': self.val_trajs, 'test_trajectories': self.test_trajs, 'trajectory_files': self.traj_files, 'counts': {'train_trajs': sum((len(v) for v in self.train_trajs.values())), 'val_trajs': sum((len(v) for v in self.val_trajs.values())), 'test_trajs': sum((len(v) for v in self.test_trajs.values())), 'train_steps': sum((len(self.steps_by_traj[t]) for v in self.train_trajs.values() for t in v)), 'val_steps': sum((len(self.steps_by_traj[t]) for v in self.val_trajs.values() for t in v)), 'test_steps': sum((len(self.steps_by_traj[t]) for v in self.test_trajs.values() for t in v))}}

def build_fold_split(data_root, fold: int, *, quiet: bool=False) -> FoldSplit:
    if fold not in FOLDS:
        raise ValueError(f'fold must be one of {sorted(FOLDS)}, got {fold}')
    files_by_game = discover_games(data_root)
    test_games = list(FOLDS[fold])
    train_games = sorted((g for g in ALL_GAMES if g not in set(test_games)))
    task_lookup = {g: i for i, g in enumerate(ALL_GAMES)}
    steps_by_traj: Dict[str, List[Step]] = {}
    traj_files: Dict[str, str] = {}
    trajs_by_game: Dict[str, List[str]] = {}
    for game, files in files_by_game.items():
        for f in files:
            traj_id = f'{game}/{f.stem}'
            steps = parse_recording(f)
            if not steps:
                if not quiet:
                    print(f'[wm_core.data] {traj_id}: no valid steps, dropped')
                continue
            steps_by_traj[traj_id] = steps
            traj_files[traj_id] = str(f)
            trajs_by_game.setdefault(game, []).append(traj_id)
    game_max_level = {g: max((traj_max_level(steps_by_traj[t]) for t in ts)) for g, ts in trajs_by_game.items()}
    train_trajs: Dict[str, List[str]] = {}
    val_trajs: Dict[str, List[str]] = {}
    for game in train_games:
        gmax = game_max_level[game]
        reaching = sorted((t for t in trajs_by_game[game] if traj_max_level(steps_by_traj[t]) == gmax))
        val = [reaching[-1]]
        train = [t for t in trajs_by_game[game] if t != val[0]]
        train_trajs[game] = train
        val_trajs[game] = val
        if not train and (not quiet):
            print(f'[wm_core.data] WARNING: training game {game}: only one recording -> 0 training trajectories')
    test_trajs = {g: list(trajs_by_game[g]) for g in test_games}
    if not quiet:
        n_tr = sum((len(v) for v in train_trajs.values()))
        n_va = sum((len(v) for v in val_trajs.values()))
        n_te = sum((len(v) for v in test_trajs.values()))
        print(f'[wm_core.data] fold {fold}: {len(train_games)} train games ({n_tr} train / {n_va} val trajs), {len(test_games)} test games ({n_te} trajs)')
    return FoldSplit(fold=fold, train_games=train_games, test_games=test_games, task_lookup=task_lookup, game_max_level=game_max_level, train_trajs=train_trajs, val_trajs=val_trajs, test_trajs=test_trajs, steps_by_traj=steps_by_traj, traj_files=traj_files)

class TrajSetDataset(AA3Dataset):

    def __init__(self, steps_by_traj: Dict[str, List[Step]], traj_to_task: Dict[str, str], task_lookup: Dict[str, int], *, augment: bool, num_color_permute: int=9, gamma: float=0.95, num_actions: int=NUM_ACTIONS, horizon: int=1) -> None:
        if horizon < 1:
            raise ValueError(f'horizon must be >= 1, got {horizon}')
        self.augment = augment
        self.num_color_permute = num_color_permute
        self.gamma = gamma
        self.num_actions = num_actions
        self.horizon = horizon
        self.task_lookup = dict(task_lookup)
        self.trajectories: Dict[str, List[Step]] = {}
        self.value_targets: Dict[str, List[float]] = {}
        self.aug_configs: Dict[str, List[Tuple]] = {}
        self.samples: List[Tuple[str, int, int]] = []
        self._traj_to_task: Dict[str, str] = {}
        n_steps_total = n_starts_total = 0
        for traj_id in sorted(steps_by_traj):
            steps = steps_by_traj[traj_id]
            if not steps:
                continue
            self.trajectories[traj_id] = steps
            self.value_targets[traj_id] = value_targets_from_steps(steps, gamma)
            self._traj_to_task[traj_id] = traj_to_task[traj_id]
            configs: List[Tuple] = [(None, None)]
            if augment:
                for transform_name in GEOMETRIC_TRANSFORMS:
                    configs.append((transform_name, None))
                    traj_seed = zlib.crc32(f'{traj_id}:{transform_name}'.encode())
                    traj_rng = np.random.RandomState(traj_seed & 4294967295)
                    for _ in range(num_color_permute):
                        perm = traj_rng.permutation(NUM_COLORS).astype(np.uint8)
                        configs.append((transform_name, perm))
            self.aug_configs[traj_id] = configs
            starts = self._window_starts(steps, horizon)
            n_steps_total += len(steps)
            n_starts_total += len(starts)
            for aug_idx in range(len(configs)):
                for step_idx in starts:
                    self.samples.append((traj_id, step_idx, aug_idx))
        if horizon > 1:
            print(f'[wm_core.data] horizon={horizon}: {n_starts_total}/{n_steps_total} contiguous window starts kept')
        self.num_tasks = len(self.task_lookup)

    @staticmethod
    def _window_starts(steps: List[Step], horizon: int) -> List[int]:
        n = len(steps)
        if horizon == 1:
            return list(range(n))
        contig = [bool(np.array_equal(steps[j].obs_t1, steps[j + 1].obs_t)) for j in range(n - 1)]
        return [i for i in range(n - horizon + 1) if all(contig[i:i + horizon - 1])]

    def __getitem__(self, idx: int) -> Dict:
        if self.horizon == 1:
            return super().__getitem__(idx)
        traj_id, i, aug_idx = self.samples[idx]
        steps = self.trajectories[traj_id][i:i + self.horizon]
        transform_name, color_map = self.aug_configs[traj_id][aug_idx]
        frames = [steps[0].obs_t.copy()] + [s.obs_t1.copy() for s in steps]
        aids = [s.action_id for s in steps]
        axs = [s.action_x for s in steps]
        ays = [s.action_y for s in steps]
        available = steps[0].available
        if transform_name is not None:
            frames = [apply_geometric_transform(f, transform_name) for f in frames]
            remap = ACTION_REMAP.get(transform_name, {})
            for k in range(len(steps)):
                a = remap.get(aids[k], aids[k])
                if a == 6 and axs[k] >= 0:
                    axs[k], ays[k] = transform_action6_coord(axs[k], ays[k], transform_name)
                aids[k] = a
            available = [remap.get(a, a) for a in available]
        if color_map is not None:
            frames = [color_map[f] for f in frames]
        task_name = self._traj_to_task[traj_id]
        return {'inputs': torch.tensor(frames[0], dtype=torch.long), 'action_id': torch.tensor(aids, dtype=torch.long), 'action_x': torch.tensor(axs, dtype=torch.long), 'action_y': torch.tensor(ays, dtype=torch.long), 'targets': torch.tensor(np.stack(frames[1:]), dtype=torch.long), 'task_id': torch.tensor(self.task_lookup[task_name], dtype=torch.long), 'available_mask': available_list_to_mask(available, self.num_actions), 'levels_completed': torch.tensor(steps[0].levels, dtype=torch.long), 'value_target': torch.tensor(self.value_targets[traj_id][i], dtype=torch.float32), 'task_name': task_name, 'trajectory_id': traj_id, 'step_index': i}

    def sample_meta(self) -> Tuple[List[str], List[str], List[int]]:
        games, trajs, levels = ([], [], [])
        for traj_id, step_idx, _aug in self.samples:
            games.append(self._traj_to_task[traj_id])
            trajs.append(traj_id)
            levels.append(self.trajectories[traj_id][step_idx].levels)
        return (games, trajs, levels)

def _subset(split: FoldSplit, trajs: Dict[str, List[str]]) -> Dict[str, List[Step]]:
    return {t: split.steps_by_traj[t] for ts in trajs.values() for t in ts}

def build_fold_loaders(split: FoldSplit, *, batch_size: int, eval_batch_size: Optional[int]=None, num_workers: int=4, augment: bool=True, num_color_permute: int=9, gamma: float=0.95, train_horizon: int=1, distributed: bool=False, rank: int=0, world_size: int=1):
    t2t = split.traj_to_task()
    train_ds = TrajSetDataset(_subset(split, split.train_trajs), t2t, split.task_lookup, augment=augment, num_color_permute=num_color_permute, gamma=gamma, horizon=train_horizon)
    val_ds = TrajSetDataset(_subset(split, split.val_trajs), t2t, split.task_lookup, augment=False, gamma=gamma)
    test_ds = TrajSetDataset(_subset(split, split.test_trajs), t2t, split.task_lookup, augment=False, gamma=gamma)
    if len(train_ds) == 0:
        raise RuntimeError('empty train dataset (all trajectories held out?)')
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    ebs = eval_batch_size or batch_size

    def _eval_loader(ds):
        return DataLoader(ds, batch_size=ebs, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    return (train_ds, train_loader, train_sampler, val_ds, _eval_loader(val_ds), test_ds, _eval_loader(test_ds))
