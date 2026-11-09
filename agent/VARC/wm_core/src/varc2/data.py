from __future__ import annotations
import json
import random
import zlib
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from varc2.constants import ACTION_REMAP, FRAME_SIZE, GEOMETRIC_TRANSFORMS, NUM_ACTIONS, NUM_COLORS, normalize_action_id

class Step(NamedTuple):
    obs_t: np.ndarray
    action_id: int
    action_x: int
    action_y: int
    obs_t1: np.ndarray
    available: List[int]
    levels: int
    completed: bool

def parse_recording(filepath: Path) -> List[Step]:
    lines = []
    with filepath.open('r') as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    steps: List[Step] = []
    for i in range(len(lines) - 1):
        d0 = lines[i]['data']
        d1 = lines[i + 1]['data']
        if not d0.get('frame') or not d1.get('frame'):
            continue
        if d0.get('state') == 'GAME_OVER':
            continue
        action_input = d1.get('action_input')
        if action_input is None:
            continue
        action_id = normalize_action_id(action_input.get('id', 0))
        if action_id == 0:
            continue
        obs_t = np.array(d0['frame'][-1], dtype=np.uint8)
        obs_t1 = np.array(d1['frame'][-1], dtype=np.uint8)
        action_x = action_y = -1
        if action_id == 6:
            coord = action_input.get('data') or {}
            action_x = int(coord.get('x', -1))
            action_y = int(coord.get('y', -1))
        lv0 = int(d0.get('levels_completed', 0))
        lv1 = int(d1.get('levels_completed', 0))
        completed = lv1 > lv0 or d1.get('state') == 'WIN'
        steps.append(Step(obs_t=obs_t, action_id=action_id, action_x=action_x, action_y=action_y, obs_t1=obs_t1, available=[int(a) for a in d0.get('available_actions') or []], levels=lv0, completed=completed))
    return steps

def value_targets_from_steps(steps: Sequence[Step], gamma: float) -> List[float]:
    n = len(steps)
    y = [0.0] * n
    next_done: Optional[int] = None
    for t in range(n - 1, -1, -1):
        if steps[t].completed:
            next_done = t
        y[t] = 0.0 if next_done is None else gamma ** (next_done - t)
    return y

def apply_geometric_transform(grid: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == 'rotate90_cw':
        return np.rot90(grid, k=-1).copy()
    elif transform_name == 'rotate90_ccw':
        return np.rot90(grid, k=1).copy()
    elif transform_name == 'rotate180':
        return np.rot90(grid, k=2).copy()
    elif transform_name == 'flip_vertical':
        return np.flipud(grid).copy()
    elif transform_name == 'flip_horizontal':
        return np.fliplr(grid).copy()
    return grid.copy()

def transform_action6_coord(x: int, y: int, transform_name: str) -> Tuple[int, int]:
    if transform_name == 'rotate90_cw':
        return (63 - y, x)
    elif transform_name == 'rotate90_ccw':
        return (y, 63 - x)
    elif transform_name == 'rotate180':
        return (63 - x, 63 - y)
    elif transform_name == 'flip_horizontal':
        return (63 - x, y)
    elif transform_name == 'flip_vertical':
        return (x, 63 - y)
    return (x, y)

def available_list_to_mask(available: List[int], num_actions: int=NUM_ACTIONS) -> torch.Tensor:
    mask = torch.zeros(num_actions, dtype=torch.bool)
    if not available:
        mask[:] = True
        return mask
    for a in available:
        if 0 <= a < num_actions:
            mask[a] = True
    return mask

class AA3Dataset(Dataset):

    def __init__(self, root, split: str, subset: str='train', task_lookup: Optional[Dict[str, int]]=None, augment: bool=False, num_color_permute: int=9, seed: int=42, task_filter: Optional[List[str]]=None, exclude_tasks: Optional[List[str]]=None, gamma: float=0.95, num_actions: int=NUM_ACTIONS) -> None:
        assert subset in {'train', 'val'}
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.num_color_permute = num_color_permute
        self.gamma = gamma
        self.num_actions = num_actions
        self.task_lookup: Dict[str, int] = dict(task_lookup) if task_lookup is not None else {}
        self.trajectories: Dict[str, List[Step]] = {}
        self.value_targets: Dict[str, List[float]] = {}
        self.aug_configs: Dict[str, List[Tuple]] = {}
        self.samples: List[Tuple[str, int, int]] = []
        self._traj_to_task: Dict[str, str] = {}
        split_dir = self.root / 'data' / split
        task_dirs = sorted((d for d in split_dir.iterdir() if d.is_dir()))
        if task_filter is not None:
            filter_set = set(task_filter)
            filtered = [d for d in task_dirs if d.name in filter_set]
            missing = filter_set - {d.name for d in filtered}
            if missing:
                avail = sorted((d.name for d in task_dirs))
                raise FileNotFoundError(f'task_filter={sorted(filter_set)} not all present under {split_dir}: missing={sorted(missing)}. available={avail}')
            task_dirs = filtered
            print(f'[AA3Dataset] task_filter applied: only loading {sorted((d.name for d in task_dirs))} (others under {split_dir} ignored)')
        if exclude_tasks:
            excl = set(exclude_tasks)
            before = {d.name for d in task_dirs}
            task_dirs = [d for d in task_dirs if d.name not in excl]
            print(f'[AA3Dataset] exclude_tasks removed={sorted(before & excl)} ignored(not present)={sorted(excl - before)}')
        for task_dir in task_dirs:
            task_name = task_dir.name
            if task_lookup is None:
                if task_name not in self.task_lookup:
                    self.task_lookup[task_name] = len(self.task_lookup)
            elif task_name not in self.task_lookup:
                continue
            jsonl_files = sorted(task_dir.glob('*.recording.jsonl'))
            if not jsonl_files:
                continue
            if len(jsonl_files) > 1:
                task_rng = random.Random(f'{seed}:{task_name}')
                val_file = task_rng.choice(jsonl_files)
                if subset == 'val':
                    selected = [val_file]
                else:
                    selected = [f for f in jsonl_files if f != val_file]
            else:
                selected = jsonl_files
            for jsonl_path in selected:
                traj_id = f'{task_name}/{jsonl_path.stem}'
                steps = parse_recording(jsonl_path)
                if not steps:
                    continue
                self.trajectories[traj_id] = steps
                self.value_targets[traj_id] = value_targets_from_steps(steps, gamma)
                self._traj_to_task[traj_id] = task_name
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
                for aug_idx in range(len(configs)):
                    for step_idx in range(len(steps)):
                        self.samples.append((traj_id, step_idx, aug_idx))
        if not self.samples:
            raise RuntimeError(f"No AA3 samples found under {split_dir} for subset='{subset}'.")
        self.num_tasks = len(self.task_lookup)
        nonzero = sum((1 for ys in self.value_targets.values() for v in ys if v > 0.0))
        total = sum((len(ys) for ys in self.value_targets.values()))
        print(f'[AA3Dataset] {split}/{subset}: {len(self.samples)} samples, {self.num_tasks} tasks; gamma={gamma} nonzero value targets: {nonzero}/{total} ({nonzero / max(total, 1):.3f})')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        traj_id, step_idx, aug_idx = self.samples[idx]
        step = self.trajectories[traj_id][step_idx]
        transform_name, color_map = self.aug_configs[traj_id][aug_idx]
        obs_t = step.obs_t.copy()
        obs_t1 = step.obs_t1.copy()
        action_id, action_x, action_y = (step.action_id, step.action_x, step.action_y)
        available = step.available
        if transform_name is not None:
            obs_t = apply_geometric_transform(obs_t, transform_name)
            obs_t1 = apply_geometric_transform(obs_t1, transform_name)
            remap = ACTION_REMAP.get(transform_name, {})
            action_id = remap.get(action_id, action_id)
            if action_id == 6 and action_x >= 0:
                action_x, action_y = transform_action6_coord(action_x, action_y, transform_name)
            available = [remap.get(a, a) for a in available]
        if color_map is not None:
            obs_t = color_map[obs_t]
            obs_t1 = color_map[obs_t1]
        task_name = self._traj_to_task[traj_id]
        return {'inputs': torch.tensor(obs_t, dtype=torch.long), 'action_id': torch.tensor(action_id, dtype=torch.long), 'action_x': torch.tensor(action_x, dtype=torch.long), 'action_y': torch.tensor(action_y, dtype=torch.long), 'targets': torch.tensor(obs_t1, dtype=torch.long), 'task_id': torch.tensor(self.task_lookup[task_name], dtype=torch.long), 'available_mask': available_list_to_mask(available, self.num_actions), 'levels_completed': torch.tensor(step.levels, dtype=torch.long), 'value_target': torch.tensor(self.value_targets[traj_id][step_idx], dtype=torch.float32), 'task_name': task_name, 'trajectory_id': traj_id, 'step_index': step_idx}

def collate_fn(batch: List[Dict]) -> Dict:
    return {'inputs': torch.stack([b['inputs'] for b in batch]), 'action_id': torch.stack([b['action_id'] for b in batch]), 'action_x': torch.stack([b['action_x'] for b in batch]), 'action_y': torch.stack([b['action_y'] for b in batch]), 'targets': torch.stack([b['targets'] for b in batch]), 'task_ids': torch.stack([b['task_id'] for b in batch]), 'available_mask': torch.stack([b['available_mask'] for b in batch]), 'levels': torch.stack([b['levels_completed'] for b in batch]), 'value_target': torch.stack([b['value_target'] for b in batch]), 'task_names': [b['task_name'] for b in batch]}

def build_dataloaders(*, data_root: str, train_split: str='training', eval_split: Optional[str]='training', batch_size: int, num_workers: int=8, augment: bool=True, num_color_permute: int=9, task_filter: Optional[List[str]]=None, exclude_tasks: Optional[List[str]]=None, gamma: float=0.95, drop_last: bool=True, distributed: bool=False, rank: int=0, world_size: int=1):
    train_dataset = AA3Dataset(data_root, train_split, subset='train', augment=augment, num_color_permute=num_color_permute, task_filter=task_filter, exclude_tasks=exclude_tasks, gamma=gamma)
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    eval_dataset = eval_loader = None
    if eval_split:
        eval_dataset = AA3Dataset(data_root, eval_split, subset='val', task_lookup=train_dataset.task_lookup, augment=False, task_filter=task_filter, exclude_tasks=exclude_tasks, gamma=gamma)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    return (train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, None)

def build_eval_loader(data_root: str, eval_splits: List[str], game: str, *, batch_size: int, num_workers: int, max_level: Optional[int]=None) -> DataLoader:
    parts = []
    n_total = n_kept = 0
    used_splits = []
    for split in eval_splits:
        for subset in ('train', 'val'):
            try:
                ds = AA3Dataset(data_root, split, subset=subset, augment=False, task_filter=[game])
            except (FileNotFoundError, RuntimeError):
                continue
            if len(ds) == 0:
                continue
            used_splits.append(split)
            n_total += len(ds)
            if max_level is None:
                parts.append(ds)
                n_kept += len(ds)
            else:
                idxs = [i for i, (tid, sidx, _a) in enumerate(ds.samples) if ds.trajectories[tid][sidx].levels <= max_level]
                if idxs:
                    parts.append(Subset(ds, idxs))
                    n_kept += len(idxs)
    if not parts:
        raise FileNotFoundError(f'no human-replay data for game={game} in splits={eval_splits} (max_level={max_level})')
    split_note = sorted(set(used_splits))
    if max_level is not None:
        print(f'[{game}] eval split(s)={split_note} level filter <= {max_level}: kept {n_kept}/{n_total} samples')
    else:
        print(f'[{game}] eval split(s)={split_note}: {n_kept} samples')
    ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
