import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
IGNORE_INDEX_AA3 = 16
NUM_COLORS_AA3 = 16
ACTION_STR_TO_INT = {'RESET': 0, 'ACTION1': 1, 'ACTION2': 2, 'ACTION3': 3, 'ACTION4': 4, 'ACTION5': 5, 'ACTION6': 6, 'ACTION7': 7}
ACTION_REMAP = {'rotate90_cw': {1: 4, 4: 2, 2: 3, 3: 1}, 'rotate90_ccw': {1: 3, 3: 2, 2: 4, 4: 1}, 'rotate180': {1: 2, 2: 1, 3: 4, 4: 3}, 'flip_vertical': {1: 2, 2: 1}, 'flip_horizontal': {3: 4, 4: 3}}
GEOMETRIC_TRANSFORMS = ['rotate90_cw', 'rotate90_ccw', 'rotate180', 'flip_vertical', 'flip_horizontal']

def normalize_action_id(raw_id) -> int:
    if isinstance(raw_id, int):
        return raw_id
    return ACTION_STR_TO_INT.get(str(raw_id), 0)

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

def parse_jsonl_trajectory(filepath: Path) -> List[Tuple]:
    lines = []
    with filepath.open('r') as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    pairs = []
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
        pairs.append((obs_t, action_id, action_x, action_y, obs_t1))
    return pairs

class ARCAA3Dataset(Dataset):

    def __init__(self, root: Path, split: str, subset: str='train', task_lookup: Optional[Dict[str, int]]=None, augment: bool=False, num_color_permute: int=9, seed: int=42, task_filter: Optional[List[str]]=None) -> None:
        assert subset in {'train', 'val'}
        self.root = Path(root)
        self.augment = augment
        self.num_color_permute = num_color_permute
        self.task_lookup: Dict[str, int] = dict(task_lookup) if task_lookup is not None else {}
        self.trajectories: Dict[str, List[Tuple]] = {}
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
                raise FileNotFoundError(f'task_filter={sorted(filter_set)}: some tasks are missing under {split_dir}: missing={sorted(missing)}. available={avail}')
            task_dirs = filtered
            print(f'[ARCAA3Dataset] task_filter applied: only loading {sorted((d.name for d in task_dirs))} (others under {split_dir} ignored)')
        print('Start loading AA3 data...')
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
                traj_id = jsonl_path.stem
                pairs = parse_jsonl_trajectory(jsonl_path)
                if not pairs:
                    continue
                self.trajectories[traj_id] = pairs
                self._traj_to_task[traj_id] = task_name
                configs: List[Tuple] = [(None, None)]
                if augment:
                    for transform_name in GEOMETRIC_TRANSFORMS:
                        configs.append((transform_name, None))
                        traj_seed = hash((traj_id, transform_name)) & 4294967295
                        traj_rng = np.random.RandomState(traj_seed)
                        for _ in range(num_color_permute):
                            perm = traj_rng.permutation(NUM_COLORS_AA3).astype(np.uint8)
                            configs.append((transform_name, perm))
                self.aug_configs[traj_id] = configs
                n_steps = len(pairs)
                n_configs = len(configs)
                for aug_idx in range(n_configs):
                    for step_idx in range(n_steps):
                        self.samples.append((traj_id, step_idx, aug_idx))
        if not self.samples:
            raise RuntimeError(f"No AA3 samples found under {split_dir} for subset='{subset}'.")
        self.num_tasks = len(self.task_lookup)
        print(f'Finish loading AA3! {len(self.samples)} samples, {self.num_tasks} tasks.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        traj_id, step_idx, aug_idx = self.samples[idx]
        obs_t, action_id, action_x, action_y, obs_t1 = self.trajectories[traj_id][step_idx]
        transform_name, color_map = self.aug_configs[traj_id][aug_idx]
        obs_t = obs_t.copy()
        obs_t1 = obs_t1.copy()
        if transform_name is not None:
            obs_t = apply_geometric_transform(obs_t, transform_name)
            obs_t1 = apply_geometric_transform(obs_t1, transform_name)
            remap = ACTION_REMAP.get(transform_name, {})
            action_id = remap.get(action_id, action_id)
            if action_id == 6 and action_x >= 0:
                action_x, action_y = transform_action6_coord(action_x, action_y, transform_name)
        if color_map is not None:
            obs_t = color_map[obs_t]
            obs_t1 = color_map[obs_t1]
        task_name = self._traj_to_task[traj_id]
        task_id = self.task_lookup[task_name]
        return {'inputs': torch.tensor(obs_t, dtype=torch.long), 'attention_mask': torch.ones(64, 64, dtype=torch.long), 'action_id': torch.tensor(action_id, dtype=torch.long), 'action_x': torch.tensor(action_x, dtype=torch.long), 'action_y': torch.tensor(action_y, dtype=torch.long), 'targets': torch.tensor(obs_t1, dtype=torch.long), 'task_id': torch.tensor(task_id, dtype=torch.long), 'task_name': task_name, 'trajectory_id': traj_id, 'step_index': step_idx}

    def disable_translation(self):
        pass

    def enable_translation(self):
        pass

    def disable_resolution_augmentation(self, fix_scale_factor=1):
        pass

    def enable_resolution_augmentation(self):
        pass

def collate_fn_aa3(batch: List[Dict]) -> Dict:
    return {'inputs': torch.stack([b['inputs'] for b in batch]), 'attention_mask': torch.stack([b['attention_mask'] for b in batch]), 'action_id': torch.stack([b['action_id'] for b in batch]), 'action_x': torch.stack([b['action_x'] for b in batch]), 'action_y': torch.stack([b['action_y'] for b in batch]), 'targets': torch.stack([b['targets'] for b in batch]), 'task_ids': torch.stack([b['task_id'] for b in batch]), 'task_names': [b['task_name'] for b in batch]}

def build_dataloaders_aa3(args, *, distributed: bool=False, rank: int=0, world_size: int=1):
    root = Path(args.data_root)
    train_split = getattr(args, 'train_split', 'training')
    augment = not getattr(args, 'disable_augment_aa3', False)
    ttt_max_steps = getattr(args, 'ttt_max_steps', None)
    aa3_task = getattr(args, 'aa3_task', None)
    task_filter = [aa3_task] if aa3_task else None
    train_dataset = ARCAA3Dataset(root, train_split, subset='train', augment=augment, task_filter=task_filter)
    if ttt_max_steps is not None:
        raw_order: dict = {}
        for traj_id, step_idx, _aug_idx in train_dataset.samples:
            key = (traj_id, step_idx)
            if key not in raw_order:
                raw_order[key] = len(raw_order)
        allowed = {k for k, v in raw_order.items() if v < ttt_max_steps}
        original_samples = len(train_dataset.samples)
        train_dataset.samples = [s for s in train_dataset.samples if (s[0], s[1]) in allowed]
        print(f'[ttt_max_steps={ttt_max_steps}] raw steps: {min(ttt_max_steps, len(raw_order))}, total samples (with aug): {original_samples} → {len(train_dataset.samples)}')
    use_shuffle = True
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=use_shuffle)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None and use_shuffle, sampler=train_sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn_aa3, pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None)
    eval_dataset = None
    eval_loader = None
    eval_sampler = None
    eval_split = getattr(args, 'eval_split', None)
    if eval_split:
        eval_dataset = ARCAA3Dataset(root, eval_split, subset='val', task_lookup=train_dataset.task_lookup, augment=False, task_filter=task_filter)
        if distributed:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, sampler=eval_sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn_aa3, pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None)
    return (train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler)
