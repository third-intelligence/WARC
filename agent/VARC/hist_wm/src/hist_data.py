from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_WM_CORE_SRC = _PROJ_ROOT / 'wm_core' / 'src'
for _p in (str(_PROJ_ROOT), str(_WM_CORE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np
import torch
from torch.utils.data import DataLoader
from varc2.constants import ACTION_REMAP
from varc2.data import Step, apply_geometric_transform, collate_fn, transform_action6_coord
from wm_core.src.data import FoldSplit, TrajSetDataset, build_fold_split
PAD_ID = 16
PAD_ACTION = (0, -1, -1)

def history_depths(steps: List[Step], n: int) -> List[int]:
    depths = [1] * len(steps)
    for i in range(1, len(steps)):
        if np.array_equal(steps[i - 1].obs_t1, steps[i].obs_t):
            depths[i] = min(n, depths[i - 1] + 1)
        else:
            depths[i] = 1
    return depths

def transform_action(aid: int, x: int, y: int, transform_name: Optional[str]) -> Tuple[int, int, int]:
    if transform_name is None:
        return (aid, x, y)
    remap = ACTION_REMAP.get(transform_name, {})
    a = remap.get(aid, aid)
    if a == 6 and x >= 0:
        x, y = transform_action6_coord(x, y, transform_name)
    return (a, x, y)

def build_window(steps: List[Step], i: int, m: int, n: int, transform_name: Optional[str], color_map: Optional[np.ndarray]):
    hist_obs = np.full((n, 64, 64), PAD_ID, dtype=np.uint8)
    for k in range(m - 1, -1, -1):
        f = steps[i - k].obs_t
        if transform_name is not None:
            f = apply_geometric_transform(f, transform_name)
        if color_map is not None:
            f = color_map[f]
        hist_obs[n - 1 - k] = f
    hist_aid = np.full(max(n - 1, 0), PAD_ACTION[0], dtype=np.int64)
    hist_ax = np.full(max(n - 1, 0), PAD_ACTION[1], dtype=np.int64)
    hist_ay = np.full(max(n - 1, 0), PAD_ACTION[2], dtype=np.int64)
    for k in range(m - 1, 0, -1):
        s = steps[i - k]
        a, x, y = transform_action(s.action_id, s.action_x, s.action_y, transform_name)
        hist_aid[n - 1 - k] = a
        hist_ax[n - 1 - k] = x
        hist_ay[n - 1 - k] = y
    return (hist_obs, hist_aid, hist_ax, hist_ay)

class HistTrajSetDataset(TrajSetDataset):

    def __init__(self, *args, hist_n: int=2, **kwargs) -> None:
        kwargs.setdefault('horizon', 1)
        if kwargs['horizon'] != 1:
            raise ValueError('hist_wm (policy_hist) supports horizon=1 only')
        if hist_n < 1:
            raise ValueError(f'hist_n must be >= 1, got {hist_n}')
        super().__init__(*args, **kwargs)
        self.hist_n = hist_n
        self._depths: Dict[str, List[int]] = {tid: history_depths(steps, hist_n) for tid, steps in self.trajectories.items()}

    def __getitem__(self, idx: int) -> Dict:
        base = super().__getitem__(idx)
        traj_id, i, aug_idx = self.samples[idx]
        steps = self.trajectories[traj_id]
        m = self._depths[traj_id][i]
        transform_name, color_map = self.aug_configs[traj_id][aug_idx]
        hist_obs, hist_aid, hist_ax, hist_ay = build_window(steps, i, m, self.hist_n, transform_name, color_map)
        base['hist_obs'] = torch.from_numpy(hist_obs)
        base['hist_action_id'] = torch.from_numpy(hist_aid)
        base['hist_action_x'] = torch.from_numpy(hist_ax)
        base['hist_action_y'] = torch.from_numpy(hist_ay)
        return base

def hist_collate_fn(batch: List[Dict]) -> Dict:
    out = collate_fn(batch)
    out['hist_obs'] = torch.stack([b['hist_obs'] for b in batch])
    out['hist_action_id'] = torch.stack([b['hist_action_id'] for b in batch])
    out['hist_action_x'] = torch.stack([b['hist_action_x'] for b in batch])
    out['hist_action_y'] = torch.stack([b['hist_action_y'] for b in batch])
    return out

def _subset(split: FoldSplit, trajs: Dict[str, List[str]]) -> Dict[str, List[Step]]:
    return {t: split.steps_by_traj[t] for ts in trajs.values() for t in ts}

def build_fold_loaders_hist(split: FoldSplit, *, hist_n: int, batch_size: int, eval_batch_size: Optional[int]=None, num_workers: int=4, augment: bool=True, num_color_permute: int=9, gamma: float=0.95, distributed: bool=False, rank: int=0, world_size: int=1):
    t2t = split.traj_to_task()
    train_ds = HistTrajSetDataset(_subset(split, split.train_trajs), t2t, split.task_lookup, augment=augment, num_color_permute=num_color_permute, gamma=gamma, hist_n=hist_n)
    val_ds = HistTrajSetDataset(_subset(split, split.val_trajs), t2t, split.task_lookup, augment=False, gamma=gamma, hist_n=hist_n)
    test_ds = HistTrajSetDataset(_subset(split, split.test_trajs), t2t, split.task_lookup, augment=False, gamma=gamma, hist_n=hist_n)
    if len(train_ds) == 0:
        raise RuntimeError('empty train dataset (all trajectories held out?)')
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, drop_last=False, num_workers=num_workers, collate_fn=hist_collate_fn, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    ebs = eval_batch_size or batch_size

    def _eval_loader(ds):
        return DataLoader(ds, batch_size=ebs, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=hist_collate_fn, pin_memory=True)
    return (train_ds, train_loader, train_sampler, val_ds, _eval_loader(val_ds), test_ds, _eval_loader(test_ds))

def steps_to_window_items(steps: List[Step], n: int) -> List[tuple]:
    depths = history_depths(steps, n)
    items = []
    for i, s in enumerate(steps):
        m = depths[i]
        frames = [steps[i - k].obs_t for k in range(m - 1, -1, -1)]
        inter = [(steps[i - k].action_id, steps[i - k].action_x, steps[i - k].action_y) for k in range(m - 1, 0, -1)]
        items.append((frames, inter, (s.action_id, s.action_x, s.action_y), s.obs_t1, int(s.levels)))
    return items

def augment_window_item(item: tuple, geo: Optional[str], color_map: Optional[np.ndarray]) -> tuple:
    frames, inter, act, nxt, lvl = item
    fs = frames
    if geo is not None:
        fs = [apply_geometric_transform(f, geo) for f in fs]
        nxt = apply_geometric_transform(nxt, geo)
    if color_map is not None:
        fs = [color_map[f] for f in fs]
        nxt = color_map[nxt]
    inter = [transform_action(a, x, y, geo) for a, x, y in inter]
    act = transform_action(act[0], act[1], act[2], geo)
    return (fs, inter, act, nxt, lvl)

def expand_window_items(items: List[tuple], *, num_color_permute: int, rng, geometric_transforms) -> List[tuple]:
    geos = list(geometric_transforms) if geometric_transforms is not None else None
    if geos is None:
        from agent_base.src.augment import GEOMETRIC_TRANSFORMS as _FULL
        geos = list(_FULL)
    out: List[tuple] = []
    for item in items:
        out.append(augment_window_item(item, None, None))
        for geo in geos:
            out.append(augment_window_item(item, geo, None))
            for _ in range(num_color_permute):
                perm = np.random.RandomState(rng.randrange(2 ** 31)).permutation(16).astype(np.uint8)
                out.append(augment_window_item(item, geo, perm))
    return out

def stack_batch_hist(items: List[tuple], n: int, device):
    B = len(items)
    hist_obs = np.full((B, n, 64, 64), PAD_ID, dtype=np.uint8)
    hist_aid = np.full((B, max(n - 1, 0)), PAD_ACTION[0], dtype=np.int64)
    hist_ax = np.full((B, max(n - 1, 0)), PAD_ACTION[1], dtype=np.int64)
    hist_ay = np.full((B, max(n - 1, 0)), PAD_ACTION[2], dtype=np.int64)
    aid = np.zeros(B, dtype=np.int64)
    ax = np.zeros(B, dtype=np.int64)
    ay = np.zeros(B, dtype=np.int64)
    nxts = []
    for b, (frames, inter, act, nxt, _lvl) in enumerate(items):
        m = len(frames)
        hist_obs[b, n - m:] = np.stack(frames)
        for j, (a, x, y) in enumerate(inter):
            slot = n - 1 - len(inter) + j
            hist_aid[b, slot] = a
            hist_ax[b, slot] = x
            hist_ay[b, slot] = y
        aid[b], ax[b], ay[b] = act
        nxts.append(nxt)
    obs = torch.from_numpy(hist_obs[:, -1].astype(np.int64)).to(device)
    return (obs, torch.from_numpy(aid).to(device), torch.from_numpy(ax).to(device), torch.from_numpy(ay).to(device), torch.from_numpy(np.stack(nxts)).long().to(device), torch.from_numpy(hist_obs).to(device), torch.from_numpy(hist_aid).to(device), torch.from_numpy(hist_ax).to(device), torch.from_numpy(hist_ay).to(device))
