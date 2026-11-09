from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
CLICK_ACTION_ID = 6
_TERMINAL_STATES = {'WIN', 'GAME_OVER'}

@dataclass
class Action:
    action_id: int
    x: int = -1
    y: int = -1

    def is_click(self) -> bool:
        return self.action_id == CLICK_ACTION_ID

class MiniArcEnv:

    def __init__(self, game_id: str, *, image_size: int=64, max_episode_steps: int=1000, api_key: str='', operation_mode: str='normal', environments_dir: str=''):
        import arc_agi
        mode_map = {'normal': arc_agi.OperationMode.NORMAL, 'online': arc_agi.OperationMode.ONLINE, 'offline': arc_agi.OperationMode.OFFLINE, 'competition': arc_agi.OperationMode.COMPETITION}
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        kw = dict(arc_api_key=api_key or os.environ.get('ARC_API_KEY', ''), operation_mode=mode_map.get(operation_mode, arc_agi.OperationMode.NORMAL))
        if environments_dir:
            kw['environments_dir'] = str(environments_dir)
        self._arcade = arc_agi.Arcade(**kw)
        scorecard_id = self._arcade.open_scorecard() if hasattr(self._arcade, 'open_scorecard') else None
        self._env = self._arcade.make(game_id, scorecard_id=scorecard_id)
        if self._env is None:
            raise RuntimeError(f'failed to make arc_agi env for game_id={game_id!r}')
        self._step_count = 0
        self._last_levels = 0

    def reset(self) -> Tuple[np.ndarray, dict]:
        resp = self._env.reset()
        self._step_count = 0
        return self._parse(resp, full_reset=True)

    def step(self, action: Action) -> Tuple[np.ndarray, dict]:
        from arcengine import GameAction
        gid = GameAction.from_id(action.action_id)
        data = {'x': int(action.x), 'y': int(action.y)} if action.action_id == CLICK_ACTION_ID else None
        resp = self._env.step(gid, data=data)
        self._step_count += 1
        return self._parse(resp, full_reset=False)

    def _parse(self, resp, *, full_reset: bool) -> Tuple[np.ndarray, dict]:
        if resp is None or not getattr(resp, 'frame', None):
            obs = np.zeros((self.image_size, self.image_size), dtype=np.int64)
            return (obs, {'state': 'GAME_OVER', 'levels_completed': self._last_levels, 'win_levels': 0, 'available_actions': [], 'done': True, 'truncated': False, 'full_reset': full_reset, 'guid': None})
        last = resp.frame[-1]
        grid = last.numpy() if hasattr(last, 'numpy') else np.asarray(last)
        if grid.ndim == 3 and grid.shape[0] == 1:
            grid = grid[0]
        if grid.shape != (self.image_size, self.image_size):
            tmp = np.zeros((self.image_size, self.image_size), dtype=np.int64)
            h = min(grid.shape[0], self.image_size)
            w = min(grid.shape[1], self.image_size)
            tmp[:h, :w] = grid[:h, :w]
            grid = tmp
        else:
            grid = grid.astype(np.int64)
        state = resp.state.name if hasattr(resp.state, 'name') else str(resp.state)
        terminal = state in _TERMINAL_STATES
        truncated = self._step_count >= self.max_episode_steps
        self._last_levels = int(resp.levels_completed)
        return (grid, {'state': state, 'levels_completed': int(resp.levels_completed), 'win_levels': int(resp.win_levels), 'available_actions': list(resp.available_actions or []), 'guid': resp.guid, 'done': bool(terminal or truncated), 'truncated': bool(truncated), 'full_reset': full_reset})
