"""Replay a whole level 1 and 2 segment of a recording to confirm a reconstructed initial state before it is used."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np

from .arc_adapter import ROOT, load_game
from .endpoint_scope import compare_frames
from .environment_compat import metadata_diff, PROFILES
from .replay_io import action_id

HERE = Path(__file__).resolve().parent


def validate(game_id, path, profile, reconstruct_initial=False):
    from arcengine import ActionInput, GameAction
    random.seed(0)
    np.random.seed(0)
    game = load_game(game_id, environment_profile=profile)
    result = {'game': game_id, 'recording': str(path), 'replay_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'environment': game._evaluation_environment, 'observations': 0, 'final_mismatch_steps': [],
              'metadata_mismatch_steps': [], 'intermediate_mismatch_steps': [], 'boundaries': [],
              'first_mismatch': None, 'stop_reason': 'recording_end',
              'initialization': 'recorded_reset', 'empty_terminal_observations': []}
    previous = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)['data']
            if 'action_input' not in row or 'frame' not in row:
                continue
            step = result['observations']
            ai = row['action_input']
            aid = action_id(ai['id'])
            if step == 0 and aid != 0:
                if not reconstruct_initial or row.get('levels_completed') != 0:
                    raise ValueError('Missing initial RESET; initial state is unknown')
                game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                result['initialization'] = 'reconstructed_standard_reset_candidate'
            response = game.perform_action(ActionInput(
                id=GameAction.RESET if aid == 0 else getattr(GameAction, f'ACTION{aid}'),
                data=ai.get('data') or {}), raw=True)
            comparison = compare_frames(response.frame, row['frame'])
            metadata = metadata_diff(response, row)
            empty_terminal = (not response.frame and not row['frame'] and
                              str(getattr(response.state, 'value', response.state)) in ('GAME_OVER', 'WIN') and not metadata)
            if empty_terminal:
                result['empty_terminal_observations'].append(step)
                # No visual target exists. State/action agreement, not pixel accuracy.
                comparison = {'final_equal': True, 'all_subframes_equal': True}
            result['observations'] += 1
            if not comparison['final_equal']:
                result['final_mismatch_steps'].append(step)
            if not comparison['all_subframes_equal']:
                result['intermediate_mismatch_steps'].append(step)
            if metadata:
                result['metadata_mismatch_steps'].append(step)
            if (metadata or not comparison['final_equal']) and result['first_mismatch'] is None:
                result['first_mismatch'] = {
                    'step': step, 'metadata': metadata, 'actual_level': game.level_index + 1,
                    'different_pixels': int(np.count_nonzero(np.asarray(response.frame[-1]) != np.asarray(row['frame'][-1]))) if response.frame and row['frame'] else None,
                    'actual': np.asarray(response.frame[-1]).astype(int).tolist() if response.frame else None,
                    'recorded': row['frame'][-1] if row['frame'] else None,
                }
            levels = (int(response.levels_completed), row.get('levels_completed'))
            if previous is not None and levels != previous:
                result['boundaries'].append({'step': step, 'actual_completed': levels[0],
                                             'recorded_completed': levels[1], **comparison})
            previous = levels
            # Compare the exit frame BEFORE stopping; do not hide m0r0-like differences.
            if game.level_index >= 2 or (row.get('levels_completed', 0) >= 2):
                result['stop_reason'] = 'L2_exit_compared'
                break
    result['status'] = 'matched' if not result['final_mismatch_steps'] and not result['metadata_mismatch_steps'] else 'mismatch'
    return result
