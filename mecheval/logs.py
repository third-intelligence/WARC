"""Game play logs that MechEval reads.

- An ARC-AGI-3 recording (.recording.jsonl, one observation per line, the format of the human recordings): the first
  line is a RESET; each later line holds the action and the frames that followed it. The replay checks every frame.
- An action list (.json): {"game_id": "m0r0-492f87ba", "actions": [[id, x, y], ...]} with id 0 for RESET and 1-7 for
  ACTION1-7 (x and y only for ACTION6, -1 otherwise). The game starts from a RESET; the frames are produced by the
  replay itself, so nothing checks them against what the player saw.

The game version is part of the game identifier ("m0r0-492f87ba"); MechEval supports the versions of the paper's
recordings (data/versions.json).
"""
import json
from pathlib import Path

from .engine.replay_io import action_id
from .paths import DATA

GAMES = ('ar25', 'cd82', 'dc22', 'ft09', 'ka59', 'lp85', 'ls20', 'm0r0', 'r11l', 're86', 's5i5', 'sb26', 'sc25',
         'sk48', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30')


def versions():
    return json.loads((DATA/'versions.json').read_text())


def inspect(path):
    """('recording' or 'actions', game identifier or None)."""
    path = Path(path)
    if path.name.endswith('.jsonl'):
        with path.open() as f:
            first = json.loads(f.readline())
        return 'recording', first.get('data', {}).get('game_id')
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or 'actions' not in data:
        raise ValueError(f'{path}: expected a .recording.jsonl or a JSON object with "actions"')
    return 'actions', data.get('game_id')


def game_of(game_id):
    if not game_id:
        raise ValueError('the log does not name its game; pass --game')
    return game_id.split('-')[0]


def check_supported(game, game_id):
    if game not in GAMES:
        raise ValueError(f'{game} is not one of the 20 MechEval games: {", ".join(GAMES)}')
    supported = versions()[game]
    if game_id not in supported:
        raise ValueError(f'game version {game_id!r} is not supported; MechEval annotates {", ".join(supported)}')


def actions_to_recording(path, out, profile):
    """Write an action list as a recording, with the frames, state and level count of the game replayed under the
    given engine profile."""
    from arcengine import ActionInput, GameAction
    from .engine.arc_adapter import load_game
    data = json.loads(Path(path).read_text())
    game_id = data['game_id']
    game = load_game(game_of(game_id), environment_profile=profile)

    def line(response, aid, x, y):
        payload = {'game_id': game_id}
        if aid == 6:
            payload.update(x=int(x), y=int(y))
        return json.dumps({'data': {'game_id': game_id, 'frame': [f.tolist() if hasattr(f, 'tolist') else f
                                                                  for f in response.frame],
                                    'state': str(getattr(response.state, 'value', response.state)),
                                    'levels_completed': int(response.levels_completed),
                                    'action_input': {'id': int(aid), 'data': payload}}})
    with Path(out).open('w') as f:
        f.write(line(game.perform_action(ActionInput(id=GameAction.RESET), raw=True), 0, -1, -1) + '\n')
        for a in data['actions']:
            aid, x, y = (list(a) + [-1, -1])[:3]
            aid = action_id(aid)
            act = ActionInput(id=GameAction.RESET if aid == 0 else getattr(GameAction, f'ACTION{aid}'),
                              data={'x': int(x), 'y': int(y)} if aid == 6 else {})
            f.write(line(game.perform_action(act, raw=True), aid, x, y) + '\n')
    return Path(out)
