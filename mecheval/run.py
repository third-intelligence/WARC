"""Run a world model on the recordings under the protocol of MechEval and write its predictions.

    python -m mecheval run --model mecheval.models:CopyLast --recordings <dir> --out predictions/copy_last
    python -m mecheval run --model mypackage.mymodule:MyModel --arg checkpoint=... --recordings <dir> --out <dir>
    python -m mecheval run --model mecheval.models:CopyLast --labels <labels dir> --plays <plays> --out <dir>

The protocol (Sec. 4 of the paper). Every recording is replayed from its first frame with the recorded actions, up to
the completion of level 2. At every transition except a RESET, the model first predicts the next frame from what it
has seen and the action, and only then receives the true next frame. Every recording starts from a new model, so
nothing carries over from one recording to another. This runner enforces that order; it writes one
<recording>.npz per recording for `python -m mecheval score`.

A model is a class with these methods (see models.py for a complete example):

    class MyModel:
        def __init__(self, game, **kwargs): ...          # once per recording; game is a code such as "ls20"
        def predict(self, history, action): ...          # the next frame (see below)
        def update(self, history, action, next_frame, valid): ...   # the true next frame; learn from it or not

history is a History: history.frames holds every frame of the recording so far (the last one is the current frame),
history.actions[i] is the action that led from frames[i] to frames[i + 1], and history.segment_start is the index of
the frame where the current segment began. A segment restarts after a RESET, a level change and a game over; the
paper's models read only the current segment. An action is (id, x, y): id 1 to 7 for ACTION1 to ACTION7, x and y the
clicked cell for ACTION6 (column, row) and -1 otherwise. RESET (id 0) is never predicted.

predict returns either the whole next frame, a 64 x 64 array of colour indices 0 to 15, or only the cells that change,
{"changes": [[x, y, colour], ...]}, which the runner applies to the current frame. valid is False for a transition that
completes a level or ends the game; the paper's models learn only from valid transitions.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .data import VERSION, add_set_arguments, locate, selected


@dataclass
class History:
    frames: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    segment_start: int = 0

    @property
    def current(self):
        return self.frames[-1]

    @property
    def segment(self):
        """(frames, actions) of the current segment."""
        return self.frames[self.segment_start:], self.actions[self.segment_start:]


def _frame(line):
    frames = line['data']['frame']
    if not frames:
        return None
    frame = np.asarray(frames[-1])
    if frame.shape != (64, 64) or frame.min() < 0 or frame.max() > 15:
        raise ValueError('expected a 64 x 64 frame of colour indices 0-15')
    return frame.astype(np.uint8)


def _action(action_input):
    value = action_input['id']
    if isinstance(value, str):
        value = 0 if value == 'RESET' else int(value.removeprefix('ACTION'))
    value = int(value)
    if value == 6:
        data = action_input.get('data') or {}
        return (6, int(data['x']), int(data['y']))
    return (value, -1, -1)


def transitions(path):
    """(step, action, before, after, valid) for every recorded action up to the completion of level 2. step is the
    line of the recording; line 0 is the first frame."""
    lines = [json.loads(x) for x in Path(path).open() if x.strip()]
    before = _frame(lines[0])
    for i in range(1, len(lines)):
        if 'frame' not in (lines[i].get('data') or {}):        # a trailing summary line
            break
        after = _frame(lines[i])
        if after is None:
            break
        prev, cur = lines[i - 1]['data'], lines[i]['data']
        action = _action(cur['action_input'])
        valid = (action[0] != 0 and cur['state'] == 'NOT_FINISHED'
                 and cur['levels_completed'] == prev['levels_completed'])
        yield i, action, before, after, valid
        before = after
        if cur['levels_completed'] >= 2:
            break


def _as_frame(pred, current):
    if isinstance(pred, dict):
        frame = current.copy()
        for x, y, colour in pred.get('changes', []):
            frame[int(y), int(x)] = int(colour)
        return frame
    frame = np.asarray(pred)
    if frame.shape != (64, 64):
        raise ValueError(f'a prediction must be 64 x 64, got {frame.shape}')
    if frame.min() < 0 or frame.max() > 15:
        raise ValueError('a prediction holds colour indices 0-15')
    return frame.astype(np.uint8)


def run_recording(model_class, kwargs, game, path):
    """Predictions for one recording, as arrays for the .npz file, and a summary."""
    model = model_class(game, **kwargs)
    history = None
    rows = []
    started = time.perf_counter()
    for step, action, before, after, valid in transitions(path):
        if history is None:
            history = History(frames=[before])
        if not np.array_equal(history.current, before):
            raise RuntimeError(f'{path}: the history does not end in the frame before action {step}')
        if action[0] != 0:
            pred = _as_frame(model.predict(history, action), history.current)
            model.update(history, action, after, valid)
            rows.append((step, pred, before, after))
        history.frames.append(after)
        history.actions.append(action)
        if not valid:
            history.segment_start = len(history.frames) - 1
    digest = lambda f: np.frombuffer(hashlib.sha1(f.tobytes()).digest(), np.uint8)
    arrays = dict(action_index=np.array([r[0] for r in rows], np.int32),
                  pred=np.stack([r[1] for r in rows]) if rows else np.zeros((0, 64, 64), np.uint8),
                  before_sha1=np.stack([digest(r[2]) for r in rows]) if rows else np.zeros((0, 20), np.uint8),
                  after_sha1=np.stack([digest(r[3]) for r in rows]) if rows else np.zeros((0, 20), np.uint8))
    return arrays, dict(predictions=len(rows), seconds=round(time.perf_counter() - started, 2))


def load_class(spec):
    module, _, name = spec.partition(':')
    if not name:
        raise SystemExit(f'--model takes module:Class, got {spec!r}')
    if '' not in sys.path:
        sys.path.insert(0, '')          # a model file in the current directory
    try:
        cls = getattr(importlib.import_module(module), name)
        lacking = [m for m in ('predict', 'update') if not callable(getattr(cls, m, None))]
        if lacking:
            raise SystemExit(f'--model {spec}: the class needs the methods predict(history, action) and '
                             f'update(history, action, next_frame, valid); missing: {", ".join(lacking)}')
        return cls
    except ModuleNotFoundError as e:
        raise SystemExit(f'--model {spec}: cannot import {e.name!r}; run from the directory that holds the module '
                         'or add it to PYTHONPATH') from None
    except AttributeError:
        raise SystemExit(f'--model {spec}: the module {module!r} has no class {name!r}') from None


def _job(spec, kwargs, row, root, out):
    try:
        arrays, summary = run_recording(load_class(spec), kwargs, row['game'], locate(root, row))
    except Exception as e:
        e.add_note(f"while running {spec} on {row['game']} {row['recording']}")
        raise
    meta = dict(mecheval=VERSION, runner=True, model=spec, model_args=kwargs, game=row['game'],
                recording=row['recording'], **summary)
    np.savez_compressed(Path(out)/f"{row['recording']}.npz", meta=np.array(json.dumps(meta)), **arrays)
    return row['recording'], summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', required=True, help='module:Class of the model')
    ap.add_argument('--arg', action='append', default=[], metavar='KEY=VALUE', help='keyword argument for the model')
    add_set_arguments(ap)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, help='only the first N recordings of each game (a partial run is not '
                                               'comparable with the paper)')
    ap.add_argument('--workers', type=int, default=1, help='parallel processes, one model per recording')
    ap.add_argument('--skip-existing', action='store_true', help='keep the predictions already in --out')
    args = ap.parse_args(argv)
    kwargs = dict(a.split('=', 1) for a in args.arg)
    chosen, root = selected(ap, args)
    rows, seen = [], {}
    for r in chosen:
        seen[r['game']] = seen.get(r['game'], 0) + 1
        if args.limit is None or seen[r['game']] <= args.limit:
            rows.append(r)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        rows = [r for r in rows if not (args.out/f"{r['recording']}.npz").exists()]
    load_class(args.model)   # fail early on a wrong name
    jobs = [(args.model, kwargs, r, root, args.out) for r in rows]
    started = time.perf_counter()
    game_of = {r['recording']: r['game'] for r in rows}

    def show(i, recording, summary):
        print(f"[{i:>3}/{len(jobs)}] {game_of[recording]} {recording[:8]}  {summary['predictions']:>4} predictions  "
              f"{summary['seconds']:.1f} s", flush=True)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            for i, (recording, summary) in enumerate(pool.map(_job, *zip(*jobs)), 1):
                show(i, recording, summary)
    else:
        for i, job in enumerate(jobs, 1):
            show(i, *_job(*job))
    print(f'{len(rows)} recordings in {time.perf_counter() - started:.0f} s -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
