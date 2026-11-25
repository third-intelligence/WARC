"""Label a play of one of the 20 games: which mechanics fire at every step and which cells each one covers.

    python -m mecheval annotate <play.recording.jsonl> [...] --out <dir>   # recordings in the ARC-AGI-3 format
    python -m mecheval annotate <actions.json> --out <dir>                 # {"game_id": ..., "actions": [[id, x, y], ...]}
    python -m mecheval annotate --recordings <dir> --out <dir>             # the 279 recordings of the paper

The play is replayed through the game's program, which logs the mechanics that fire. For every action up to the
completion of level 2, the labels record these mechanics, how often each has fired before (its exposure), and the
cells on which a prediction of the next frame is scored. When the play carries frames, every replayed frame is checked
against the recorded one and the replay stops at the first mismatch.

Writes <dir>/<name>.jsonl.gz in the format of labels/ (a header line, then one line per action). For the 279
recordings of the paper the output is identical to the files in labels/. For an action list it also writes the
replayed play, <dir>/<name>.recording.jsonl, which run and score read the frames from.

Two games, tn36 and s5i5, behave differently on the server that recorded the human plays and on the current offline
engine (layer order, rendering and which overlapping sprite a click picks). --engine recorded replays them as the
server did (the labels of the paper), current as the offline engine does, and auto (the default) tries recorded first
and falls back to current when a frame does not match. Every other game replays the same either way.

The annotator needs arcengine 0.9.3 and scipy in addition to numpy.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from . import logs
from .data import VERSION

FORMAT = 'mecheval-labels'
COMPLETE = 'L1_L2_checked'      # the replay reached the end of the play, or the completion of level 2
ENGINES = {'recorded': 'replay_v3', 'current': 'replay_v3_current'}
_installed = False


def _install():
    """The v6 overlay patches the engine once per process, before the engine's other modules run."""
    global _installed
    if not _installed:
        from .engine.overlay import install
        install()
        _installed = True


def runs(indices):
    """Flat cell indices (row-major, 64 x 64) as [start, length] runs."""
    out = []
    for i in sorted(indices):
        if out and out[-1][0] + out[-1][1] == i:
            out[-1][1] += 1
        else:
            out.append([i, 1])
    return out


def _action(a):
    data = a.get('data') or {}
    return [int(logs.action_id(a['id'])), int(data.get('x', -1)), int(data.get('y', -1))]


def annotate_recording(game, recording, *, engine='recorded'):
    """Label one recording file (ARC-AGI-3 format) with one engine. Returns (info, steps): info holds the replay
    status and any error, steps the label lines."""
    _install()
    from .engine.collect_samples import collect
    from .engine.gimmick_occurrences import ExposureCounter
    counter = ExposureCounter()
    steps = []

    def sink(sample):
        rows = counter.update(sample['gimmick_occurrences'])
        regions = {c: runs(r['bbox']) for c, r in sample['endpoint_regions']['categories'].items()}
        inter = {c: runs(r['bbox']) for c, r in sample['interaction_regions']['categories'].items()}
        diff = np.asarray(sample['before'], np.uint8) != np.asarray(sample['after'], np.uint8)
        covered = np.zeros(diff.size, bool)
        for r in list(sample['endpoint_regions']['categories'].values()) + list(sample['interaction_regions']['categories'].values()):
            covered[r['bbox']] = True
        mechanics = []
        for row in rows:
            if row['occurred'] is False:
                continue
            cat = row['category']
            mechanics.append(dict(mechanic=cat, occurred=row['occurred'], exposure=row['prior_occurrences'],
                                  occurrence_index=row['occurrence_index']))
        steps.append(dict(step=int(sample['action_index']), level=int(sample['level']), action=_action(sample['action']),
                          initialization=sample['initialization'], exposure_origin=sample['exposure_origin'],
                          mechanics=mechanics, regions=regions, interaction_regions=inter,
                          changed_outside=int((diff.ravel() & ~covered).sum())))

    result = collect(game, None, environment_profile=ENGINES[engine], recording_path=Path(recording), include_initial=False,
                     compute_metrics=False, compute_occurrences=True, transition_sink=sink, recover_initial=True)
    alignment = result.get('alignment', {})
    info = dict(engine=engine, status=alignment.get('status'), mismatch_action=alignment.get('mismatch_action'),
                error=result.get('error') or alignment.get('error'))
    return info, steps


def _annotate_auto(game, recording, engine):
    if engine != 'auto':
        return annotate_recording(game, recording, engine=engine)
    first = annotate_recording(game, recording, engine='recorded')
    if first[0]['status'] not in ('frame_mismatch', 'metadata_mismatch') or game not in ('tn36', 's5i5'):
        return first
    second = annotate_recording(game, recording, engine='current')
    second[0]['note'] = ('the recorded engine stopped at action {}: {}; replayed with the current engine'
                         .format(first[0]['mismatch_action'], first[0]['status']))
    return second


def annotate(log, *, game=None, engine='auto', out=None):
    """Label a recording or an action list. Returns (header, steps, info). For an action list, the replayed play is
    written to <out>/<name>.recording.jsonl, the file the scorer reads the frames from."""
    log = Path(log)
    kind, game_id = logs.inspect(log)
    game = game or logs.game_of(game_id)
    logs.check_supported(game, game_id)
    name = log.name.split('.')[0]
    header = dict(format=FORMAT, version=VERSION, game=game, game_id=game_id, recording=name,
                  recording_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    from .run import transitions
    if kind == 'recording':
        info, steps = _annotate_auto(game, log, engine)
        play = log
    else:
        engine = 'recorded' if engine == 'auto' else engine
        directory = Path(out) if out is not None else Path(tempfile.mkdtemp())
        directory.mkdir(parents=True, exist_ok=True)
        play = logs.actions_to_recording(log, directory/f'{name}.recording.jsonl', ENGINES[engine])
        info, steps = annotate_recording(game, play, engine=engine)
        header.update(recording_sha256=hashlib.sha256(play.read_bytes()).hexdigest(), input='actions')
    # the protocol (run.transitions) ends a play at its first empty frame; later steps are not scored
    last = max((step for step, *_ in transitions(play)), default=-1)
    steps = [s for s in steps if s['step'] <= last]
    header['steps'] = len(steps)
    if info['status'] != COMPLETE:        # the replay stopped early; the labels cover the steps before that
        header['replay'] = {k: info[k] for k in ('status', 'mismatch_action', 'error') if info.get(k) is not None}
    if info.get('note'):
        header['note'] = info['note']
    if info['engine'] != 'recorded':
        header['engine'] = info['engine']
    return header, steps, info


def write(out_dir, header, steps):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir/f"{header['recording']}.jsonl.gz"
    data = ('\n'.join(json.dumps(x) for x in [header] + steps) + '\n').encode()
    with open(path, 'wb') as fh:
        with gzip.GzipFile(fileobj=fh, mode='wb', mtime=0, filename='') as g:      # the same bytes on every run
            g.write(data)
    return path


def _one(log, out, game, engine):
    try:
        header, steps, info = annotate(log, game=game, engine=engine, out=out)
    except Exception as e:
        return True, f'{Path(log).name}: not labelled: {type(e).__name__}: {e}'
    path = write(out, header, steps)
    line = f"{Path(log).name}: {header['steps']} steps labelled ({info['engine']} engine)"
    stopped = info['status'] != COMPLETE
    if stopped:
        line += f", replay stopped: {info['status']}"
        if info.get('mismatch_action') is not None:
            line += f" at step {info['mismatch_action']}"
        if info['error']:
            line += f" ({info['error']})"
    return stopped, line + f' -> {path}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('logs', nargs='*', type=Path, help='recordings (.recording.jsonl) or action lists (.json)')
    ap.add_argument('--recordings', type=Path,
                    help='label the 279 recordings of the paper found below this directory')
    ap.add_argument('--games', nargs='+', help='with --recordings: only these games')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--game', help='game code (e.g. m0r0) when the log does not name it')
    ap.add_argument('--engine', choices=('auto', 'recorded', 'current'), default='auto',
                    help='how tn36 and s5i5 are replayed (see above); other games are not affected')
    ap.add_argument('--workers', type=int, default=1, help='parallel processes')
    args = ap.parse_args(argv)
    try:
        import arcengine  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        raise SystemExit(f'the annotator needs arcengine 0.9.3 and scipy ({e}); pip install arcengine==0.9.3 scipy')
    logs_ = list(args.logs)
    if args.recordings:
        from .data import locate, recordings, require
        require(args.recordings, args.games)
        logs_ += [locate(args.recordings, r) for r in recordings(args.games)]
    if not logs_:
        ap.error('give logs or --recordings')
    jobs = [(log, args.out, args.game, args.engine) for log in logs_]
    failed = 0
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            results = pool.map(_one, *zip(*jobs))
            for n, (err, line) in enumerate(results, 1):
                print(f'[{n:>3}/{len(jobs)}] {line}', flush=True)
                failed += err
    else:
        for n, job in enumerate(jobs, 1):
            err, line = _one(*job)
            print(f'[{n:>3}/{len(jobs)}] {line}', flush=True)
            failed += err
    if failed:
        print(f'{failed} of {len(jobs)} logs were not labelled to the end')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
