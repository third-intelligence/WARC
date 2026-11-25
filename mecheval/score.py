"""Score next-frame predictions on the labels of MechEval.

    python -m mecheval score --recordings <dir> --pred name=<dir with <recording>.npz> [...] --out <dir>
    python -m mecheval score --labels <labels dir> --plays <plays> --pred name=<dir> --out <dir>      # own plays

For every labelled action, each predictor is scored on the whole frame, on the union of the mechanics' regions and
outside it; for every firing of a mechanic with a region, on that region: changed-cell accuracy (cells the action
changes), unchanged-cell accuracy and false edits (cells it leaves alone). Copy-last (the frame before) and the oracle
(the true next frame) are always scored as references, so a predictor cannot take these two names. The frames come from the recordings; the labels
(labels/<recording>.jsonl.gz) say which mechanics fire and which cells each one covers.

By default the firings of the 69 mechanics of the paper are scored (data/mechanics.json, in_paper); --mechanics all
adds the seven that do not recur.

A prediction file holds `action_index` (the step of each prediction) and `pred` (N x 64 x 64 colour indices); optional
`before_sha1` / `after_sha1` (SHA-1 of the frames the prediction was made for) are checked against the recording.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .data import DATA, add_set_arguments, cells, labels, locate, selected
from .run import transitions


def excluded_mechanics():
    """(game, mechanic) of the catalogued mechanics left out of the paper's set."""
    return {(m['game'], m['mechanic']) for m in json.loads((DATA/'mechanics.json').read_text()) if not m['in_paper']}


def mask_metrics(pred, before, after, mask):
    correct = pred == after
    changed = before != after

    def accuracy(m):
        n = int(m.sum())
        k = int((correct & m).sum())
        return {'correct': k, 'pixels': n, 'accuracy': k/n if n else None}
    keep = mask & ~changed
    errors = int((keep & ~correct).sum())
    n = int(keep.sum())
    return {'all': accuracy(mask), 'changed': accuracy(mask & changed), 'unchanged': accuracy(keep),
            'false_edits': errors, 'unchanged_pixels': n, 'false_edit_rate': errors/n if n else None}


def _mask(runs, shape=(64, 64)):
    m = np.zeros(shape[0]*shape[1], bool)
    m[cells(runs)] = True
    return m.reshape(shape)


def load_predictions(path):
    z = np.load(path)
    sha = 'before_sha1' in z.files
    return {int(i): (z['pred'][k], bytes(z['before_sha1'][k]) if sha else None, bytes(z['after_sha1'][k]) if sha else None)
            for k, i in enumerate(z['action_index'])}


def score_recording(row, recording_path, predictions, mechanics='paper'):
    """row: an entry of data/recordings.json; predictions: {name: {step: (pred, sha1 before or None, sha1 after or
    None)}}; mechanics: 'paper' or 'all'. Returns the rows (dicts) and counts."""
    header, steps = labels(row['recording'], row.get('labels'))
    frames = {step: (before, after) for step, _, before, after, _ in transitions(recording_path)}
    absent = [s['step'] for s in steps if s['step'] not in frames]
    if absent:
        raise RuntimeError(f"{row['recording']}: the labels name step {absent[0]}, which the play does not have")
    skip = excluded_mechanics() if mechanics == 'paper' else set()
    stats = dict(transitions=0, occurrences=0, scored_occurrences=0, missing_predictions=0, unscored_transitions=0)
    rows = []
    for s in steps:
        before, after = frames[s['step']]
        sha = (hashlib.sha1(before.tobytes()).digest(), hashlib.sha1(after.tobytes()).digest())
        predictors = {'copy_last': before, 'oracle': after}
        for name, preds in predictions.items():
            p = preds.get(s['step'])
            if p is None:
                stats['missing_predictions'] += 1
                continue
            if p[1] is not None and (p[1], p[2]) != sha:
                raise RuntimeError(f"{row['recording']}: frame mismatch at action {s['step']} ({name})")
            predictors[name] = p[0]
        if len(predictors) != len(predictions) + 2:
            stats['unscored_transitions'] += 1
            continue
        stats['transitions'] += 1
        regions = {c: _mask(r) for c, r in s['regions'].items()}
        inter = {c: _mask(r) for c, r in s['interaction_regions'].items()}
        union = np.zeros((64, 64), bool)
        for m in regions.values():
            union |= m
        whole = np.ones((64, 64), bool)
        rows.append(dict(kind='transition', game=header['game'], level=s['level'], action_index=s['step'],
                         initialization=s['initialization'],
                         frame={n: mask_metrics(p, before, after, whole) for n, p in predictors.items()},
                         union={n: mask_metrics(p, before, after, union) for n, p in predictors.items()},
                         outside={n: mask_metrics(p, before, after, ~union) for n, p in predictors.items()}))
        for occ in s['mechanics']:
            if occ['occurred'] is not True or (header['game'], occ['mechanic']) in skip:
                continue
            stats['occurrences'] += 1
            m = regions.get(occ['mechanic'])
            if m is None:
                m = inter.get(occ['mechanic'])
            if m is None or not m.any():
                continue
            stats['scored_occurrences'] += 1
            rows.append(dict(kind='occurrence', game=header['game'], level=s['level'], action_index=s['step'],
                             category=occ['mechanic'], occurrence_index=occ['occurrence_index'],
                             prior_occurrences=occ['exposure'], initialization=s['initialization'],
                             exposure_origin=s['exposure_origin'],
                             bbox={n: mask_metrics(p, before, after, m) for n, p in predictors.items()}))
    return rows, stats


RESERVED = ('copy_last', 'oracle')


def _job(job):
    row, root, preds, out, mechanics = job
    name = row['recording']
    loaded = {n: load_predictions(Path(d)/f'{name}.npz') for n, d in preds.items()}
    rows, stats = score_recording(row, locate(root, row), loaded, mechanics)
    with gzip.open(Path(out)/f'{name}.scores.jsonl.gz', 'wt') as f:
        for x in rows:
            f.write(json.dumps(x) + '\n')
    return row, stats


def _collect(results, totals, quiet, n):
    done = 0
    for row, stats in results:
        done += 1
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        if not quiet:
            print(f"[{done:>3}/{n}] {row['game']} {row['recording'][:8]}  {stats['transitions']:>4} transitions  "
                  f"{stats['scored_occurrences']:>4} occurrences", flush=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_set_arguments(ap)
    ap.add_argument('--pred', action='append', default=[], metavar='NAME=DIR',
                    help='a predictor and the directory of its <recording>.npz files (repeatable)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mechanics', choices=('paper', 'all'), default='paper')
    ap.add_argument('--workers', type=int, default=1, help='parallel processes')
    ap.add_argument('--quiet', action='store_true', help='print only the totals')
    args = ap.parse_args(argv)
    preds = {}
    for p in args.pred:
        name, sep, directory = p.partition('=')
        if not sep or not name:
            raise SystemExit(f'--pred takes NAME=DIR, got {p!r}')
        if name in RESERVED:
            raise SystemExit(f'--pred: the name {name!r} is taken by a built-in reference; choose another name')
        if name in preds:
            raise SystemExit(f'--pred: the name {name!r} is given twice')
        preds[name] = directory
    rows_, root = selected(ap, args)
    missing = [(n, r['recording']) for r in rows_ for n, d in preds.items()
               if not (Path(d)/f"{r['recording']}.npz").exists()]
    if missing:
        raise SystemExit(f'{len(missing)} prediction files missing, e.g. {missing[0][0]}: {missing[0][1]}.npz')
    args.out.mkdir(parents=True, exist_ok=True)
    totals = {}
    jobs = [(r, root, preds, args.out, args.mechanics) for r in rows_]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            results = pool.map(_job, jobs)
            done = _collect(results, totals, args.quiet, len(jobs))
    else:
        done = _collect(map(_job, jobs), totals, args.quiet, len(jobs))
    print(f"{done} recordings scored: {totals['transitions']} transitions, {totals['scored_occurrences']} occurrences"
          + (f", {totals['unscored_transitions']} transitions without a prediction" if totals['unscored_transitions'] else ''))
    if not totals['transitions']:
        raise SystemExit('nothing was scored: no labelled step has a prediction from every predictor')
    return 0


if __name__ == '__main__':
    sys.exit(main())
