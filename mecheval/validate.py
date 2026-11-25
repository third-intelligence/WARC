"""Check a directory of predictions before scoring it.

    python -m mecheval validate <predictions dir> [--recordings <dir>] [--games ...]
    python -m mecheval validate <predictions dir> --labels <labels dir> --plays <plays>      # own plays

Every scored recording (data/recordings.json, or the recordings of --games) needs <recording>.npz with action_index
(the steps predicted) and pred (N x 64 x 64 colour indices 0-15), and every labelled step needs a prediction. With
--recordings, the frame digests that the runner stores (before_sha1, after_sha1) are checked against the recordings.
The check also reports whether the files come from `python -m mecheval run`, which enforces the order of the protocol;
files made otherwise can be scored, but nothing shows that each prediction came before its update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .data import add_set_arguments, labels, locate, selected
from .run import transitions


def check_file(path, recording, recording_path=None, labels_dir=None):
    """A list of problems with one prediction file."""
    problems = []
    z = np.load(path)
    for key in ('action_index', 'pred'):
        if key not in z.files:
            return [f'no {key}']
    steps, pred = z['action_index'], z['pred']
    if pred.ndim != 3 or pred.shape[1:] != (64, 64) or len(pred) != len(steps):
        problems.append(f'pred has shape {pred.shape} for {len(steps)} steps')
    elif not np.issubdtype(pred.dtype, np.integer) or pred.min(initial=0) < 0 or pred.max(initial=0) > 15:
        problems.append('pred must hold integer colour indices 0-15')
    if len(set(steps.tolist())) != len(steps):
        problems.append('repeated steps in action_index')
    have = {int(s): k for k, s in enumerate(steps)}
    need = [r['step'] for r in labels(recording, labels_dir)[1]]
    lacking = [s for s in need if s not in have]
    if lacking:
        problems.append(f'{len(lacking)} labelled steps without a prediction, e.g. step {lacking[0]}')
    if recording_path is not None and 'before_sha1' in z.files and not lacking:
        frames = {step: (before, after) for step, _, before, after, _ in transitions(recording_path)}
        for s in need:
            j = have[s]
            if (bytes(z['before_sha1'][j]) != hashlib.sha1(frames[s][0].tobytes()).digest()
                    or bytes(z['after_sha1'][j]) != hashlib.sha1(frames[s][1].tobytes()).digest()):
                problems.append(f'step {s} was predicted for other frames than the recording\'s')
                break
    return problems


def _job(job):
    predictions, row, root, frames = job
    path = Path(predictions)/f"{row['recording']}.npz"
    if not path.exists():
        return path.name, ['missing'], False
    problems = check_file(path, row['recording'], locate(root, row) if frames else None, row.get('labels'))
    z = np.load(path)
    return path.name, problems, bool('meta' in z.files and json.loads(str(z['meta'])).get('runner'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('predictions', type=Path)
    add_set_arguments(ap)
    ap.add_argument('--workers', type=int, default=1, help='parallel processes')
    args = ap.parse_args(argv)
    if not args.predictions.is_dir():
        raise SystemExit(f'{args.predictions} is not a directory')
    rows, root = selected(ap, args, need_recordings=False)
    frames = bool(args.recordings or args.labels)      # the plays are at hand for the frame check
    jobs = [(str(args.predictions), row, root, frames) for row in rows]
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    bad, runner, n = 0, 0, len(jobs)
    for name, problems, from_runner in results:
        runner += from_runner
        if problems:
            bad += 1
            print(f"{'missing ' if problems == ['missing'] else 'invalid '} {name}"
                  + ('' if problems == ['missing'] else ': ' + '; '.join(problems)))
    print(f'{n - bad} of {n} prediction files valid; {runner} written by python -m mecheval run')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
