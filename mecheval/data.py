"""The 279 human recordings that MechEval scores, their labels, and a check that a local copy matches them.

The recordings belong to the ARC Prize Foundation and are not redistributed here. Download the public demo set of the
ARC-AGI-3 human dataset (https://arcprize.org/blog/arc-agi-3-human-dataset) and pass a directory that holds the
recordings as --recordings: the unzipped download as it is, or the data/ directory of ../data/README.md. A recording is
found at its path in data/recordings.json, or else by its file name anywhere below that directory, and is checked
against its SHA-256.
labels/<recording>.jsonl.gz holds the labels of a recording: the mechanics that fire at every step and their cells.

    python -m mecheval check-data --recordings <dir>

run, validate and score work on these recordings by default. With --labels <dir> --plays <files or directories> they
work on other labelled plays instead: every <name>.jsonl.gz in the labels directory is paired with the play
<name>.recording.jsonl (see "Label format" in README.md).
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE/'data'
LABELS = PACKAGE/'labels'
VERSION = '1.0'


def recordings(games=None):
    """The scored recordings: game, game_id, recording (its id), path (below the data root) and sha256."""
    rows = json.loads((DATA/'recordings.json').read_text())
    return [r for r in rows if not games or r['game'] in games]


_INDEX = {}


def locate(root, row):
    """The file of a recording: at its path below root, or else the file with its name anywhere below root."""
    if root is None:
        return Path(row['path'])
    path = Path(root)/row['path']
    if path.exists():
        return path
    key = Path(root).resolve()
    if key not in _INDEX:
        _INDEX[key] = {p.name: p for p in sorted(key.rglob('*.recording.jsonl'))}
    return _INDEX[key].get(Path(row['path']).name, path)


def labels(recording, directory=None):
    """(header, steps) of a recording's labels, from labels/ or from another labels directory."""
    with gzip.open(Path(directory or LABELS)/f'{recording}.jsonl.gz', 'rt') as f:
        header = json.loads(f.readline())
        return header, [json.loads(line) for line in f]


def cells(runs):
    """Flat cell indices (row-major, 64 x 64) of [first cell, length] runs."""
    return [s + j for s, n in runs for j in range(n)]


def check(root, games=None):
    """(missing, mismatched) recordings below root."""
    missing, mismatched = [], []
    for row in recordings(games):
        path = locate(root, row)
        if not path.exists():
            missing.append(Path(row['path']).name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            mismatched.append(str(path))
    return missing, mismatched


def require(root, games=None):
    missing, mismatched = check(root, games)
    if missing or mismatched:
        raise SystemExit(f'{len(missing)} recordings missing and {len(mismatched)} different below {root};'
                         ' run python -m mecheval check-data for the list')


def own_plays(labels_dir, plays):
    """Rows like those of recordings() for the label files of a directory, each paired with its play
    <name>.recording.jsonl among the given files and directories."""
    labels_dir = Path(labels_dir)
    files = sorted(labels_dir.glob('*.jsonl.gz'))
    if not files:
        raise SystemExit(f'no <name>.jsonl.gz label files in {labels_dir}')
    found = {}
    for p in map(Path, plays):
        if not p.exists():
            raise SystemExit(f'--plays: {p} does not exist')
        for c in (sorted(p.rglob('*.recording.jsonl')) if p.is_dir() else [p]):
            found.setdefault(c.name.split('.')[0], c)
    rows, lacking = [], []
    for f in files:
        name = f.name.split('.')[0]
        header, _ = labels(name, labels_dir)
        if 'game' not in header:
            raise SystemExit(f'{f}: the header line has no "game"')
        play = found.get(name)
        if play is None:
            lacking.append(name)
            continue
        digest = header.get('recording_sha256')
        if digest and hashlib.sha256(play.read_bytes()).hexdigest() != digest:
            raise SystemExit(f'{play} is not the play that {f.name} was written for (recording_sha256 differs)')
        rows.append(dict(game=header['game'], game_id=header.get('game_id'), recording=name, path=str(play.resolve()),
                         sha256=digest, labels=str(labels_dir.resolve())))
    if lacking:
        raise SystemExit(f'{len(lacking)} label files have no play among --plays, e.g. {lacking[0]}.recording.jsonl')
    return rows


def add_set_arguments(ap):
    ap.add_argument('--recordings', type=Path, help='a directory that holds the 279 recordings of the paper '
                                                   '(any layout; found by file name)')
    ap.add_argument('--games', nargs='+', help='only these games (not comparable with the paper)')
    ap.add_argument('--labels', type=Path, help='a directory of your own label files, instead of the paper\'s')
    ap.add_argument('--plays', type=Path, nargs='+', help='with --labels: the plays (<name>.recording.jsonl files, or '
                                                          'directories that hold them)')


def selected(ap, args, need_recordings=True):
    """(rows, root) of the set a command works on: the paper's recordings, or own labelled plays (root None)."""
    if args.labels or args.plays:
        if not (args.labels and args.plays):
            ap.error('--labels and --plays go together')
        if args.recordings:
            ap.error('give --recordings (the paper\'s set) or --labels with --plays (your own), not both')
        rows = own_plays(args.labels, args.plays)
        return [r for r in rows if not args.games or r['game'] in args.games], None
    if args.recordings:
        require(args.recordings, args.games)
    elif need_recordings:
        ap.error('give --recordings, or --labels with --plays')
    return recordings(args.games), args.recordings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--recordings', type=Path, required=True, help='a directory that holds the recordings')
    ap.add_argument('--games', nargs='+', help='check only these games')
    args = ap.parse_args(argv)
    missing, mismatched = check(args.recordings, args.games)
    n = len(recordings(args.games))
    for p in missing:
        print('missing   ', p)
    for p in mismatched:
        print('different ', p)
    print(f'{n - len(missing) - len(mismatched)} of {n} recordings found and identical')
    return 1 if missing or mismatched else 0


if __name__ == '__main__':
    sys.exit(main())
