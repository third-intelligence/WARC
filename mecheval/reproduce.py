"""Reproduce the paper's values of WARC and frozen WARC from the reference predictions.

    python -m mecheval reproduce --recordings <dir> [--work <dir>] [--workers 8]

Scores the predictions in reference/warc and reference/warc_frozen, summarizes them and compares every value with
reference/expected.json at the paper's rounding. Exits with 1 if anything differs.
"""
import argparse
import json
from pathlib import Path
import sys

from . import report, score
from .data import PACKAGE

REFERENCE = PACKAGE/'reference'
DIGITS = {'k0': 1, 'lca@10': 1, 'lca@100': 1, 'reached@100': 1, 'nr': 1, 'fe': 2}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--recordings', type=Path, required=True, help='a directory that holds the recordings (any layout)')
    ap.add_argument('--work', type=Path, default=Path('mecheval_reproduce'), help='where the scores go')
    ap.add_argument('--workers', type=int, default=1, help='parallel processes')
    args = ap.parse_args(argv)
    expected = json.loads((REFERENCE/'expected.json').read_text())
    names = list(expected['rows'])
    score.main(['--recordings', str(args.recordings), *[f'--pred={n}={REFERENCE/n}' for n in names],
                '--out', str(args.work/'scores'), '--workers', str(args.workers), '--quiet'])
    occurrences, false_edits = report.load(args.work/'scores')
    table = {r['predictor']: r for r in report.summarize(occurrences, false_edits, expected['tau'], names)[0]}
    failed = 0
    for name, want in expected['rows'].items():
        for key, value in want.items():
            got = round(table[name][key], DIGITS[key])
            ok = abs(got - value) < 1e-9
            failed += not ok
            print(f"{'ok  ' if ok else 'DIFF'} {name:<12} {key:<11} expected {value:<8} reproduced {got}")
    print('reproduced the paper' if not failed else f'{failed} values differ')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
