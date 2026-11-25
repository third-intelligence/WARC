"""Summarize scored predictions in the measures of the paper (its main table).

    python -m mecheval report <scores dir> --out <dir> [--tau 0.7] [--k 100]

Reads the <recording>.scores.jsonl.gz files of `python -m mecheval score` and writes, for every predictor except the
oracle:

- table.csv / table.json: one row per predictor with
  k0          the share of changed cells predicted correctly before any exposure (%),
  lca@10      the learning curve area at exposure 10 (%),
  lca@100     the learning curve area at exposure 100 (%),
  reached@100 the share of mechanics that reach the criterion within 100 exposures (%; --k sets the 100),
  nr          the share of mechanics that do not reach the criterion (%),
  fe          the false-edit rate (%);
- mechanics.csv: every mechanic's LCA at exposures 10 and 100, the exposure at which it reaches the criterion, and
  whether it does;
- reached.csv: Reached@k at every k from 0 to the last exposure of any mechanic;
- curves.csv: every mechanic's learning curve and LCA at each exposure k, and the number of recordings behind it;
- learning_curves.csv: at each k, the mean of the learning curves over the mechanics with an occurrence at k.

Definitions, as in the paper:
- An occurrence of a mechanic is scored by the share of the cells its mechanic changes that the prediction gets right.
  An occurrence whose region has no changed cell is not scored; it still counts as an exposure.
- A mechanic's learning curve at exposure k is the mean score of its occurrences with k earlier occurrences, over all
  recordings.
- Its learning curve area (LCA) at exposure K is the mean of its learning curve over the scored exposures 0 to K.
  LCA@K is the mean over the mechanics that have a scored exposure up to K, each with equal weight; k0 is LCA@0.
- A mechanic reaches the criterion at the first exposure from which its LCA stays at or above tau (0.7) up to its last
  exposure. A mechanic whose LCA is below tau at its last exposure has not reached the criterion (n.r.).
- Reached@k: the share of all mechanics that have reached the criterion by exposure k.
- fe: unchanged cells predicted to change, over the unchanged cells of every labelled transition (pooled).
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from .data import recordings

TAU = 0.7
REACH_K = 100
LCA_KS = (10, 100)


def load(scores_dir):
    """(occurrences, false edits): occurrences as (predictor, mechanic, k, recording, accuracy); false edits as
    {predictor: [false edits, unchanged cells]}."""
    occurrences, false_edits = [], defaultdict(lambda: [0, 0])
    files = sorted(Path(scores_dir).glob('*.scores.jsonl.gz'))
    if not files:
        raise SystemExit(f'no .scores.jsonl.gz files in {scores_dir}')
    for path in files:
        recording = path.name.split('.')[0]
        with gzip.open(path, 'rt') as f:
            for line in f:
                r = json.loads(line)
                if r['kind'] == 'transition':
                    for name, m in r['frame'].items():
                        false_edits[name][0] += m['false_edits']
                        false_edits[name][1] += m['unchanged_pixels']
                    continue
                # mechanics only: interaction outcomes (collision.wall, ...) carry a dot and are not mechanics
                if '.' in r['category'] or not r['bbox']['oracle']['changed']['pixels']:
                    continue
                mechanic = (r['game'], r['category'])
                for name, m in r['bbox'].items():
                    occurrences.append((name, mechanic, r['prior_occurrences'], recording,
                                        m['changed']['correct'] / m['changed']['pixels']))
    return occurrences, false_edits


def criterion(curve, tau=TAU):
    """(first k from which the values stay at or above tau to the end, reached) of a curve [(k, value)] sorted by k.
    A curve that ends below tau has not reached it; its last k is returned."""
    if curve[-1][1] < tau:
        return curve[-1][0], False
    i = len(curve) - 1
    while i > 0 and curve[i - 1][1] >= tau:
        i -= 1
    return curve[i][0], True


def lca(curve):
    """[(k, LCA at k)]: the running mean of a learning curve [(k, value)] sorted by k."""
    out, total = [], 0.0
    for i, (k, a) in enumerate(curve):
        total += a
        out.append((k, total/(i + 1)))
    return out


def lca_at(series, K):
    """The LCA at exposure K of an LCA series, or None if no scored exposure is at or below K."""
    values = [v for k, v in series if k <= K]
    return values[-1] if values else None


def reached_at(mechanic_rows, k):
    """Reached@k (%) of one predictor's mechanic rows."""
    return round(100*sum(r['reached'] and r['reached_at'] <= k for r in mechanic_rows)/len(mechanic_rows), 3)


def summarize(occurrences, false_edits, tau=TAU, predictors=None, reach_k=REACH_K):
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))     # predictor -> mechanic -> k -> [(rec, acc)]
    for name, mechanic, k, recording, acc in occurrences:
        by[name][mechanic][k].append((recording, acc))
    names = predictors or [n for n in by if n != 'oracle']
    table, mechanic_rows, curve_rows, mean_rows, reach_rows = [], [], [], [], []
    for name in names:
        mechanics = by[name]
        rows, reached = [], 0
        at = {K: [] for K in (0,) + LCA_KS}
        perk = defaultdict(list)
        for mechanic in sorted(mechanics):
            ks = mechanics[mechanic]
            curve = [(k, float(np.mean([a for _, a in ks[k]]))) for k in sorted(ks)]
            series = lca(curve)
            for K in at:
                v = lca_at(series, K)
                if v is not None:
                    at[K].append(v)
            first, ok = criterion(series, tau)
            reached += ok
            rows.append(dict(predictor=name, game=mechanic[0], mechanic=mechanic[1],
                             **{f'lca@{K}': None if lca_at(series, K) is None else round(100*lca_at(series, K), 3)
                                for K in LCA_KS},
                             reached_at=first if ok else None, reached=ok,
                             occurrences=sum(len(v) for v in ks.values())))
            for (k, a), (_, l) in zip(curve, series):
                perk[k].append(a)
                curve_rows.append(dict(predictor=name, game=mechanic[0], mechanic=mechanic[1], k=k,
                                       recordings=len({r for r, _ in ks[k]}), accuracy=round(100*a, 3),
                                       lca=round(100*l, 3)))
        mean_rows += [dict(predictor=name, k=k, mechanics=len(v), accuracy=round(100*float(np.mean(v)), 3))
                      for k, v in sorted(perk.items())]
        mechanic_rows += rows
        last = max(max(ks) for ks in mechanics.values())
        reach_rows += [dict(predictor=name, k=k, reached=reached_at(rows, k)) for k in range(last + 1)]
        fe = false_edits[name]
        table.append(dict(predictor=name, mechanics=len(mechanics),
                          k0=round(100*float(np.mean(at[0])), 3),
                          **{f'lca@{K}': round(100*float(np.mean(at[K])), 3) for K in LCA_KS},
                          **{f'reached@{reach_k}': reached_at(rows, reach_k)},
                          nr=round(100*(1 - reached/len(mechanics)), 3),
                          fe=round(100*fe[0]/fe[1], 3) if fe[1] else None))
    return table, mechanic_rows, curve_rows, mean_rows, reach_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('scores', type=Path, help='directory of <recording>.scores.jsonl.gz files')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tau', type=float, default=TAU, help='the LCA criterion (default 0.7)')
    ap.add_argument('--k', type=int, default=REACH_K, help='the k of Reached@k in the table (default 100)')
    args = ap.parse_args(argv)
    occurrences, false_edits = load(args.scores)
    table, mechanic_rows, curve_rows, mean_rows, reach_rows = summarize(occurrences, false_edits, args.tau,
                                                                         reach_k=args.k)
    if not table:
        raise SystemExit(f'no scored occurrence in {args.scores}; run python -m mecheval score first')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'table.json').write_text(json.dumps(dict(tau=args.tau, k=args.k, rows=table), indent=1) + '\n')
    write_csv(args.out/'table.csv', table)
    write_csv(args.out/'mechanics.csv', mechanic_rows)
    write_csv(args.out/'reached.csv', reach_rows)
    write_csv(args.out/'curves.csv', curve_rows)
    write_csv(args.out/'learning_curves.csv', mean_rows)
    key = f'reached@{args.k}'
    print(f"{'predictor':<22}{'k=0':>7}{'LCA@10':>8}{'LCA@100':>9}{'Reached@' + str(args.k):>13}{'n.r.':>7}{'FE':>7}")
    for r in table:
        print(f"{r['predictor']:<22}{r['k0']:>7.1f}{r['lca@10']:>8.1f}{r['lca@100']:>9.1f}{r[key]:>13.1f}{r['nr']:>7.1f}"
              + (f"{r['fe']:>7.2f}" if r['fe'] is not None else f"{'-':>7}"))
    scored = {p.name.split('.')[0] for p in Path(args.scores).glob('*.scores.jsonl.gz')}
    own = scored - {r['recording'] for r in recordings()}
    if own:
        print(f'note: {len(own)} of the {len(scored)} scored plays are not among the 279 recordings of the paper; '
              'these values are for your own set and are not the paper\'s')
    elif len(scored) < 279 or any(r['mechanics'] < 69 for r in table):
        print(f'note: {len(scored)} of 279 recordings and {max(r["mechanics"] for r in table)} of 69 mechanics are '
              'scored; these values are not comparable with the paper')
    print(f'wrote table.csv, table.json, mechanics.csv, reached.csv, curves.csv, learning_curves.csv to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
