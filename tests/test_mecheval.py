"""Tests of MechEval. Two tests need the recordings: set MECHEVAL_RECORDINGS to a directory that holds them. One reproduces the paper; the other labels three recordings again with the annotator (it needs
arcengine and scipy) and compares them with the included labels.

    python -m pytest tests/test_mecheval.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

from mecheval import reproduce, report, run, validate
from mecheval.data import cells, labels, recordings

REFERENCE = Path(__file__).resolve().parents[1]/'mecheval'/'reference'


def test_criterion():
    assert report.criterion([(0, 0.1), (1, 0.9), (2, 0.5), (3, 0.85), (4, 0.9)], 0.8) == (3, True)
    assert report.criterion([(0, 0.9), (1, 0.95)], 0.8) == (0, True)
    assert report.criterion([(0, 0.9), (5, 0.5)]) == (5, False)      # ends below the criterion (0.7): its last exposure


def test_lca_is_the_running_mean_of_the_learning_curve():
    series = report.lca([(0, 0.0), (1, 1.0), (3, 1.0)])
    assert [k for k, _ in series] == [0, 1, 3]
    assert [v for _, v in series] == pytest.approx([0.0, 0.5, 2/3])
    assert report.lca_at(series, 2) == pytest.approx(0.5)            # the scored exposures 0 and 1
    assert report.lca_at(report.lca([(2, 1.0)]), 1) is None          # no scored exposure up to K


def test_summary_weights_mechanics_equally():
    occ = [('m', ('g', 'a'), 0, 'r1', 0.0), ('m', ('g', 'a'), 1, 'r1', 1.0), ('m', ('g', 'a'), 2, 'r1', 1.0),
           ('m', ('g', 'a'), 3, 'r1', 1.0), ('m', ('g', 'b'), 0, 'r1', 1.0)]
    table, rows = report.summarize(occ, {'m': [1, 100]}, predictors=['m'])[:2]
    table = table[0]
    assert table['k0'] == 50.0                                        # (0 + 1) / 2
    assert table['lca@10'] == pytest.approx(87.5, abs=0.001)          # (3/4 + 1) / 2
    assert table['reached@100'] == 100.0 and table['nr'] == 0.0 and table['fe'] == 1.0
    assert [r['reached_at'] for r in rows] == [3, 0]                  # LCA of a: 0, 0.5, 0.67, 0.75; of b: 1
    assert report.reached_at([dict(reached_at=3, reached=True), dict(reached_at=None, reached=False)], 3) == 50.0
    assert report.reached_at([dict(reached_at=3, reached=True), dict(reached_at=None, reached=False)], 2) == 0.0


def test_criterion_counts_exposures_that_one_recording_reaches():
    # k = 2 is reached by one recording only; its low score still lowers the LCA below the criterion
    occ = [('m', ('g', 'a'), k, r, acc) for k, r, acc in
           [(0, 'r1', 0.0), (0, 'r2', 0.0), (1, 'r1', 1.0), (1, 'r2', 1.0), (2, 'r1', 0.0)]]
    rows = report.summarize(occ, {'m': [0, 1]}, predictors=['m'])[1]
    assert not rows[0]['reached'] and rows[0]['reached_at'] is None


def test_runner_applies_changes():
    current = np.zeros((64, 64), np.uint8)
    frame = run._as_frame({'changes': [[3, 5, 7]]}, current)
    assert frame[5, 3] == 7 and frame.sum() == 7
    with pytest.raises(ValueError):
        run._as_frame(np.full((64, 64), 16), current)


def test_labels_cover_every_recording():
    rows = recordings()
    assert len(rows) == 279 and len({r['game'] for r in rows}) == 20
    header, steps = labels(rows[0]['recording'])
    assert header['recording_sha256'] == rows[0]['sha256'] and header['steps'] == len(steps)
    assert cells([[3, 2], [10, 1]]) == [3, 4, 10]


def test_reference_predictions_are_valid():
    for name in ('warc', 'warc_frozen'):
        files = sorted((REFERENCE/name).glob('*.npz'))
        assert {f.stem for f in files} == {r['recording'] for r in recordings()}
        assert validate.check_file(files[0], files[0].stem) == []


@pytest.mark.skipif(not os.environ.get('MECHEVAL_RECORDINGS'), reason='set MECHEVAL_RECORDINGS to reproduce the paper')
def test_reproduce_paper(tmp_path):
    assert reproduce.main(['--recordings', os.environ['MECHEVAL_RECORDINGS'], '--work', str(tmp_path)]) == 0


@pytest.mark.skipif(not os.environ.get('MECHEVAL_RECORDINGS'), reason='set MECHEVAL_RECORDINGS to run the annotator')
def test_annotator_gives_the_included_labels(tmp_path):
    pytest.importorskip('arcengine')
    pytest.importorskip('scipy')
    from mecheval import annotate
    from mecheval.data import LABELS, locate
    rows = [next(r for r in recordings() if r['game'] == g) for g in ('ls20', 'tr87', 'tn36')]
    logs = [str(locate(os.environ['MECHEVAL_RECORDINGS'], r)) for r in rows]
    assert annotate.main(logs + ['--out', str(tmp_path)]) == 0
    for r in rows:
        name = f"{r['recording']}.jsonl.gz"
        assert (tmp_path/name).read_bytes() == (LABELS/name).read_bytes()


def test_hand_made_labels_of_another_environment(tmp_path, capsys):
    """A made-up environment, written by hand in the label format: a lamp at cell (0, 0) that toggles at every action
    and a marker in row 5 that moves one cell to the right at every second action."""
    import gzip
    import json
    from mecheval import score
    frames = [np.zeros((64, 64), int)]
    frames[0][5, 0] = 3
    lines, lamp, marker = [], 0, 0
    for step in range(1, 7):
        f = frames[-1].copy()
        f[0, 0] = 9 - f[0, 0]
        mechanics = [dict(mechanic='lamp', occurred=True, exposure=lamp, occurrence_index=lamp + 1)]
        regions = {'lamp': [[0, 1]]}
        lamp += 1
        if step % 2 == 0:
            x = step // 2
            f[5, x - 1], f[5, x] = 0, 3
            mechanics.append(dict(mechanic='marker', occurred=True, exposure=marker, occurrence_index=marker + 1))
            regions['marker'] = [[5*64 + x - 1, 2]]
            marker += 1
        frames.append(f)
        lines.append(dict(step=step, level=1, action=[1, -1, -1], initialization='start', exposure_origin='start',
                          mechanics=mechanics, regions=regions, interaction_regions={}, changed_outside=0))
    play = tmp_path/'plays'/'demo.recording.jsonl'
    play.parent.mkdir()
    with play.open('w') as out:
        for i, f in enumerate(frames):
            data = dict(frame=[f.tolist()], state='NOT_FINISHED', levels_completed=0)
            if i:
                data['action_input'] = dict(id=1, data={})
            out.write(json.dumps(dict(data=data)) + '\n')
    (tmp_path/'labels').mkdir()
    header = dict(format='mecheval-labels', version='1.0', game='lamps', recording='demo', steps=len(lines))
    with gzip.open(tmp_path/'labels'/'demo.jsonl.gz', 'wt') as out:
        out.write('\n'.join(json.dumps(x) for x in [header] + lines) + '\n')
    # a predictor that knows the lamp from its second exposure and never the marker
    pred = np.stack([frames[0] if i == 1 else np.where(np.arange(64*64).reshape(64, 64) == 0, frames[i][0, 0],
                                                       frames[i - 1]) for i in range(1, 7)]).astype(np.uint8)
    (tmp_path/'pred').mkdir()
    np.savez(tmp_path/'pred'/'demo.npz', action_index=np.arange(1, 7), pred=pred)
    common = ['--labels', str(tmp_path/'labels'), '--plays', str(tmp_path/'plays')]
    assert validate.main([str(tmp_path/'pred')] + common) == 0
    assert run.main(['--model', 'mecheval.models:CopyLast', '--out', str(tmp_path/'copy')] + common) == 0
    assert score.main(common + ['--pred', f"lamp_learner={tmp_path/'pred'}", '--out', str(tmp_path/'scores')]) == 0
    assert report.main([str(tmp_path/'scores'), '--out', str(tmp_path/'report')]) == 0
    assert "not the paper's" in capsys.readouterr().out
    import csv
    rows = {(r['predictor'], r['mechanic']): r for r in csv.DictReader(open(tmp_path/'report'/'mechanics.csv'))}
    # the lamp scores 0, 1, 1, 1, 1, 1, so its LCA is 0, 0.5, 0.67, 0.75, ...: it reaches the criterion (0.7) at exposure 3
    assert rows[('lamp_learner', 'lamp')]['reached_at'] == '3' and rows[('lamp_learner', 'lamp')]['reached'] == 'True'
    assert rows[('lamp_learner', 'marker')]['reached'] == 'False' and rows[('copy_last', 'lamp')]['lca@10'] == '0.0'
