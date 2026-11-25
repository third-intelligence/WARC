"""WARC as a MechEval model: the fold model that holds the game out, with or without test-time training.

    python -m mecheval run --model mecheval.examples.warc:WARC --arg arm=ttt --arg device=cuda \\
        --recordings <dir> --out predictions/warc
    python -m mecheval run --model mecheval.examples.warc:WARC --arg arm=frozen --arg device=cuda \\
        --recordings <dir> --out predictions/warc_frozen

The model reads the last three frames and actions of the current segment, as in the paper, and with arm=ttt it takes
20 gradient steps after every valid transition. Test-time training on a GPU is not deterministic, so a new run matches
the paper's predictions (reference/) only up to small differences.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'agent'), str(ROOT/'agent'/'VARC')]


class WARC:
    def __init__(self, game, arm='ttt', device='cuda', seed='0', models=str(ROOT/'models')):
        from hist_window.src.history import HistoryWindow
        from warc_model.final_ttt import FinalPredictor
        manifest = json.loads((Path(models)/'manifest.json').read_text())
        fold = manifest['folds'][str(manifest['game_to_fold'][game])]
        self.predictor = FinalPredictor(str(Path(models)/fold['file']), game=game, arm=arm, device=device,
                                        seed=int(seed))
        self.window_class = HistoryWindow
        self.n = self.predictor.hist_n

    def _window(self, history):
        """The last n frames of the current segment and the actions between them."""
        frames, actions = history.segment
        k = min(self.n, len(frames))
        window = self.window_class(n=self.n)
        window.reset(frames[-k])
        for action, frame in zip(actions[len(actions) - (k - 1):] if k > 1 else [], frames[len(frames) - k + 1:]):
            window.push(action, frame)
        return window

    def predict(self, history, action):
        return self.predictor.predict(self._window(history), [action])[1][0]

    def update(self, history, action, next_frame, valid):
        self.predictor.observe(self._window(history), action, next_frame, valid_transition=valid, update=valid)
