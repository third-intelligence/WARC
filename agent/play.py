"""Play one ARC-AGI-3 game with the agent of the submission: the graph explorer with WARC updated after every
transition. The world model, the explorer and the test-time training recipe are the ones of the Kaggle notebook.

    python -B agent/play.py --game tr87 --ckpt models/fold2/checkpoint_best.pt \\
        --environments-dir data/pub25/games --budget 1000 --seed 0

The fold models refuse the games they were pretrained on; add --no-strict-split for the all-25 model.
Environment files of the public games: python -c "from warc_eval.prepare import prepare; prepare('data/pub25')"
(run with agent on sys.path). Results go to --out-dir (summary.json, steps.json).
"""
import sys
from pathlib import Path

AA3 = Path(__file__).resolve().parent
sys.path[:0] = [str(AA3), str(AA3 / 'VARC')]

from automask_agent.eval.run_agent import main  # noqa: E402

if __name__ == '__main__':
    main()
