"""sys.path for the training scripts: the VARC tree (data, trainer, backbone) and warc_model (the WARC model).

Both live under agent, the code the Kaggle notebook runs, so the model that is trained
here is the model that is loaded there."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
AA3 = REPO / 'agent'
VARC = AA3 / 'VARC'


def use_training_paths():
    """The trainer's own resolution: wm_core/src first (varc2, the data module), then the VARC tree, then warc_model."""
    for p in (AA3, VARC, VARC / 'wm_core' / 'src'):
        while str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))
