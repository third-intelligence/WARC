from __future__ import annotations
import sys
from pathlib import Path
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from hist_window.src.history import HistoryWindow, norm_action

class PlanWindow(HistoryWindow):

    def clone(self) -> 'PlanWindow':
        other = PlanWindow(self.n, self.image_size)
        other.frames = [f.copy() for f in self.frames]
        other.inter = list(self.inter)
        return other
