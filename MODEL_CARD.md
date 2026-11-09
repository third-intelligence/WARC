---
license: cc0-1.0
tags:
  - world-model
  - arc-agi-3
  - test-time-training
  - next-frame-prediction
  - vision-transformer
  - pytorch
---

# WARC

[![Code: GitHub](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/third-intelligence/WARC)
[![License: CC0 1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey)](https://creativecommons.org/publicdomain/zero/1.0/)

WARC (World model for ARC-AGI-3) is a 35M-parameter action-conditioned vision transformer. It
predicts the next frame of a game, cell by cell, from the last three frames and
the action, and it keeps learning while an agent plays: after every observed
transition it updates all of its parameters by gradient descent. Pretrained on
human recordings of other games, it learns to predict four in five mechanics
of an unseen game with at least 80% accuracy on the cells they change and keeps
that accuracy to the end of the recordings.

## Model details

| | |
|---|---|
| Input | The last three 64x64 frames of 16 colours, the action after the last frame, and the two actions between the frames |
| Output | A distribution over the 16 colours of every cell of the next frame |
| Architecture | 2x2 patches, 10 pre-norm transformer blocks of width 512 with rotary position embeddings, modulated by the action through adaptive layer normalization; a learned vector added to the token of the clicked cell |
| Parameters | 35M |
| Pretraining | Next-frame cross-entropy on human recordings of the public games; Muon and AdamW, 30 epochs |
| Test-time training | 20 gradient steps on the ten newest and twenty earlier transitions after every observed transition; all parameters are updated |
| License | CC0 1.0 |

## Files

| File | Pretrained on | Held-out games | Validation exact-frame accuracy |
|---|---|---|---:|
| `fold1/checkpoint_best.pt` | 16 games | bp35 lf52 lp85 r11l re86 sb26 sc25 su15 wa30 | 85.7% |
| `fold2/checkpoint_best.pt` | 17 games | cd82 cn04 dc22 m0r0 sk48 tn36 tr87 tu93 | 79.2% |
| `fold3/checkpoint_best.pt` | 17 games | ar25 ft09 g50t ka59 ls20 s5i5 sp80 vc33 | 78.5% |
| `all25/checkpoint_best.pt` | all 25 games | none (one recording per game held out for validation) | 80.0% |

Each checkpoint (146 MB) holds the weights, the model configuration, and the
game split; `config.json` and `eval_history.jsonl` next to it record the
training configuration and the per-epoch validation accuracy. `manifest.json`
lists the SHA-256 of every checkpoint and the fold of every game.

## Usage

Install the code from the [GitHub repository](https://github.com/third-intelligence/WARC)
and download the weights:

```bash
git clone https://github.com/third-intelligence/WARC
cd WARC
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
hf download third-intelligence/WARC --local-dir models    # the four checkpoints (pip install huggingface_hub)
```

Predict the next frame:

```python
import sys
import torch

sys.path[:0] = ["agent", "agent/VARC"]
from warc_model.model import build_from_checkpoint

device = torch.device("cuda")
model, meta = build_from_checkpoint("models/all25/checkpoint_best.pt", device=device)

# The last three frames, (1, 3, 64, 64) colours 0-15. Actions: 1-5 are the keys ACTION1-ACTION5,
# 6 is a click at column x and row y; x and y are -1 for a key.
frames = torch.randint(0, 16, (1, 3, 64, 64), device=device)     # replace with frames of a game
action, x, y = (torch.tensor(v, device=device) for v in ([6], [10], [20]))
hist_action, hist_x, hist_y = (torch.tensor(v, device=device) for v in ([[1, 2]], [[-1, -1]], [[-1, -1]]))

with torch.no_grad():
    logits = model(frames[:, -1], torch.zeros(1, dtype=torch.long, device=device), action, x, y,
                   hist_obs=frames, hist_action_ids=hist_action, hist_action_xs=hist_x,
                   hist_action_ys=hist_y)["ffp_logits"]          # (1, 16, 64, 64)
next_frame = logits.argmax(1)[0]                                 # (64, 64) predicted colours
```

Test-time training inside a game loop (`FinalPredictor` in the repository), the
agent, the Kaggle notebook, and the training commands are documented in the
[README](https://github.com/third-intelligence/WARC#quick-start) and in
[docs/reproducibility.md](https://github.com/third-intelligence/WARC/blob/main/docs/reproducibility.md).

## Training data

The 340 human gameplay recordings of the 25 public ARC-AGI-3 games published by
the ARC Prize Foundation ([ARC-AGI-3 human dataset](https://arcprize.org/blog/arc-agi-3-human-dataset)).
The fold models use the recordings of their training games only; the all-25
model uses one recording per game for validation and the rest for training.

## Acknowledgements

WARC follows VARC ("ARC Is a Vision Problem!", Hu et al., 2025) and
reuses parts of the [VARC code](https://github.com/lillian039/VARC) (MIT). The
games and the recordings belong to the [ARC Prize Foundation](https://arcprize.org).
The repository lists every third-party component and its licence.

## License

The weights and the code are released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Third-party
components keep their own licences.
