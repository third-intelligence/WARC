# Reproducibility

Two levels: rerun the Kaggle submission, or retrain the weights (13 to 20 hours per model on six GPUs).

## 1. Rerun the submission

The submitted notebook and the dataset it reads are public on Kaggle:

- notebook (Version 5): https://www.kaggle.com/code/onodaku/warc-arc3-notebook-v2?scriptVersionId=362286067
- dataset: https://www.kaggle.com/datasets/onodaku/warc-arc-agi-3-solver

The dataset holds the code of this repository's `agent/` as `src/`, the all-25 weights (`all25/checkpoint_best.pt`
on Hugging Face), and the solver settings. To rerun it, copy the notebook on Kaggle with the dataset and the
competition data attached, a GPU, and internet off.

Locally, the same agent (explorer, world model and update recipe) plays one game at a time through `agent/play.py`
against the environment files of the public games, which `agent/warc_eval/prepare.py` downloads from
the ARC Prize API; the README's quick start gives the commands. Each run writes `summary.json` (levels cleared, first
clear step, action counts, update statistics) and `steps.json` (every action with the prediction and update records).

The explorer alone, the Graph rows of the paper's closed-loop table, runs through the local runner, which also
scores the runs and writes a leaderboard:

```bash
cd agent
python -m warc_eval.cli run --dataset ../data/pub25 --output ../runs/explorer \
    --agents warc_agents.graph:GraphAgent --games tr87 --seeds 0 --budget 1000
```

`--agents` also takes the built-in `random`, `component_random` and `count` agents. The runner checks the vendored
explorer files against the hashes in `warc_agents/graph.py` before a run.

## 2. Retrain the weights

### Data

The models are trained on the human gameplay recordings of the 25 public ARC-AGI-3 games (340 recordings; see
`data/README.md` for where to get them and how to check them). The data root must contain
`data/training/<game>/<id>.recording.jsonl` and `data/evaluation/<game>/<id>.recording.jsonl` with the file names
listed in `data/SHA256SUMS`: the file names decide which recording of each game is the validation recording.

### Environment

Python 3.12, torch 2.7.0 (every released model was trained with it), and `requirements.txt`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### The submitted model (all 25 games)

```bash
cd warc
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PYTHONDONTWRITEBYTECODE=1 python -B -m torch.distributed.run \
    --nproc_per_node=6 --master_port=29615 train_all25.py --data-root /path/to/ARC-AGI-3 \
    --distributed --global-batch-size 516 --epochs 30 --num-workers 8 --no-wandb --save-dir runs/all25_s0
```

All 25 games are training games; one recording per game (the last by file name among those that reach the game's
top level) is the validation set, 314 training recordings and 25 validation recordings in all, 8,201,259 training
examples per epoch. `checkpoint_best.pt` is the epoch with the best validation exact-frame accuracy: 80.0% at
epoch 30 for the released model.

### The fold models of the paper

```bash
cd warc
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PYTHONDONTWRITEBYTECODE=1 python -B -m torch.distributed.run \
    --nproc_per_node=6 --master_port=29606 train.py --fold 2 --data-root /path/to/ARC-AGI-3 \
    --distributed --global-batch-size 516 --epochs 30 --num-workers 8 --no-wandb --save-dir runs/fold2
```

Each fold model is trained on the recordings of its 16 or 17 training games and evaluated on its held-out games.
The split (split seed 0, `agent/VARC/unified_wm/src/folds.py`):

| fold | held-out games | validation exact-frame accuracy (best epoch) |
|---|---|---|
| 1 | bp35 lf52 lp85 r11l re86 sb26 sc25 su15 wa30 | 85.7% (30) |
| 2 | cd82 cn04 dc22 m0r0 sk48 tn36 tr87 tu93 | 79.2% (30) |
| 3 | ar25 ft09 g50t ka59 ls20 s5i5 sp80 vc33 | 78.5% (29) |

Fold 2 trains on 225 recordings (5,555,175 examples per epoch) with 17 validation recordings.

### Recipe

| item | value |
|---|---|
| model | 10 pre-norm adaLN-Zero transformer blocks, width 512, 8 heads, RoPE, patch 2 on 64 x 64 frames; a learned vector added to the patch token of the clicked cell; 35.3M parameters (arch `unified_hist_cm`) |
| input, target | the last 3 frames and the action; cross-entropy over the colours of the next frame |
| optimizer | Muon on the 2-D weights of the blocks except the adaLN modulation (40 tensors), AdamW on the rest (72 tensors); lr 3e-4, weight decay 0, gradient clip 1 |
| schedule | 30 epochs, 6 warmup epochs then cosine; epoch 1 runs at lr 0 |
| batch | global 516 = 6 GPUs x 86 |
| augmentation | 9 colour permutations and the geometric transforms of the data module |
| other | bf16 autocast, torch.compile (reduce-overhead), seed 42, best checkpoint by validation exact-frame accuracy |

Six GPUs are part of the recipe: with 4 GPUs (`--nproc_per_node=4`, 4 x 129) the global batches are the same, but the
per-rank loss averaging and the dropout streams differ. Eight GPUs do not divide 516. On six RTX PRO 6000 Blackwell
Max-Q GPUs an epoch took 27 to 39 minutes plus evaluation, 13 to 20 hours per model. To continue an interrupted run,
add `--resume-checkpoint <save-dir>/checkpoint_last.pt`. `provenance.json` in the save directory records the command
and the SHA-256 of the code that ran.

### Test-time training (the recipe inside the agent)

`agent/warc_model/final_ttt.py`. After every valid transition (the game continues on the
same level): 20 gradient steps of minibatch 8 drawn from the newest 10 transitions and 20 older ones, each transition
in its rotation views; Muon and AdamW split as in pretraining, lr 3e-4 constant, weight decay 0.01, gradient clip 1,
bf16 autocast, all parameters trainable. A step with a non-finite loss or gradient is skipped; if a parameter becomes
non-finite the last finite weights are restored and updates stop for the rest of the game.

## 3. A check that needs no GPU

```bash
cd warc
python -B train.py --fold 2 --dry-run     # resolved arguments, model config and optimizer groups, no data
```
