# MechEval

MechEval measures how fast a world model learns the mechanics of a game it has not seen. It covers 69 mechanics in 20
ARC-AGI-3 games, such as a player that moves one step per key or a step counter that shrinks with every action. For
every step of 279 human recordings, it reads from the game program which mechanics fire and which cells each one
changes. A model replays a recording and predicts every next frame before it may learn from that transition. Each
prediction is scored on the cells that each firing mechanic changes, against the number of times the mechanic has fired
before, its exposures.

MechEval is the evaluation of the paper. It has two parts. The annotator replays a play through the game program and writes its labels: the
mechanics that fire at every step and the cells each one changes. The scorer measures predictions on these labels. The
labels of the paper's 279 recordings are included, so scoring a model does not need the annotator. MechEval needs no
GPU and none of the rest of this repository, except the example that wraps WARC. The list of mechanics and the logging
code of the annotator build on a concurrent study of the same games, credited once it is published
([docs/third_party.md](../docs/third_party.md)).

## Commands

Run every command from the repository root as `python -m mecheval <command>`; `--help` lists the options. Paths in this
file are relative to the repository root.

| Command | What it does |
|---|---|
| `check-data` | Checks the downloaded recordings against their SHA-256. |
| `run` | Runs a world model on the recordings under the protocol below and writes its predictions. |
| `validate` | Checks a directory of predictions before scoring. |
| `score` | Scores predictions on the labels. |
| `report` | Summarizes the scores in the measures of the paper. |
| `reproduce` | Reproduces the values of WARC and frozen WARC from the paper's predictions. |
| `annotate` | Labels a play of one of the 20 games: a recording or a list of actions. |

## Installation

Python 3.12 with `numpy` (version in `requirements.txt`):

```bash
pip install numpy==2.5.3
pip install arcengine==0.9.3 scipy==1.18.1      # only for `annotate`
```

`tests/test_mecheval.py` checks the scorer, the runner and the label format (`pip install pytest`, then
`python -m pytest tests`). With `MECHEVAL_RECORDINGS` set to the directory of the recordings, it also reproduces the
paper and labels three recordings again with the annotator.

## Data

The recordings belong to the ARC Prize Foundation and are not part of this repository. Download the Public Demo dataset of
the ARC-AGI-3 human dataset (https://arcprize.org/blog/arc-agi-3-human-dataset), the same files that WARC is pretrained
on. `--recordings` takes the `data/` directory set up as in `data/README.md`, or any directory that holds the
recordings, such as the unzipped download as it is; MechEval finds each recording by its file name and checks its
SHA-256:

```bash
python -m mecheval check-data --recordings data
```

`mecheval/data/recordings.json` lists the 279 recordings of the 20 games with their game versions and SHA-256. The labels of
each recording are in `mecheval/labels/`. They hold no frames; the scorer reads the frames from the recordings.

## Reproducing the paper

```bash
python -m mecheval reproduce --recordings data --workers 8
```

This scores the paper's predictions in `mecheval/reference/` (WARC and frozen WARC, test-time seed 0) and compares
the result with the values in `mecheval/reference/expected.json`.
Frozen WARC does not update, and its LCA@10, LCA@100, n.r. and FE are those of the paper's main table. For WARC,
the main table averages three test-time seeds.

## Evaluating a model

Write a class with three methods:

```python
class MyModel:
    def __init__(self, game, **kwargs):              # a new model for every recording; game is a code such as "ls20"
        ...

    def predict(self, history, action):              # the next frame
        ...

    def update(self, history, action, next_frame, valid):   # the true next frame; learn from it or not
        ...
```

- `history.frames` holds the frames of the recording so far, and the last one, `history.current`, is the current
  frame. `history.actions[i]` led from `frames[i]` to `frames[i + 1]`.
- `history.segment_start` is where the current segment began. A segment restarts after a RESET, a level change, or a
  game over. `history.segment` returns its frames and actions.
- An action is `(id, x, y)`, with `id` 1 to 7 for ACTION1 to ACTION7. `x` (column) and `y` (row) are the clicked cell
  for ACTION6, and -1 otherwise.
- `predict` returns the next frame as a 64 x 64 array of colour indices 0 to 15. It may instead return only the cells
  that change, `{"changes": [[x, y, colour], ...]}`, which the runner applies to the current frame. This suits models
  that write text, such as language models.
- `valid` is False for a transition that completes a level or ends the game.

`mecheval/models.py` holds the smallest model, CopyLast, and `mecheval/examples/warc.py` wraps WARC. Then:

```bash
python -m mecheval run --model mymodule:MyModel --arg key=value --recordings data --out mecheval_out/predictions/mymodel
python -m mecheval validate mecheval_out/predictions/mymodel --recordings data
python -m mecheval score --recordings data --pred mymodel=mecheval_out/predictions/mymodel \
    --pred warc=mecheval/reference/warc --out mecheval_out/scores --workers 8
python -m mecheval report mecheval_out/scores --out mecheval_out/report
```

The model's module must be importable: run from the directory that holds it, or add that directory to `PYTHONPATH`.
A predictor cannot be named `copy_last` or `oracle`, which `score` adds as references.

`run` takes `--games` and `--limit` (the first recordings of each game) for a quick check. A subset is not comparable
with the paper. `--workers` runs recordings in parallel, one model per process.

### Running WARC

`mecheval/examples/warc.py` wraps WARC with the fold model that holds each game out (`models/`). It needs the
installation of `README.md` (PyTorch). Frozen WARC runs on a CPU; WARC with test-time training needs a GPU:

```bash
OMP_NUM_THREADS=2 python -m mecheval run --model mecheval.examples.warc:WARC --arg arm=frozen --arg device=cpu \
    --recordings data --out mecheval_out/predictions/warc_frozen --workers 24
python -m mecheval run --model mecheval.examples.warc:WARC --arg arm=ttt --arg device=cuda \
    --recordings data --out mecheval_out/predictions/warc
python -m mecheval score --recordings data --pred warc_frozen=mecheval_out/predictions/warc_frozen \
    --pred warc=mecheval_out/predictions/warc --out mecheval_out/scores_warc --workers 8
python -m mecheval report mecheval_out/scores_warc --out mecheval_out/report_warc
```

Test-time training on a GPU is not deterministic, so a new run differs slightly from the reference predictions. Frozen WARC
on a CPU with the weights from Hugging Face gives the reference predictions cell for cell.

## Annotating your own plays

`annotate` labels any play of the 20 games, in the game versions of the paper (`mecheval/data/versions.json`):

```bash
python -m mecheval annotate my_play.recording.jsonl --out mecheval_out/labels
python -m mecheval annotate my_actions.json --out mecheval_out/labels
```

- A recording in the ARC-AGI-3 format (`.recording.jsonl`) starts with a RESET, and each later line holds an action
  and the frames that followed it. The annotator replays the actions and checks every replayed frame against the
  recorded one. It stops at the first frame that differs, reports the step, and labels the steps before it.
- An action list is `{"game_id": "ls20-9607627b", "actions": [[id, x, y], ...]}`, with `id` 0 for RESET and 1 to 7
  for ACTION1 to ACTION7, and `x`, `y` for ACTION6 only (-1 otherwise). The frames come from the replay itself.

It writes `<name>.jsonl.gz` in the format of `mecheval/labels/`, up to the completion of level 2. `annotate --recordings data
--out <dir> --workers 8` labels the paper's 279 recordings again and gives the files in `mecheval/labels/` byte for byte.

The annotator runs each game with a program that logs which mechanics fire and on which objects
(`mecheval/games/gamesofts/`), and assigns the logged events to the catalogued mechanics and their cells (`mecheval/engine/`). Two
games, tn36 and s5i5, behave differently on the server that recorded the human plays and on the offline engine;
`--engine` chooses between them (see `annotate --help`).

For an action list, `annotate` also writes the replayed play, `<name>.recording.jsonl`, next to the labels.

`run`, `validate` and `score` take your labels with `--labels` and the plays with `--plays` (files, or directories
that hold `<name>.recording.jsonl`) in place of `--recordings`:

```bash
python -m mecheval run --model mymodule:MyModel --labels mecheval_out/labels --plays my_plays mecheval_out/labels \
    --out mecheval_out/predictions/mymodel
python -m mecheval score --labels mecheval_out/labels --plays my_plays mecheval_out/labels \
    --pred mymodel=mecheval_out/predictions/mymodel --out mecheval_out/scores
python -m mecheval report mecheval_out/scores --out mecheval_out/report
```

`report` then notes that the values are for your own set and are not the paper's.

## The protocol

- Every recording is followed from its first frame with the recorded actions, up to the completion of level 2. A
  recording that has no frame after some step ends there.
- At every action except RESET, the model predicts the next frame and only then receives it. `run` enforces this
  order. Predictions written in other ways can be scored, but nothing shows that the order was kept, and `validate`
  reports them as such.
- Every recording starts from a new model, and nothing carries over from one recording to the next.
- A model must not be trained on the recordings or the program of a game it is scored on. The paper holds games out
  in three folds (`mecheval/data/folds.json`) and pretrains on the human recordings of the other two folds.

## Measures

`report` writes `table.csv` with one row per model, `reached.csv` with Reached@k at every k, and `mechanics.csv`,
`curves.csv` and `learning_curves.csv` with the per-mechanic values behind them.

- **Score** of an occurrence: the share of the cells its mechanic changes that the prediction gets right. An
  occurrence whose region has no changed cell is not scored, but it still counts as an exposure.
- **Learning curve** of a mechanic: at each exposure k, the mean score of its occurrences with k earlier occurrences,
  over all recordings.
- **LCA** (learning curve area) of a mechanic at exposure K: the mean of its learning curve over the scored exposures
  0 to K. **LCA@K** is the mean over the mechanics, each with equal weight. The table gives LCA@10, LCA@100, and the
  score before any exposure (k=0, which is LCA@0).
- **Criterion**: a mechanic reaches it at the first exposure from which its LCA stays at or above 70% up to its last
  exposure (`--tau` changes the 70%). A mechanic whose LCA is below 70% at its last exposure has not reached the
  criterion (**n.r.**).
- **Reached@k**: the share of all mechanics that have reached the criterion by exposure k. The table gives Reached@100
  (`--k` changes the 100) and the share not reached (n.r.).
- **FE** (false-edit rate): the unchanged cells predicted to change, over all unchanged cells of every labelled
  transition, including steps where nothing changes, such as a blocked move.

## Label format

The scorer needs only two files per play: the labels and the play. `annotate` writes labels for the 20 games at the
game versions of the paper. For any other environment, labels written in this format by other means, for example by
a simulator that logs its own mechanics, are scored in the same way.

**Labels**, `<name>.jsonl.gz`: gzip-compressed JSON lines. The first line is the header.

| Header field | Content |
|---|---|
| `format`, `version` | `"mecheval-labels"`, `"1.0"`. |
| `game` | A name for the environment. Mechanics are counted per game: `(game, mechanic)` identifies a mechanic across plays. |
| `recording` | The name of the play, equal to `<name>`. |
| `steps` | The number of lines that follow. |
| `game_id`, `recording_sha256` | Optional. The game version, and the SHA-256 of the play file, which `run` and `score` check when it is given. |

Every later line labels one action of the play.

| Field | Content |
|---|---|
| `step` | The line of the play that holds this action and the frame after it. Line 0 is the first frame, so the first action is step 1. |
| `level` | The level the action is taken in (1 or 2 in the paper). |
| `action` | `[id, x, y]`, as in the play. |
| `mechanics` | The mechanics that fire at this step, one entry each (below). An entry whose name carries a dot (`collision.wall`, `invalid_action.no_button`) is an outcome that is not a mechanic, such as a blocked move; its cells, if any, are in `interaction_regions`, and `report` leaves it out. |
| `regions` | Cells by name. A firing mechanic is scored on the region of its name. A mechanic that fires without a visible change has no region; it is not scored and still counts as an exposure. Some lines also hold regions of names that do not fire at that step; no mechanic is scored on them. |
| `interaction_regions` | The cells of outcomes that are not mechanics: the dotted outcomes, and `movement`, the objects that moved. `score` scores them and `report` leaves them out. May be `{}`. |
| `changed_outside` | The number of changed cells that lie in no region. Informative only. |
| `initialization`, `exposure_origin` | How the play starts and where exposures are counted from. Copied into the scores and not used otherwise; any string. |

An entry of `mechanics`:

| Field | Content |
|---|---|
| `mechanic` | The name of the mechanic. It is the key of its cells in `regions`. |
| `occurred` | `true`. `annotate` writes `null` when the program could not decide whether the mechanic fired; such an entry is not scored. |
| `exposure` | The exposure index k: the number of earlier steps of this play at which the mechanic fired. The first firing has 0. |
| `occurrence_index` | `exposure + 1`. |

A region is a list of runs `[first cell, length]` over the frame in row-major order, so cell `i` is row `i // 64`,
column `i % 64`. It is a superset of the cells the mechanic changes: `annotate` writes the bounding boxes of the
mechanic's objects before and after the step. A firing is scored on the cells of its region that differ between the
frame before and the frame after. A firing whose region has no changed cell is not
scored, but it still counts as an exposure for the later ones.

**Play**, `<name>.recording.jsonl`: JSON lines in the format of the ARC-AGI-3 recordings. The scorer reads these
fields of each line's `data`:

| Field | Content |
|---|---|
| `frame` | A list of 64 x 64 grids of colour indices 0 to 15. The last grid is the frame after the action; line 0 holds the first frame. |
| `action_input` | `{"id": n, "data": {"x": ..., "y": ...}}` with `n` 0 for RESET and 1 to 7 otherwise; `x`, `y` only for action 6. Not read on line 0. |
| `state` | `"NOT_FINISHED"` while the play goes on. |
| `levels_completed` | The number of completed levels. The play is read up to the completion of level 2. |

Frames are 64 x 64 with 16 colours in this version. RESET actions are not predicted, so the labels must not have a
line for them.

## Other files

A prediction file, `<recording>.npz`, holds `action_index` (the step of each prediction) and `pred` (N x 64 x 64 colour
indices). `run` also stores `before_sha1` and `after_sha1`, the SHA-1 of the frames each prediction was made for, which
`score` checks against the recording, and `meta`, a JSON string that names the model.

`mecheval/data/mechanics.json` lists 76 mechanics with their game, kind, and a short description. They are the 81 of
the paper's catalogue except the three that never fire in the recordings and the two that never change a visible cell.
The paper scores 69 of them. It leaves out seven that fire fewer than three times in the median recording that contains them;
`score --mechanics all` includes them.

## Versions and limits

- Labels version 1.0, the ones in the paper. They were read from the game programs: each game was replayed with a
  program that logs which mechanics fire and on which objects, and every replayed frame was checked against the
  recording.
- The included labels cover the 279 human recordings. `annotate` labels other plays of the same 20 games; it does
  not cover other games or other versions of these games.
- Levels 1 and 2 only.
- The mechanics were catalogued from the human recordings. At 37 of the 29,893 labelled actions, some changed cells lie
  outside every region (`changed_outside`).

## Licence

Code, labels, and reference predictions are released under CC0, like the rest of this repository. The game programs
in `mecheval/games/` are derived from the ARC-AGI-3 game files of the ARC Prize Foundation and keep their MIT licence
(`mecheval/games/LICENSE`). The recordings belong to the ARC Prize Foundation.
