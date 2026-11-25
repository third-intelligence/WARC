# MechEval labels

MechEval measures how fast a world model learns the mechanics of a game it has not seen. It covers 69 mechanics in 20
ARC-AGI-3 games, such as a player that moves one step per key or a step counter that shrinks with every action. For
every step of 279 human recordings, it reads from the game program which mechanics fire and which cells each one
changes. A model replays a recording and predicts every next frame before it may learn from that transition. Each
prediction is scored on the cells that each firing mechanic changes, against the number of times the mechanic has fired
before, its exposures.

This directory holds the labels of the paper's 279 recordings. The annotator and the scorer, with the measures of
the current version of the paper, are on the [`paper-v2`](https://github.com/third-intelligence/WARC/tree/paper-v2) branch. Its README explains how to
[evaluate your own world model](https://github.com/third-intelligence/WARC/blob/paper-v2/README.md#evaluating-your-world-model-on-mecheval).

| File | Content |
|---|---|
| `labels/<recording>.jsonl.gz` | The labels of one recording: the mechanics that fire at every step and the cells each one changes. |
| `data/recordings.json` | The 279 recordings of the 20 games, with their game versions, paths and SHA-256. |
| `data/mechanics.json` | The 76 catalogued mechanics with their game, kind, and a short description. `in_paper` marks the 69 the paper scores. |
| `data/versions.json` | The game version of each game. |
| `data/folds.json` | The three folds of the paper over the 25 public games. |

## Data

The recordings belong to the ARC Prize Foundation and are not part of this repository. Download the Public Demo dataset of
the ARC-AGI-3 human dataset (https://arcprize.org/blog/arc-agi-3-human-dataset). `data/recordings.json` gives each
recording's file name, its path once set up as in `../data/README.md`
(`ARC-AGI-3/data/training/<game>/<id>.recording.jsonl`), and its SHA-256.
The labels hold no frames; the frames come from the recordings.

## Measures

- **Mechanic accuracy** of an occurrence: the fraction of the cells its mechanic changes that the prediction gets
  right. These are the cells of its region that differ between the frame before and the frame after the step. An
  occurrence whose region has no changed cell is not scored, but it still counts as an exposure.
- **k=0** and **All**: per mechanic, the mean over its occurrences at exposure 0 (k=0) or at every exposure (All),
  then the mean over mechanics, each with equal weight.
- **Learning curve** of a mechanic: at each exposure k, the mean accuracy of its occurrences with k earlier occurrences,
  over all recordings.

The paper's protocol: every recording is followed from its first frame with the recorded actions, up to the completion
of level 2. At every action except RESET, the model predicts the next frame and only then receives it. Every recording
starts from a new model. A model must not be trained on the recordings or the program of a game it is scored on; the
paper holds games out in three folds (`data/folds.json`).

## Reading the labels

```python
import gzip, json
import numpy as np

def cells(runs):                                   # a region: runs [first cell, length], row-major over 64 x 64
    return np.concatenate([np.arange(a, a + n) for a, n in runs]) if runs else np.zeros(0, int)

def frames(path):                                  # line i of a recording: the frame after its action
    data = [json.loads(l)["data"] for l in open(path) if l.strip()]
    return [np.array(d["frame"][-1]) if d.get("frame") else None for d in data]   # the last line is a summary

def mechanic_accuracy(labels_path, recording_path, predict):
    """predict(step) returns the predicted 64 x 64 frame after the action of that step."""
    lines = gzip.open(labels_path, "rt").read().splitlines()
    f = frames(recording_path)
    out = []                                       # (mechanic, exposure, accuracy)
    for row in map(json.loads, lines[1:]):
        before, after = f[row["step"] - 1].ravel(), f[row["step"]].ravel()
        pred = np.asarray(predict(row["step"])).ravel()
        for m in row["mechanics"]:
            if not m["occurred"]:
                continue
            c = cells(row["regions"].get(m["mechanic"], []))
            c = c[before[c] != after[c]]           # the cells the mechanic changes
            if len(c):
                out.append((m["mechanic"], m["exposure"], float((pred[c] == after[c]).mean())))
    return out
```

## Label format

**Labels**, `<name>.jsonl.gz`: gzip-compressed JSON lines. The first line is the header.

| Header field | Content |
|---|---|
| `format`, `version` | `"mecheval-labels"`, `"1.0"`. |
| `game` | A name for the environment. Mechanics are counted per game: `(game, mechanic)` identifies a mechanic across plays. |
| `recording` | The name of the play, equal to `<name>`. |
| `steps` | The number of lines that follow. |
| `game_id`, `recording_sha256` | Optional. The game version, and the SHA-256 of the play file, of the recording it was written for. |

Every later line labels one action of the play.

| Field | Content |
|---|---|
| `step` | The line of the play that holds this action and the frame after it. Line 0 is the first frame, so the first action is step 1. |
| `level` | The level the action is taken in (1 or 2 in the paper). |
| `action` | `[id, x, y]`, as in the play. |
| `mechanics` | The mechanics that fire at this step, one entry each (below). An entry whose name carries a dot (`collision.wall`, `invalid_action.no_button`) is an outcome that is not a mechanic, such as a blocked move; its cells, if any, are in `interaction_regions`. The paper does not score these entries. |
| `regions` | Cells by name. A firing mechanic is scored on the region of its name. A mechanic that fires without a visible change has no region; it is not scored and still counts as an exposure. Some lines also hold regions of names that do not fire at that step; no mechanic is scored on them. |
| `interaction_regions` | The cells of outcomes that are not mechanics: the dotted outcomes, and `movement`, the objects that moved. The paper does not score them. May be `{}`. |
| `changed_outside` | The number of changed cells that lie in no region. Informative only. |
| `initialization`, `exposure_origin` | How the play starts and where exposures are counted from. Informative only. |

An entry of `mechanics`:

| Field | Content |
|---|---|
| `mechanic` | The name of the mechanic. It is the key of its cells in `regions`. |
| `occurred` | `true`. `null` when the game program could not decide whether the mechanic fired; such an entry is not scored. |
| `exposure` | The exposure index k: the number of earlier steps of this play at which the mechanic fired. The first firing has 0. |
| `occurrence_index` | `exposure + 1`. |

A region is a list of runs `[first cell, length]` over the frame in row-major order, so cell `i` is row `i // 64`,
column `i % 64`. It is a superset of the cells the mechanic changes: the bounding boxes of the
mechanic's objects before and after the step. The mechanic accuracy of a firing is computed on the cells of its
region that differ between the frame before and the frame after. A firing whose region has no changed cell is not
scored, but it still counts as an exposure for the later ones.

**Play**, `<name>.recording.jsonl`: the recording of the ARC-AGI-3 human dataset. These fields of each line's `data` are
the ones the labels refer to:

| Field | Content |
|---|---|
| `frame` | A list of 64 x 64 grids of colour indices 0 to 15. The last grid is the frame after the action; line 0 holds the first frame. |
| `action_input` | `{"id": n, "data": {"x": ..., "y": ...}}` with `n` 0 for RESET and 1 to 7 otherwise; `x`, `y` only for action 6. Not read on line 0. |
| `state` | `"NOT_FINISHED"` while the play goes on. |
| `levels_completed` | The number of completed levels. The play is read up to the completion of level 2. |

Frames are 64 x 64 with 16 colours in this version. RESET actions are not predicted, so the labels must not have a
line for them.


## Versions and limits

- Labels version 1.0, the ones in the paper. They were read from the game programs: each game was replayed with a
  program that logs which mechanics fire and on which objects, and every replayed frame was checked against the
  recording. The list of mechanics and the logging programs build on a concurrent study of the same games, credited
  once it is published.
- The labels cover the 279 human recordings of 20 games, at the game versions in `data/versions.json`, levels 1 and 2.
- The mechanics were catalogued from the human recordings. At 37 of the 29,893 labelled actions, some changed cells lie
  outside every region (`changed_outside`).

## Licence

The labels and the files in `data/` are released under CC0, like the rest of this repository. The games and the
recordings belong to the ARC Prize Foundation.
