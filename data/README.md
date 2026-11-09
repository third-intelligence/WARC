# Pretraining data

The models are pretrained on the human gameplay recordings of the 25 public ARC-AGI-3 games published by the ARC Prize
Foundation with its human dataset (https://arcprize.org/blog/arc-agi-3-human-dataset, "Public Demo dataset"). Download
them from the link in that post; the 340 recordings the released models were trained on are listed in `SHA256SUMS`.

Place the recordings as

```
data/ARC-AGI-3/data/training/<game>/<id>.recording.jsonl
data/ARC-AGI-3/data/evaluation/<game>/<id>.recording.jsonl
```

and check them:

```bash
cd data && sha256sum -c --quiet SHA256SUMS && echo OK      # 340 files
```

The file names matter: for every game, the last recording by file name among those that reach the game's top level
is the validation recording. `SHA256SUMS` lists the 340 files with the names the released models were trained with.
