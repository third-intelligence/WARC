# Pretraining data

The models are pretrained on the human gameplay recordings of the 25 public ARC-AGI-3 games published by the ARC Prize
Foundation with its human dataset (https://arcprize.org/blog/arc-agi-3-human-dataset, "Public Demo dataset"). Download
the zip file from the link in that post. It holds the recordings as `public_games-dataset/<game>/<id>.recording.jsonl`.
Move that folder into place and check the files:

```bash
unzip <downloaded zip> -d /tmp/arc3 -x "__MACOSX/*"
mkdir -p data/ARC-AGI-3/data
mv /tmp/arc3/public_games-dataset data/ARC-AGI-3/data/training
cd data && sha256sum -c --quiet SHA256SUMS && echo OK      # 340 files
```

`SHA256SUMS` lists the 340 recordings the models of the paper were trained on. The trainer reads every recording in
`data/ARC-AGI-3/data/training/<game>/`, so a file that is not listed there (a later version of the download may hold
more) must be removed to train on the same data. The file names matter: for every game, the last recording by file
name among those that reach the game's top level is the validation recording.
