# Pretraining

`train.py` trains a fold model of the paper and `train_all25.py` the submitted model; both run the vendored trainer in
`agent/VARC/unified_wm/src/offline_train.py` with the model builder replaced by the WARC
model of the agent (`warc_model/model.py`), so a checkpoint written here loads in the agent as it is. Commands, the
recipe and the expected numbers are in `../docs/reproducibility.md`.
