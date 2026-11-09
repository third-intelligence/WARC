# agent

- `VARC/`: the subset of our research code built on VARC that the agent, the local runner and the training scripts
  import (no data, outputs or tests), with comments and docstrings removed and descriptive package names; the code
  is otherwise unchanged. Scripts put `agent/` and `agent/VARC/` on `sys.path`, as in the README's quick start.
- `warc_model/`: `model.py` (the WARC world model), `final_ttt.py` (the test-time training recipe), `varc2_optim.py`
  (Muon), `player.py` (the agent of the Kaggle notebook behind an act / observe protocol).

`agent/play.py` puts this directory and `VARC/` on `sys.path` and runs the agent on one game through
`VARC/automask_agent/eval/run_agent.py`; the training-free explorer (`warc_agents/`) needs only `warc_eval/` and
`vendor/`.
