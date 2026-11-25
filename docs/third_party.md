# Third-party code

Everything not listed here is released under CC0 1.0 (`LICENSE`). The licence texts of the components below are in
`incl_licenses/`; vendored files keep the notices of their origin.

| Component | Path | Origin | Licence |
|---|---|---|---|
| VARC code: the vision-transformer backbone, the argument parser, the distributed-training and learning-rate helpers | `agent/VARC/vit_vendor/vendor/arc_vit.py` (adapted from `src/ARC_ViT.py`), `.../VARC/utils/args.py` (adapted from `utils/args.py`), `.../VARC/utils/distribution.py` and `.../VARC/utils/lr_scheduler.py` (reformatted, comments and docstrings removed); each file names its origin in a header | Keya Hu et al., https://github.com/lillian039/VARC ("ARC Is a Vision Problem!", Hu et al., 2025) | MIT (`incl_licenses/LICENSE_VARC`) |
| Cosine learning-rate schedule with warm-up | `agent/VARC/utils/lr_scheduler.py` | Hugging Face Transformers, https://github.com/huggingface/transformers (`src/transformers/optimization.py`, v4.23.1), through the VARC code | Apache License 2.0 (`incl_licenses/LICENSE_transformers`; notice in the file header) |
| Rotary position embedding | `agent/VARC/vit_vendor/vendor/pos_embed.py` | EVA-02, Beijing Academy of Artificial Intelligence, https://github.com/baaivision/EVA, based on https://github.com/lucidrains/rotary-embedding-torch; taken from the VARC code (`utils/pos_embed.py`) | MIT (`incl_licenses/LICENSE_EVA02`; notice in the file header) |
| Graph exploration agent: a training-free explorer in this repository | `agent/vendor/graph_exploration/` | Evgenii Rudakov, https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore, commit c2d98318ecca38c8ee9fb49b5af159e51770bad7 (`warc_agents/upstream_manifest.json`) | MIT (`LICENSE` in the directory; copy in `incl_licenses/LICENSE_graph_exploration`) |
| ARC-AGI-3 game programs for the MechEval annotator: 20 instrumented games and eight unmodified game files | `mecheval/games/gamesofts/modify/`, `mecheval/games/environment_files/` | ARC Prize Foundation, the ARC-AGI-3 game files; the instrumented versions are those of a concurrent study (credited once it is published) at an earlier revision, with some comments, docstrings and display strings removed (`mecheval/games/LICENSE`) | MIT, ARC Prize Foundation (`mecheval/games/LICENSE`) |
| Transition-record module of the instrumented games | `mecheval/games/gamesofts/_action_log.py` | A concurrent study (credited once it is published), at an earlier revision, with part of its records removed | MIT (`incl_licenses/LICENSE_instrumented_games`; notice in the file header) |
| Event-map reader and the event maps of ten games | `mecheval/engine/event_map.py`, `mecheval/data/event_maps/*.tsv` | A concurrent study (credited once it is published), at an earlier revision: the reader is adapted from its mapping module, and the maps keep columns of its mapping tables | MIT (`incl_licenses/LICENSE_instrumented_games`; notice in the file header) |
| Muon optimizer | `agent/warc_model/varc2_optim.py` (a copy of `agent/VARC/wm_core/src/varc2/optim.py`) | written by the team; the Newton-Schulz orthogonalization function is taken from Keller Jordan's Muon, https://github.com/KellerJordan/Muon | MIT for that function (`incl_licenses/LICENSE_Muon`; notice in the file headers); the rest CC0 |

The rest of `agent/VARC/` (data module, trainer, world-model backbone, frame window,
explorer) and `warc_model/` were written by the team and are released under CC0.

Python dependencies (`requirements.txt`): `arc-agi` and `arcengine` (ARC Prize Foundation, MIT), `timm` (Apache-2.0),
`einops` (MIT), `numpy`, `scipy`, `matplotlib`, `pillow`, `requests`; `torch` (BSD-3).

The human gameplay recordings used for pretraining and by MechEval belong to the ARC Prize Foundation and are not part
of this repository (`data/README.md`, `mecheval/README.md`). Apart from the MechEval game programs above, the games are
not part of it either.
