from __future__ import annotations
import argparse
import functools
import json
import sys
from pathlib import Path
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_AA3_ROOT = _PROJ_ROOT.parent
for _p in (_PROJ_ROOT, _AA3_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

def parse_params(text: str):
    from automask_agent.src.agent import DEFAULT_PARAMS
    out = {}
    for item in filter(None, (t.strip() for t in text.split(','))):
        k, v = item.split('=')
        if k not in DEFAULT_PARAMS:
            raise SystemExit(f'unknown param: {k} (choices: {sorted(DEFAULT_PARAMS)})')
        out[k] = type(DEFAULT_PARAMS[k])(float(v))
    return out

def main() -> None:
    from automask_agent.src.agent import ARM_CONFIGS, AutomaskAgent
    from warc_model.final_ttt import FinalPredictor
    from warc_model.player import injected_predictor
    p = argparse.ArgumentParser(
        description='the agent of the submission: the graph explorer with WARC updated after every transition',
        epilog='Other options (--environments-dir DIR, --device DEVICE, --candidates N, --branch N, --epsilon E, '
               '--update-steps N, --mode {full,l1}, --log-every N) are passed on to the explorer, graph_agent/eval/run_agent.py.')
    p.add_argument('--arm', default='automask_strict_h2', choices=sorted(ARM_CONFIGS))
    p.add_argument('--ckpt', default='')
    p.add_argument('--game', default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--budget', type=int, default=1000)
    p.add_argument('--out-dir', default='')
    p.add_argument('--out-tag', default='')
    p.add_argument('--params', default='', help='for example stall=20,macro_len=12')
    p.add_argument('--no-strict-split', action='store_true', help='allow a game the checkpoint was pretrained on (needed for the all-25 model)')
    ours, rest = p.parse_known_args()
    cfg = dict(ARM_CONFIGS[ours.arm])
    params = parse_params(ours.params)
    import graph_agent.src.agent as agent_mod
    agent_mod.MpcAgent = functools.partial(AutomaskAgent, dedup=cfg['dedup'], credit=cfg['credit'], topk=cfg['topk'], predict_batch=cfg['predict_batch'], macro=cfg['macro'], restore=cfg['restore'], objnav=cfg['objnav'], bandit=cfg['bandit'], tree=cfg['tree'], automask=cfg['automask'], run_budget=ours.budget, params=params)
    out_dir = ours.out_dir
    if not out_dir:
        ckpt_tag = Path(ours.ckpt).parent.name if ours.ckpt else 'no_model'
        tag = f'_{ours.out_tag}' if ours.out_tag else ''
        out_dir = str(_PROJ_ROOT / 'automask_agent' / 'outputs' / ours.arm / ckpt_tag / f'{ours.game}_s{ours.seed}{tag}')
    passthrough = ['--arm', cfg['base_arm'], '--game', ours.game, '--seed', str(ours.seed), '--budget', str(ours.budget), '--out-dir', out_dir, '--candidates', str(cfg['candidates'])]
    if ours.ckpt:
        passthrough += ['--ckpt', ours.ckpt]
    if ours.out_tag:
        passthrough += ['--out-tag', ours.out_tag]
    sys.argv = [sys.argv[0]] + passthrough + rest
    print(f'[agent] arm={ours.arm} {cfg} params={params}', flush=True)
    strict_split = not ours.no_strict_split

    def factory(ckpt_, game_, arm_, device_, seed_, *, update_steps=100, budget=4000):
        return FinalPredictor(ckpt_, game_, arm_, device_, seed_, update_steps=update_steps, budget=budget, strict_split=strict_split)
    from graph_agent.eval import run_agent as base_run
    with injected_predictor(factory):
        base_run.main()
    path = Path(out_dir) / 'summary.json'
    summary = json.loads(path.read_text())
    arm = ours.arm + (f'_{ours.out_tag}' if ours.out_tag else '')
    summary.update(arm=arm, base_arm=cfg['base_arm'], config=cfg, agent_arm=ours.arm, predictor='final_ttt', strict_split=strict_split)
    path.write_text(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
