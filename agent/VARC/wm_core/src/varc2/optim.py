from __future__ import annotations
import math
from typing import List, Tuple
import torch
from torch.optim.lr_scheduler import LambdaLR
_NS_A, _NS_B, _NS_C = (3.4445, -4.775, 2.0315)

@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int=5) -> torch.Tensor:
    assert G.ndim == 2, f'expected a 2D matrix, got shape {tuple(G.shape)}'
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-07)
    for _ in range(steps):
        A = X @ X.T
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    if transposed:
        X = X.T
    return X

def _lr_adjust(adjust_lr_fn, rows: int, cols: int) -> float:
    if adjust_lr_fn == 'match_rms_adamw':
        return 0.2 * max(rows, cols) ** 0.5
    return max(1.0, rows / cols) ** 0.5

class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr: float=0.001, weight_decay: float=0.1, momentum: float=0.95, nesterov: bool=True, ns_steps: int=5, adjust_lr_fn=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim != 2:
                    raise ValueError(f'Muon only supports 2D parameters; got a parameter with shape {tuple(p.shape)}. Route non-2D params (bias / embedding / norm) to AdamW instead.')

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            mom = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            adjust = group['adjust_lr_fn']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - mom)
                d = g.lerp(buf, mom) if nesterov else buf
                d = zeropower_via_newtonschulz5(d, steps=ns_steps)
                eff_lr = lr * _lr_adjust(adjust, p.size(0), p.size(1))
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(d.to(p.dtype), alpha=-eff_lr)
        return loss

def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float=0.5, last_epoch: int=-1) -> LambdaLR:

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * progress * 2)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def split_muon_params(model: torch.nn.Module) -> Tuple[List[str], List[str]]:
    muon_names, adamw_names = ([], [])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_muon = p.ndim == 2 and name.endswith('.weight') and (name.split('.', 1)[0] in ('enc', 'dyn', 'dec')) and ('ada_mod' not in name)
        (muon_names if is_muon else adamw_names).append(name)
    n_trainable = sum((1 for p in model.parameters() if p.requires_grad))
    assert len(muon_names) + len(adamw_names) == n_trainable, 'param grouping dropped params'
    assert muon_names, 'muon group is empty -- check enc/dyn/dec module naming'
    assert all(('ada_mod' not in n for n in muon_names)), 'adaLN linear leaked into Muon'
    return (muon_names, adamw_names)

def build_optimizers(model: torch.nn.Module, *, lr: float, weight_decay: float, muon_lr: float=0.0, muon_momentum: float=0.95, muon_adjust_lr: str='match_rms_adamw', warmup_steps: int, total_steps: int):
    muon_names, adamw_names = split_muon_params(model)
    params = dict(model.named_parameters())
    muon = Muon([params[n] for n in muon_names], lr=muon_lr if muon_lr > 0 else lr, weight_decay=weight_decay, momentum=muon_momentum, adjust_lr_fn=muon_adjust_lr)
    adamw = torch.optim.AdamW([params[n] for n in adamw_names], lr=lr, weight_decay=weight_decay)
    optimizers = [muon, adamw]
    schedulers = [get_cosine_schedule_with_warmup(o, warmup_steps, total_steps) for o in optimizers]
    return (optimizers, schedulers)
