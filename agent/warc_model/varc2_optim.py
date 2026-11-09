"""Muon, a copy of the training code's wm_core/src/varc2/optim.py (``torch.optim.Muon`` only exists in torch >= 2.9).

The final TTT recipe (final_ttt.split_params) gives Muon the 2-D weights of the transformer blocks except the
adaLN modulation linears (zero-initialised; Newton-Schulz would normalise their first update to unit RMS) and
AdamW everything else.
"""

from __future__ import annotations

import torch

# Quintic Newton-Schulz coefficients (Keller Jordan). Tuned so the iteration pushes the
# singular values of a unit-norm matrix towards 1 (~ orthogonalization) in ~5 steps.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of a 2D matrix ``G`` via 5-step Newton-Schulz.

    Runs in bfloat16 for speed/stability. Returns a bf16 tensor (caller casts to the
    param dtype). The ``+1e-7`` in the normalization guards an all-zero gradient.
    """
    assert G.ndim == 2, f"expected a 2D matrix, got shape {tuple(G.shape)}"
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:                       # work on the shorter side (fewer NS flops)
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = _NS_B * A + _NS_C * (A @ A)
        X = _NS_A * X + B @ X
    if transposed:
        X = X.T
    return X


def _lr_adjust(adjust_lr_fn, rows: int, cols: int) -> float:
    """Per-matrix learning-rate multiplier (matches torch.optim.Muon semantics).

    - ``"match_rms_adamw"``: ``0.2 * sqrt(max(rows, cols))`` -- makes the update RMS
      match AdamW so AdamW-tuned lr/weight_decay can be reused ("Muon is Scalable").
    - ``"original"`` / ``None``: ``sqrt(max(1, rows / cols))``.
    """
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * (max(rows, cols) ** 0.5)
    return max(1.0, rows / cols) ** 0.5


class Muon(torch.optim.Optimizer):
    """Single-device Muon for 2D hidden weight matrices.

    Single-device: no ``torch.distributed`` calls. This is correct under DDP because
    DDP all-reduces gradients before ``step()``, so every rank holds identical grads
    and the deterministic Newton-Schulz update keeps the param replicas in sync.
    """

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.1,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 adjust_lr_fn=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters; got a parameter with shape "
                        f"{tuple(p.shape)}. Route non-2D params (bias / embedding / "
                        "norm) to AdamW instead.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            adjust = group["adjust_lr_fn"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mom)                  # EMA: buf = mom*buf + (1-mom)*g
                d = g.lerp(buf, mom) if nesterov else buf   # Nesterov lookahead
                d = zeropower_via_newtonschulz5(d, steps=ns_steps)
                eff_lr = lr * _lr_adjust(adjust, p.size(0), p.size(1))
                if wd != 0:
                    p.mul_(1 - lr * wd)                # decoupled weight decay (base lr)
                p.add_(d.to(p.dtype), alpha=-eff_lr)
        return loss
