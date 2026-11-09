from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = Path(__file__).resolve().parent
for _p in (str(_PROJ_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch
from torch import nn
from varc2.model import PRESETS, VARC2Model
from varc2.optim import Muon
ARCH_CHOICES = ('varc2', 'policy')

class VARC2ModelMS(VARC2Model):

    def decode_frame(self, patches: torch.Tensor) -> torch.Tensor:
        h = patches
        for blk in self.dec:
            h = blk(h, self.rope, num_prefix=0)
        h = self.dec_norm(h)
        return self._unpatchify_colors(self.wm_head(h))

    def step_latent(self, z_patches: torch.Tensor, action_ids: torch.Tensor, action_xs: torch.Tensor, action_ys: torch.Tensor) -> torch.Tensor:
        c = self.action_type_embed(action_ids.long())
        h = self._click_delta_add(z_patches, action_ids, action_xs, action_ys)
        for blk in self.dyn:
            h = blk(h, c, self.rope)
        return h

    def forward(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids: torch.Tensor, action_xs=None, action_ys=None, with_recon: bool=False):
        squeeze = action_ids.dim() == 1
        a_seq = action_ids.unsqueeze(1) if squeeze else action_ids
        if action_xs is None:
            action_xs = torch.full_like(a_seq, -1)
        x_seq = action_xs.unsqueeze(1) if action_xs.dim() == 1 else action_xs
        if action_ys is None:
            action_ys = torch.full_like(a_seq, -1)
        y_seq = action_ys.unsqueeze(1) if action_ys.dim() == 1 else action_ys
        z = self.encode(pixel_values, task_ids)
        zd = z.detach()
        out = {'type_logits': self.bc_type_head(zd[:, 0]), 'click_logits': self._unpatchify_scalar(self.bc_click_head(zd[:, 1:])), 'value': torch.sigmoid(self.value_head(zd[:, 0]).squeeze(-1))}
        if with_recon:
            out['recon_logits'] = self.decode_frame(z[:, 1:])
        h = z[:, 1:]
        ffp_list = []
        for k in range(a_seq.size(1)):
            h = self.step_latent(h, a_seq[:, k], x_seq[:, k], y_seq[:, k])
            ffp_list.append(self.decode_frame(h))
        out['ffp_logits'] = ffp_list[0]
        out['ffp_logits_list'] = ffp_list
        return out

    def forward_dynamics_multistep(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids_seq: torch.Tensor, action_xs_seq: torch.Tensor, action_ys_seq: torch.Tensor, with_recon: bool=False):
        z = self.encode(pixel_values, task_ids)
        h = z[:, 1:]
        outs = []
        for k in range(action_ids_seq.size(1)):
            h = self.step_latent(h, action_ids_seq[:, k], action_xs_seq[:, k], action_ys_seq[:, k])
            outs.append(self.decode_frame(h))
        if with_recon:
            return (outs, self.decode_frame(z[:, 1:]))
        return outs
POLICY_DEFAULTS = dict(embed_dim=512, enc_layers=7, wm_layers=3, policy_layers=3, value_layers=3, num_heads=8, mlp_dim=512, dropout=0.1, num_task_tokens=1, num_actions=8, max_mask_ratio=0.0)

def build_arch(arch: str, num_tasks: int, *, preset: str='s', patch_size: int=2, device=None, **overrides) -> Tuple[nn.Module, Dict[str, Any]]:
    ov = {k: v for k, v in overrides.items() if v is not None}
    if arch == 'varc2':
        kwargs = dict(PRESETS[preset])
        kwargs.update(patch_size=patch_size, **ov)
        model = VARC2ModelMS(num_tasks, **kwargs)
        if device is not None:
            model = model.to(device)
        resolved = dict(kwargs)
        resolved.update(num_tasks=num_tasks, arch=arch, preset=preset)
        return (model, resolved)
    if arch == 'policy':
        from wm_core.src.policy_model import ARCViTPolicyWMAdaLN
        cfg = dict(POLICY_DEFAULTS)
        if 'embed_dim' in ov:
            cfg['embed_dim'] = ov['embed_dim']
        if 'enc_layers' in ov:
            cfg['enc_layers'] = ov['enc_layers']
        if 'dyn_layers' in ov:
            cfg['wm_layers'] = ov['dyn_layers']
        if 'dec_layers' in ov:
            cfg['policy_layers'] = ov['dec_layers']
            cfg['value_layers'] = ov['dec_layers']
        if 'num_heads' in ov:
            cfg['num_heads'] = ov['num_heads']
        if 'mlp_ratio' in ov:
            cfg['mlp_dim'] = int(cfg['embed_dim'] * ov['mlp_ratio'])
        model = ARCViTPolicyWMAdaLN(num_tasks=num_tasks, patch_size=patch_size, **cfg)
        model.mask_token.requires_grad_(False)
        if device is not None:
            model = model.to(device)
        resolved = dict(cfg)
        resolved.update(patch_size=patch_size, num_tasks=num_tasks, arch=arch, conditioning='adaln_zero', attention='sdpa')
        return (model, resolved)
    raise ValueError(f'unknown arch {arch!r}; choose from {ARCH_CHOICES}')

def head_modules(model: nn.Module, arch: str) -> List[nn.Module]:
    if arch == 'varc2':
        return list(model.policy_value_modules())
    return [model.policy, model.policy_norm, model.bc_type_head, model.bc_click_head, model.value, model.value_norm, model.value_head]

def split_param_groups(model: nn.Module, arch: str):
    head_ids = {id(p) for m in head_modules(model, arch) for p in m.parameters()}
    dyn, head = ([], [])
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head if id(p) in head_ids else dyn).append(p)
    return (dyn, head)

def freeze_for_ttt(model: nn.Module, arch: str) -> List[torch.nn.Parameter]:
    if arch == 'policy':
        from policy_wm.src.ttt import freeze_policy_side, trainable_parameters
        freeze_policy_side(model)
        return trainable_parameters(model)
    for m in head_modules(model, arch):
        m.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]
MUON_STACKS = {'varc2': ('enc', 'dyn', 'dec'), 'policy': ('enc', 'wm', 'policy', 'value')}

def split_muon_param_names(model: nn.Module, arch: str):
    stacks = MUON_STACKS[arch]
    muon_names, adamw_names = ([], [])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_muon = p.ndim == 2 and name.endswith('.weight') and (name.split('.', 1)[0] in stacks) and ('ada_mod' not in name)
        (muon_names if is_muon else adamw_names).append(name)
    assert muon_names, f'muon group is empty for arch={arch} -- stack naming changed?'
    return (muon_names, adamw_names)

def build_optimizers(model: nn.Module, arch: str, *, optimizer: str, lr: float, weight_decay: float, muon_lr: float=0.0, muon_momentum: float=0.95, muon_adjust_lr: str='match_rms_adamw'):
    if optimizer == 'adamw':
        trainable = [p for p in model.parameters() if p.requires_grad]
        return [torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)]
    muon_names, adamw_names = split_muon_param_names(model, arch)
    params = dict(model.named_parameters())
    muon = Muon([params[n] for n in muon_names], lr=muon_lr if muon_lr > 0 else lr, weight_decay=weight_decay, momentum=muon_momentum, adjust_lr_fn=muon_adjust_lr)
    adamw = torch.optim.AdamW([params[n] for n in adamw_names], lr=lr, weight_decay=weight_decay)
    return [muon, adamw]

def build_from_wm_core_checkpoint(ckpt_path: str, *, device) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    mc = dict(config['model_config'])
    arch = mc.pop('arch', config.get('arch', 'varc2'))
    num_tasks = mc.pop('num_tasks')
    preset = mc.pop('preset', 's')
    patch_size = mc.pop('patch_size')
    if arch == 'varc2':
        model, _ = build_arch(arch, num_tasks, preset=preset, patch_size=patch_size, device=device, **{k: mc.get(k) for k in ('embed_dim', 'enc_layers', 'dyn_layers', 'dec_layers', 'num_heads', 'mlp_ratio')})
    else:
        from wm_core.src.policy_model import ARCViTPolicyWMAdaLN
        model = ARCViTPolicyWMAdaLN(num_tasks=num_tasks, patch_size=patch_size, **{k: mc[k] for k in POLICY_DEFAULTS if k in mc})
        model = model.to(device)
    state = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != 'model_state'}
    meta['arch'] = arch
    return (model, meta)
