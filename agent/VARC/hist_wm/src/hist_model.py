from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
import torch
import torch.nn.functional as F
from torch import nn
from wm_core.src.policy_model import ARCViTPolicyWMAdaLN
from wm_core.src.models import POLICY_DEFAULTS
ARCH_NAME = 'policy_hist'
PAD_ID = 16
NUM_INPUT_COLORS = 17

class ARCViTPolicyWMHist(ARCViTPolicyWMAdaLN):

    def __init__(self, *args, hist_n: int=2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if hist_n < 1:
            raise ValueError(f'hist_n must be >= 1, got {hist_n}')
        self.hist_n = int(hist_n)
        d = self.embed_dim
        self.color_embed = nn.Embedding(NUM_INPUT_COLORS, d)
        self.patch_embed = None
        self.patch_embed3d = nn.Conv3d(d, d, kernel_size=(self.hist_n, self.patch_size, self.patch_size), stride=(self.hist_n, self.patch_size, self.patch_size), bias=True)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.patch_embed3d.weight, std=0.02)
        nn.init.zeros_(self.patch_embed3d.bias)
        if self.hist_n >= 2:
            self.w_hist = nn.Linear((self.hist_n - 1) * d, d, bias=False)
            nn.init.zeros_(self.w_hist.weight)
        else:
            self.w_hist = None
        self.hist_fused = True

    def _default_window(self, pixel_values: torch.Tensor):
        B = pixel_values.size(0)
        hist_obs = torch.full((B, self.hist_n, self.image_size, self.image_size), PAD_ID, dtype=torch.long, device=pixel_values.device)
        hist_obs[:, -1] = pixel_values.long()
        m = max(self.hist_n - 1, 0)
        hist_aid = torch.zeros(B, m, dtype=torch.long, device=pixel_values.device)
        hist_ax = torch.full((B, m), -1, dtype=torch.long, device=pixel_values.device)
        hist_ay = torch.full((B, m), -1, dtype=torch.long, device=pixel_values.device)
        return (hist_obs, hist_aid, hist_ax, hist_ay)

    def _hist_patch_tokens(self, hist_obs: torch.Tensor) -> torch.Tensor:
        B, n = hist_obs.shape[:2]
        d = self.embed_dim
        if self.hist_fused:
            with torch.autocast(device_type=hist_obs.device.type, enabled=False):
                wp = self.patch_embed3d.weight.float()
                e = self.color_embed.weight.float()
                wf = torch.einsum('odnyx,cd->ocnyx', wp, e)
                colors = torch.arange(NUM_INPUT_COLORS, device=hist_obs.device)
                x = hist_obs.long().unsqueeze(-1) == colors
                x = x.permute(0, 4, 1, 2, 3).float()
                x = F.conv3d(x, wf, self.patch_embed3d.bias.float(), stride=self.patch_embed3d.stride)
        else:
            x = self.color_embed(hist_obs.long())
            x = x.permute(0, 4, 1, 2, 3)
            x = self.patch_embed3d(x)
        return x.squeeze(2).flatten(2).transpose(1, 2)

    def _build_condition(self, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor], action_ys: Optional[torch.Tensor], hist_aid: Optional[torch.Tensor], hist_ax: Optional[torch.Tensor], hist_ay: Optional[torch.Tensor]) -> torch.Tensor:
        c = self._build_action_token(action_ids, action_xs, action_ys).squeeze(1)
        if self.w_hist is None or hist_aid is None or hist_aid.size(1) == 0:
            return c
        hs = [self._build_action_token(hist_aid[:, k], hist_ax[:, k], hist_ay[:, k]).squeeze(1) for k in range(hist_aid.size(1))]
        with torch.autocast(device_type=c.device.type, enabled=False):
            c_hist = self.w_hist(torch.cat(hs, dim=-1).float())
        return c + c_hist

    def _build_input(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], hist_obs: Optional[torch.Tensor]=None):
        if pixel_values.dim() != 3:
            raise ValueError('`pixel_values` must be (batch, 64, 64).')
        B = pixel_values.size(0)
        device = pixel_values.device
        if hist_obs is None:
            hist_obs = self._default_window(pixel_values)[0]
        tokens = self._hist_patch_tokens(hist_obs)
        if self.training and self.max_mask_ratio > 0.0:
            n_patch = tokens.size(1)
            ratios = torch.rand(B, 1, device=device) * self.max_mask_ratio
            drop = torch.rand(B, n_patch, device=device) < ratios
            mask_tok = self.mask_token.view(1, 1, -1).to(tokens.dtype)
            tokens = torch.where(drop.unsqueeze(-1), mask_tok, tokens)
        tokens = tokens + self.positional_embed[:, :tokens.size(1), :]
        task_tokens = self.task_token_embed(task_ids.long())
        task_tokens = task_tokens.reshape(B, self.num_task_tokens, -1)
        pad_token = self.action_pad_embed.view(1, 1, -1).expand(B, 1, -1)
        hidden_states = torch.cat([task_tokens, pad_token, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (B, self.image_size, self.image_size):
                raise ValueError('`attention_mask` must match pixel grid size.')
            am = attention_mask.reshape(B, self.image_size // self.patch_size, self.patch_size, self.image_size // self.patch_size, self.patch_size)
            am = torch.max(torch.max(am, dim=2)[0], dim=3)[0]
            flat_mask = am.view(B, self.seq_length)
            pad_mask = ~flat_mask.bool()
            pad_mask = torch.cat([torch.zeros(B, self.num_prefix, device=device, dtype=torch.bool), pad_mask], dim=1)
            key_padding_mask = pad_mask
        return (hidden_states, key_padding_mask)

    def encode(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, hist_obs: Optional[torch.Tensor]=None):
        hidden_states, kpm = self._build_input(pixel_values, task_ids, attention_mask, hist_obs=hist_obs)
        z = self.enc(hidden_states, key_padding_mask=kpm)
        return (z, kpm)

    def _wm_apply(self, z: torch.Tensor, c: torch.Tensor, kpm) -> torch.Tensor:
        h = z
        for blk in self.wm:
            h = blk(h, c, key_padding_mask=kpm)
        h = self.wm_norm(h)
        return self._ffp_logits(h[:, self.num_prefix:])

    def forward(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, hist_obs: Optional[torch.Tensor]=None, hist_action_ids: Optional[torch.Tensor]=None, hist_action_xs: Optional[torch.Tensor]=None, hist_action_ys: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        z, kpm = self.encode(pixel_values, task_ids, attention_mask, hist_obs=hist_obs)
        pol = self.policy_head(z.detach(), kpm)
        value = self.value_head_forward(z.detach(), kpm)
        c = self._build_condition(action_ids, action_xs, action_ys, hist_action_ids, hist_action_xs, hist_action_ys)
        ffp_logits = self._wm_apply(z, c, kpm)
        return {'ffp_logits': ffp_logits, 'type_logits': pol['type_logits'], 'click_logits': pol['click_logits'], 'value': value}

    def forward_dynamics(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, hist_obs: Optional[torch.Tensor]=None, hist_action_ids: Optional[torch.Tensor]=None, hist_action_xs: Optional[torch.Tensor]=None, hist_action_ys: Optional[torch.Tensor]=None) -> torch.Tensor:
        z, kpm = self.encode(pixel_values, task_ids, attention_mask, hist_obs=hist_obs)
        c = self._build_condition(action_ids, action_xs, action_ys, hist_action_ids, hist_action_xs, hist_action_ys)
        return self._wm_apply(z, c, kpm)

def build_hist_model(num_tasks: int, *, patch_size: int=2, hist_n: int=2, device=None, **overrides) -> Tuple[nn.Module, Dict]:
    ov = {k: v for k, v in overrides.items() if v is not None}
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
    model = ARCViTPolicyWMHist(num_tasks=num_tasks, patch_size=patch_size, hist_n=hist_n, **cfg)
    model.mask_token.requires_grad_(False)
    if device is not None:
        model = model.to(device)
    resolved = dict(cfg)
    resolved.update(patch_size=patch_size, num_tasks=num_tasks, arch=ARCH_NAME, hist_n=hist_n, num_input_colors=NUM_INPUT_COLORS, conditioning='adaln_zero', attention='sdpa')
    return (model, resolved)

def build_from_hist_wm_checkpoint(ckpt_path: str, *, device) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    mc = dict(config['model_config'])
    arch = mc.pop('arch', ARCH_NAME)
    if arch != ARCH_NAME:
        raise ValueError(f'not an hist_wm checkpoint (arch={arch!r})')
    num_tasks = mc.pop('num_tasks')
    patch_size = mc.pop('patch_size')
    hist_n = mc.pop('hist_n', 2)
    for k in ('num_input_colors', 'conditioning', 'attention'):
        mc.pop(k, None)
    model = ARCViTPolicyWMHist(num_tasks=num_tasks, patch_size=patch_size, hist_n=hist_n, **{k: mc[k] for k in POLICY_DEFAULTS if k in mc})
    model = model.to(device)
    state = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != 'model_state'}
    meta['arch'] = arch
    return (model, meta)
