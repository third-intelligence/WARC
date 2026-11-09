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
from wm_core.src.policy_model import AdaLNEncoderLayer
ARCH_NAME = 'unified_hist'
PAD_ID = 16
NUM_INPUT_COLORS = 17
STACK_NAME = 'blocks'
UNIFIED_DEFAULTS = dict(embed_dim=512, depth=10, num_heads=8, mlp_dim=512, dropout=0.1, num_actions=8)

class UnifiedHistWM(nn.Module):

    def __init__(self, *, hist_n: int=3, image_size: int=64, num_colors: int=16, patch_size: int=2, embed_dim: int=512, depth: int=10, num_heads: int=8, mlp_dim: int=512, dropout: float=0.1, num_actions: int=8) -> None:
        super().__init__()
        if hist_n < 1:
            raise ValueError(f'hist_n must be >= 1, got {hist_n}')
        self.hist_n = int(hist_n)
        self.image_size = image_size
        self.num_colors = num_colors
        self.patch_size = patch_size
        self.patch_grid = image_size // patch_size
        self.seq_length = self.patch_grid ** 2
        self.embed_dim = embed_dim
        self.num_actions = num_actions
        d = embed_dim
        self.color_embed = nn.Embedding(NUM_INPUT_COLORS, d)
        self.patch_embed3d = nn.Conv3d(d, d, kernel_size=(self.hist_n, patch_size, patch_size), stride=(self.hist_n, patch_size, patch_size), bias=True)
        self.dropout = nn.Dropout(dropout)
        self.action_type_embed = nn.Embedding(num_actions, d)
        self.action_coord_embed_x = nn.Embedding(64, d // 2)
        self.action_coord_embed_y = nn.Embedding(64, d // 2)
        self.w_hist = nn.Linear((self.hist_n - 1) * d, d, bias=False) if self.hist_n >= 2 else None
        self.blocks = nn.ModuleList([AdaLNEncoderLayer(d, num_heads, mlp_dim, dropout, max_seq_len=self.seq_length, no_rope=0) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_colors * patch_size ** 2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.patch_embed3d.weight, std=0.02)
        nn.init.zeros_(self.patch_embed3d.bias)
        nn.init.trunc_normal_(self.action_type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.action_coord_embed_x.weight, std=0.02)
        nn.init.trunc_normal_(self.action_coord_embed_y.weight, std=0.02)
        if self.w_hist is not None:
            nn.init.zeros_(self.w_hist.weight)
        nn.init.zeros_(self.head.bias)

    def _default_window(self, pixel_values: torch.Tensor):
        B = pixel_values.size(0)
        dev = pixel_values.device
        hist_obs = torch.full((B, self.hist_n, self.image_size, self.image_size), PAD_ID, dtype=torch.long, device=dev)
        hist_obs[:, -1] = pixel_values.long()
        m = max(self.hist_n - 1, 0)
        return (hist_obs, torch.zeros(B, m, dtype=torch.long, device=dev), torch.full((B, m), -1, dtype=torch.long, device=dev), torch.full((B, m), -1, dtype=torch.long, device=dev))

    def _patch_tokens(self, hist_obs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=hist_obs.device.type, enabled=False):
            wp = self.patch_embed3d.weight.float()
            e = self.color_embed.weight.float()
            wf = torch.einsum('odnyx,cd->ocnyx', wp, e)
            colors = torch.arange(NUM_INPUT_COLORS, device=hist_obs.device)
            x = hist_obs.long().unsqueeze(-1) == colors
            x = x.permute(0, 4, 1, 2, 3).float()
            x = F.conv3d(x, wf, self.patch_embed3d.bias.float(), stride=self.patch_embed3d.stride)
        return x.squeeze(2).flatten(2).transpose(1, 2)

    def _action_embed(self, action_ids, action_xs, action_ys) -> torch.Tensor:
        emb = self.action_type_embed(action_ids.long())
        if action_xs is not None and action_ys is not None:
            valid = (action_xs >= 0).to(emb.dtype).unsqueeze(-1)
            coord = torch.cat([self.action_coord_embed_x(action_xs.clamp(min=0).long()), self.action_coord_embed_y(action_ys.clamp(min=0).long())], dim=-1)
            emb = emb + coord * valid
        return emb

    def _build_condition(self, action_ids, action_xs, action_ys, hist_aid, hist_ax, hist_ay) -> torch.Tensor:
        c = self._action_embed(action_ids, action_xs, action_ys)
        if self.w_hist is None or hist_aid is None or hist_aid.size(1) == 0:
            return c
        hs = [self._action_embed(hist_aid[:, k], hist_ax[:, k], hist_ay[:, k]) for k in range(hist_aid.size(1))]
        with torch.autocast(device_type=c.device.type, enabled=False):
            c_hist = self.w_hist(torch.cat(hs, dim=-1).float())
        return c + c_hist

    def _ffp_logits(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        ps, g = (self.patch_size, self.patch_grid)
        logits = self.head(h).reshape(B, g, g, ps, ps, self.num_colors)
        logits = logits.permute(0, 1, 3, 2, 4, 5).reshape(B, self.image_size, self.image_size, self.num_colors)
        return logits.permute(0, 3, 1, 2)

    def forward_dynamics(self, pixel_values: torch.Tensor, task_ids: Optional[torch.Tensor], action_ids: torch.Tensor, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, hist_obs: Optional[torch.Tensor]=None, hist_action_ids: Optional[torch.Tensor]=None, hist_action_xs: Optional[torch.Tensor]=None, hist_action_ys: Optional[torch.Tensor]=None) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError('unified_hist does not take an attention_mask')
        if pixel_values.dim() != 3:
            raise ValueError('`pixel_values` must be (batch, 64, 64).')
        if hist_obs is None:
            hist_obs, hist_action_ids, hist_action_xs, hist_action_ys = self._default_window(pixel_values)
        x = self.dropout(self._patch_tokens(hist_obs))
        c = self._build_condition(action_ids, action_xs, action_ys, hist_action_ids, hist_action_xs, hist_action_ys)
        for blk in self.blocks:
            x = blk(x, c)
        return self._ffp_logits(self.norm(x))

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return {'ffp_logits': self.forward_dynamics(*args, **kwargs)}

def build_unified_model(*, hist_n: int=3, patch_size: int=2, device=None, **overrides) -> Tuple[nn.Module, Dict]:
    cfg = dict(UNIFIED_DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    model = UnifiedHistWM(hist_n=hist_n, patch_size=patch_size, **cfg)
    if device is not None:
        model = model.to(device)
    resolved = dict(cfg, arch=ARCH_NAME, hist_n=hist_n, patch_size=patch_size, num_input_colors=NUM_INPUT_COLORS, conditioning='adaln_zero_all_layers', norm='pre', positional='rope_only', task_token=False, attention='sdpa')
    return (model, resolved)

def build_from_unified_checkpoint(ckpt_path: str, *, device) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = dict(ckpt['config']['model_config'])
    if mc.get('arch') != ARCH_NAME:
        raise ValueError(f'not an unified_wm checkpoint (arch={mc.get('arch')!r})')
    model = UnifiedHistWM(hist_n=mc['hist_n'], patch_size=mc['patch_size'], **{k: mc[k] for k in UNIFIED_DEFAULTS if k in mc}).to(device)
    state = {k.replace('_orig_mod.', '', 1): v for k, v in ckpt['model_state'].items()}
    model.load_state_dict(state)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != 'model_state'}
    meta['arch'] = ARCH_NAME
    return (model, meta)
