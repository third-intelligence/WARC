from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from vit_vendor.vendor.arc_vit import ARCTransformerEncoder
from timm.models.vision_transformer import PatchEmbed

class ARCViTPolicyWM(nn.Module):

    def __init__(self, num_tasks: int, image_size: int=64, num_colors: int=16, embed_dim: int=512, enc_layers: int=7, wm_layers: int=3, policy_layers: int=3, value_layers: int=3, num_heads: int=8, mlp_dim: int=512, dropout: float=0.1, num_task_tokens: int=1, patch_size: int=2, num_actions: int=8, max_mask_ratio: float=0.0) -> None:
        super().__init__()
        if num_actions <= 0:
            raise ValueError('ARCViTPolicyWM always has an action token, so num_actions > 0 is required.')
        if patch_size is None:
            raise ValueError('ARCViTPolicyWM requires patch_size (None is not supported).')
        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_grid = image_size // patch_size
        self.seq_length = self.patch_grid ** 2
        self.num_task_tokens = num_task_tokens
        self.num_actions = num_actions
        self.num_prefix = num_task_tokens + 1
        self.max_mask_ratio = float(max_mask_ratio)
        total_seq_len = self.num_prefix + self.seq_length
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.action_type_embed = nn.Embedding(num_actions, embed_dim)
        self.action_coord_embed_x = nn.Embedding(64, embed_dim // 2)
        self.action_coord_embed_y = nn.Embedding(64, embed_dim // 2)
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        self.action_pad_embed = nn.Parameter(torch.zeros(embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        self.dropout = nn.Dropout(dropout)
        enc_kwargs = dict(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, max_seq_len=total_seq_len, no_rope=self.num_prefix)
        self.enc = ARCTransformerEncoder(depth=enc_layers, **enc_kwargs)
        self.wm = ARCTransformerEncoder(depth=wm_layers, **enc_kwargs)
        self.policy = ARCTransformerEncoder(depth=policy_layers, **enc_kwargs)
        self.value = ARCTransformerEncoder(depth=value_layers, **enc_kwargs)
        self.wm_norm = nn.LayerNorm(embed_dim)
        self.wm_head = nn.Linear(embed_dim, num_colors * patch_size ** 2)
        self.policy_norm = nn.LayerNorm(embed_dim)
        self.bc_type_head = nn.Linear(embed_dim, num_actions - 1)
        self.bc_click_head = nn.Linear(embed_dim, patch_size ** 2)
        self.value_norm = nn.LayerNorm(embed_dim)
        self.value_head = nn.Linear(embed_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.action_type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.action_coord_embed_x.weight, std=0.02)
        nn.init.trunc_normal_(self.action_coord_embed_y.weight, std=0.02)
        nn.init.trunc_normal_(self.action_pad_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.zeros_(self.wm_head.bias)
        nn.init.trunc_normal_(self.bc_type_head.weight, std=0.02)
        nn.init.zeros_(self.bc_type_head.bias)
        nn.init.trunc_normal_(self.bc_click_head.weight, std=0.02)
        nn.init.zeros_(self.bc_click_head.bias)
        nn.init.trunc_normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def _build_action_token(self, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor], action_ys: Optional[torch.Tensor]) -> torch.Tensor:
        action_emb = self.action_type_embed(action_ids.long())
        if action_xs is not None and action_ys is not None:
            valid = (action_xs >= 0).to(action_emb.dtype).unsqueeze(-1)
            safe_x = action_xs.clamp(min=0)
            safe_y = action_ys.clamp(min=0)
            coord_x = self.action_coord_embed_x(safe_x.long())
            coord_y = self.action_coord_embed_y(safe_y.long())
            coord_emb = torch.cat([coord_x, coord_y], dim=-1)
            action_emb = action_emb + coord_emb * valid
        return action_emb.unsqueeze(1)

    def _build_input(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if pixel_values.dim() != 3:
            raise ValueError('`pixel_values` must be (batch, 64, 64).')
        B = pixel_values.size(0)
        device = pixel_values.device
        tokens = self.color_embed(pixel_values.long())
        tokens = self.patch_embed(tokens.permute((0, 3, 1, 2)))
        if self.training and self.max_mask_ratio > 0.0:
            Npatch = tokens.size(1)
            ratios = torch.rand(B, 1, device=device) * self.max_mask_ratio
            drop = torch.rand(B, Npatch, device=device) < ratios
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

    def _ffp_logits(self, patch_states: torch.Tensor) -> torch.Tensor:
        B = patch_states.size(0)
        logits = self.wm_head(patch_states)
        ps = self.patch_size
        g = self.patch_grid
        logits = logits.reshape(-1, g, g, ps, ps, self.num_colors)
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(B, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits

    def _click_logits(self, patch_states: torch.Tensor) -> torch.Tensor:
        B = patch_states.size(0)
        click = self.bc_click_head(patch_states)
        ps = self.patch_size
        g = self.patch_grid
        click = click.reshape(-1, g, g, ps, ps)
        click = click.permute((0, 1, 3, 2, 4))
        click = click.reshape(B, self.image_size, self.image_size)
        return click

    def encode(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        hidden_states, kpm = self._build_input(pixel_values, task_ids, attention_mask)
        z = self.enc(hidden_states, key_padding_mask=kpm)
        return (z, kpm)

    def policy_head(self, z: torch.Tensor, kpm: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        zp = self.policy_norm(self.policy(z, key_padding_mask=kpm))
        type_logits = self.bc_type_head(zp[:, self.num_task_tokens])
        click_logits = self._click_logits(zp[:, self.num_prefix:])
        return {'type_logits': type_logits, 'click_logits': click_logits}

    def value_head_forward(self, z: torch.Tensor, kpm: Optional[torch.Tensor]) -> torch.Tensor:
        zv = self.value_norm(self.value(z, key_padding_mask=kpm))
        return torch.sigmoid(self.value_head(zv[:, self.num_task_tokens]).squeeze(-1))

    def wm_head_forward(self, z: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor], action_ys: Optional[torch.Tensor], kpm: Optional[torch.Tensor]) -> torch.Tensor:
        act_vec = self._build_action_token(action_ids, action_xs, action_ys)
        ntt = self.num_task_tokens
        z_act = torch.cat([z[:, :ntt], act_vec, z[:, ntt + 1:]], dim=1)
        wm_out = self.wm_norm(self.wm(z_act, key_padding_mask=kpm))
        return self._ffp_logits(wm_out[:, self.num_prefix:])

    def forward(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        z, kpm = self.encode(pixel_values, task_ids, attention_mask)
        pol = self.policy_head(z.detach(), kpm)
        value = self.value_head_forward(z.detach(), kpm)
        ffp_logits = self.wm_head_forward(z, action_ids, action_xs, action_ys, kpm)
        return {'ffp_logits': ffp_logits, 'type_logits': pol['type_logits'], 'click_logits': pol['click_logits'], 'value': value}

    def forward_dynamics(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        z, kpm = self.encode(pixel_values, task_ids, attention_mask)
        return self.wm_head_forward(z, action_ids, action_xs, action_ys, kpm)

    @torch.no_grad()
    def predict_logits(self, pixel_values: torch.Tensor, task_ids: Optional[torch.Tensor]=None):
        B = pixel_values.size(0)
        if task_ids is None:
            task_ids = torch.zeros(B, dtype=torch.long, device=pixel_values.device)
        z, kpm = self.encode(pixel_values, task_ids, attention_mask=None)
        pol = self.policy_head(z, kpm)
        return (pol['type_logits'], pol['click_logits'])
