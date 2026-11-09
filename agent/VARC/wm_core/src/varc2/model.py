from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from varc2.constants import CLICK_ACTION_ID, FRAME_SIZE, NUM_ACTIONS, NUM_COLORS
from varc2.rope import Rope2D
PRESETS = {'s': dict(embed_dim=512, enc_layers=6, dyn_layers=3, dec_layers=3, num_heads=8), 'l': dict(embed_dim=768, enc_layers=12, dyn_layers=6, dec_layers=6, num_heads=12)}

class Mlp(nn.Module):

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: Rope2D, num_prefix: int=0) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = rope(q, num_prefix)
        k = rope(k, num_prefix)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))

class Block(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-06)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-06)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, rope: Rope2D, num_prefix: int=0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, num_prefix)
        x = x + self.mlp(self.norm2(x))
        return x

class AdaLNBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-06, elementwise_affine=False)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ada_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: Rope2D) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_mod(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(h, rope, 0)
        h = self.norm2(x) * (1 + scale_m) + shift_m
        x = x + gate_m * self.mlp(h)
        return x

class VARC2Model(nn.Module):

    def __init__(self, num_tasks: int, *, image_size: int=FRAME_SIZE, num_colors: int=NUM_COLORS, patch_size: int=2, embed_dim: int=512, enc_layers: int=6, dyn_layers: int=3, dec_layers: int=3, num_heads: int=8, mlp_ratio: float=1.0, num_actions: int=NUM_ACTIONS, click_action_id: int=CLICK_ACTION_ID, rope_theta: float=10000.0) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError('image_size must be divisible by patch_size')
        head_dim = embed_dim // num_heads
        if embed_dim % num_heads != 0 or head_dim % 4 != 0:
            raise ValueError('embed_dim / num_heads must be an integer divisible by 4')
        self.num_tasks = num_tasks
        self.image_size = image_size
        self.num_colors = num_colors
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.click_action_id = click_action_id
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.num_task_tokens = 1
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim)
        self.action_type_embed = nn.Embedding(num_actions, embed_dim)
        self.click_embed = nn.Embedding(patch_size * patch_size, embed_dim)
        self.patch_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.rope = Rope2D(head_dim, self.grid, theta=rope_theta)
        self.enc = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(enc_layers)])
        self.dyn = nn.ModuleList([AdaLNBlock(embed_dim, num_heads, mlp_ratio) for _ in range(dyn_layers)])
        self.dec = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(dec_layers)])
        self.enc_norm = nn.RMSNorm(embed_dim, eps=1e-06)
        self.dec_norm = nn.RMSNorm(embed_dim, eps=1e-06)
        self.wm_head = nn.Linear(embed_dim, num_colors * patch_size * patch_size)
        self.bc_type_head = nn.Linear(embed_dim, num_actions - 1)
        self.bc_click_head = nn.Linear(embed_dim, patch_size * patch_size)
        self.value_head = nn.Linear(embed_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.dyn:
            nn.init.zeros_(blk.ada_mod[1].weight)
            nn.init.zeros_(blk.ada_mod[1].bias)

    def encode(self, pixel_values: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        x = self.color_embed(pixel_values.long())
        x = self.patch_proj(x.permute(0, 3, 1, 2))
        x = x.flatten(2).transpose(1, 2)
        t = self.task_token_embed(task_ids.long()).unsqueeze(1)
        x = torch.cat([t, x], dim=1)
        for blk in self.enc:
            x = blk(x, self.rope, num_prefix=1)
        return self.enc_norm(x)

    def _click_delta_add(self, patches: torch.Tensor, action_ids: torch.Tensor, action_xs: torch.Tensor, action_ys: torch.Tensor) -> torch.Tensor:
        p, g = (self.patch_size, self.grid)
        is_click = (action_ids == self.click_action_id) & (action_xs >= 0)
        sx = action_xs.clamp(min=0)
        sy = action_ys.clamp(min=0)
        patch_idx = sy // p * g + sx // p
        sub_idx = sy % p * p + sx % p
        vec = self.click_embed(sub_idx) * is_click.to(patches.dtype).unsqueeze(-1)
        idx = patch_idx.view(-1, 1, 1).expand(-1, 1, patches.size(-1))
        return patches.scatter_add(1, idx, vec.unsqueeze(1).to(patches.dtype))

    def forward_dynamics_from_z(self, z_patches: torch.Tensor, action_ids: torch.Tensor, action_xs: torch.Tensor, action_ys: torch.Tensor) -> torch.Tensor:
        c = self.action_type_embed(action_ids.long())
        h = self._click_delta_add(z_patches, action_ids, action_xs, action_ys)
        for blk in self.dyn:
            h = blk(h, c, self.rope)
        for blk in self.dec:
            h = blk(h, self.rope, num_prefix=0)
        h = self.dec_norm(h)
        return self._unpatchify_colors(self.wm_head(h))

    def _unpatchify_colors(self, logits: torch.Tensor) -> torch.Tensor:
        B = logits.size(0)
        ps, g = (self.patch_size, self.grid)
        logits = logits.reshape(-1, g, g, ps, ps, self.num_colors)
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(B, self.image_size, self.image_size, self.num_colors)
        return logits.permute(0, 3, 1, 2)

    def _unpatchify_scalar(self, click: torch.Tensor) -> torch.Tensor:
        B = click.size(0)
        ps, g = (self.patch_size, self.grid)
        click = click.reshape(-1, g, g, ps, ps)
        click = click.permute((0, 1, 3, 2, 4))
        return click.reshape(B, self.image_size, self.image_size)

    def forward(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        if action_xs is None:
            action_xs = torch.full_like(action_ids, -1)
        if action_ys is None:
            action_ys = torch.full_like(action_ids, -1)
        z = self.encode(pixel_values, task_ids)
        zd = z.detach()
        type_logits = self.bc_type_head(zd[:, 0])
        click_logits = self._unpatchify_scalar(self.bc_click_head(zd[:, 1:]))
        value = torch.sigmoid(self.value_head(zd[:, 0]).squeeze(-1))
        ffp_logits = self.forward_dynamics_from_z(z[:, 1:], action_ids, action_xs, action_ys)
        return {'ffp_logits': ffp_logits, 'type_logits': type_logits, 'click_logits': click_logits, 'value': value}

    def forward_dynamics(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None) -> torch.Tensor:
        if action_xs is None:
            action_xs = torch.full_like(action_ids, -1)
        if action_ys is None:
            action_ys = torch.full_like(action_ids, -1)
        z = self.encode(pixel_values, task_ids)
        return self.forward_dynamics_from_z(z[:, 1:], action_ids, action_xs, action_ys)

    @torch.no_grad()
    def predict_logits(self, pixel_values: torch.Tensor, task_ids: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if task_ids is None:
            task_ids = torch.zeros(pixel_values.size(0), dtype=torch.long, device=pixel_values.device)
        z = self.encode(pixel_values, task_ids)
        pol = self.policy_head(z)
        return (pol['type_logits'], pol['click_logits'])

    @torch.no_grad()
    def predict_value(self, pixel_values: torch.Tensor, task_ids: Optional[torch.Tensor]=None) -> torch.Tensor:
        if task_ids is None:
            task_ids = torch.zeros(pixel_values.size(0), dtype=torch.long, device=pixel_values.device)
        return self.value_head_forward(self.encode(pixel_values, task_ids))

    def policy_head(self, z: torch.Tensor, kpm=None) -> Dict[str, torch.Tensor]:
        return {'type_logits': self.bc_type_head(z[:, 0]), 'click_logits': self._unpatchify_scalar(self.bc_click_head(z[:, 1:]))}

    def value_head_forward(self, z: torch.Tensor, kpm=None) -> torch.Tensor:
        return torch.sigmoid(self.value_head(z[:, 0]).squeeze(-1))

    def policy_value_modules(self) -> Iterable[nn.Module]:
        return [self.bc_type_head, self.bc_click_head, self.value_head]

    def dynamics_modules(self) -> Iterable[nn.Module]:
        return [self.dyn, self.dec, self.dec_norm, self.wm_head]

    def encoder_modules(self) -> Iterable[nn.Module]:
        return [self.enc, self.enc_norm]

    def input_embed_modules(self) -> Iterable[nn.Module]:
        return [self.color_embed, self.action_type_embed, self.click_embed, self.patch_proj]

def build_model(num_tasks: int, preset: str='s', device=None, **overrides) -> VARC2Model:
    if preset not in PRESETS:
        raise ValueError(f'unknown preset {preset!r}; choose from {sorted(PRESETS)}')
    kwargs = dict(PRESETS[preset])
    kwargs.update(overrides)
    model = VARC2Model(num_tasks, **kwargs)
    if device is not None:
        model = model.to(device)
    return model
