from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
_PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))
from vit_vendor.vendor.arc_vit import MultiHeadSelfAttention
from policy_wm.src.model import ARCViTPolicyWM

class SDPASelfAttention(MultiHeadSelfAttention):

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None, rope_index: Optional[torch.Tensor]=None) -> torch.Tensor:
        if key_padding_mask is not None or attn_mask is not None or rope_index is not None:
            return super().forward(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope_index=rope_index)
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        q = self.rotary(q)
        k = self.rotary(k)
        context = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.proj_dropout(self.proj(context))

def convert_attention_to_sdpa(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if type(m) is MultiHeadSelfAttention:
            m.__class__ = SDPASelfAttention
            n += 1
    return n

class AdaLNEncoderLayer(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, max_seq_len: int, no_rope: int) -> None:
        super().__init__()
        self.self_attn = SDPASelfAttention(embed_dim=embed_dim, num_heads=num_heads, max_seq_len=max_seq_len, dropout=dropout, no_rope=no_rope)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.ada_mod = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim))
        nn.init.zeros_(self.ada_mod[1].weight)
        nn.init.zeros_(self.ada_mod[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_mod(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.dropout1(self.self_attn(h, key_padding_mask=key_padding_mask))
        h = self.norm2(x) * (1 + scale_m) + shift_m
        x = x + gate_m * self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(h)))))
        return x

class ARCViTPolicyWMAdaLN(ARCViTPolicyWM):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        old = self.wm.layers
        depth = len(old)
        mlp_dim = old[0].linear1.out_features
        num_heads = old[0].self_attn.num_heads
        dropout = old[0].dropout1.p
        total_seq_len = self.num_prefix + self.seq_length
        self.wm = nn.ModuleList([AdaLNEncoderLayer(self.embed_dim, num_heads, mlp_dim, dropout, max_seq_len=total_seq_len, no_rope=self.num_prefix) for _ in range(depth)])
        n = convert_attention_to_sdpa(self)
        self._sdpa_converted = n

    def wm_head_forward(self, z: torch.Tensor, action_ids: torch.Tensor, action_xs: Optional[torch.Tensor], action_ys: Optional[torch.Tensor], kpm: Optional[torch.Tensor]) -> torch.Tensor:
        c = self._build_action_token(action_ids, action_xs, action_ys).squeeze(1)
        h = z
        for blk in self.wm:
            h = blk(h, c, key_padding_mask=kpm)
        h = self.wm_norm(h)
        return self._ffp_logits(h[:, self.num_prefix:])
