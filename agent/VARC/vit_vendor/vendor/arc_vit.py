from typing import Optional, Tuple
from vit_vendor.vendor.pos_embed import VisionRotaryEmbeddingFast
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.vision_transformer import PatchEmbed
_MASK_NEG = -1000000000.0

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float=0.1, no_rope: int=1, rope_pt_seq_len: Optional[int]=None) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        if self.head_dim % 2 != 0:
            raise ValueError('Rotary embeddings require the head dimension to be even')
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        half_head_dim = embed_dim // num_heads // 2
        pt_seq_len = rope_pt_seq_len if rope_pt_seq_len is not None else int(max_seq_len ** 0.5)
        self.rotary = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=pt_seq_len, no_rope=no_rope)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None, rope_index: Optional[torch.Tensor]=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        if attn_mask is None and rope_index is None:
            q = self.rotary(q)
            k = self.rotary(k)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if key_padding_mask is not None:
                mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
                attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            context = torch.matmul(attn_weights, v)
            context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
            context = self.proj(context)
            context = self.proj_dropout(context)
            return context
        q = self.rotary(q, rope_index=rope_index)
        k = self.rotary(k, rope_index=rope_index)
        bias = attn_mask
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            kbias = torch.zeros(batch_size, 1, 1, seq_len, dtype=q.dtype, device=q.device)
            kbias = kbias.masked_fill(kpm, _MASK_NEG)
            bias = kbias if bias is None else bias.to(q.dtype) + kbias
        if bias is not None:
            bias = bias.to(q.dtype)
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.attn_dropout.p if self.training else 0.0)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        context = self.proj(context)
        context = self.proj_dropout(context)
        return context

class ARCTransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, max_seq_len: int, no_rope: int=1, rope_pt_seq_len: Optional[int]=None) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, max_seq_len=max_seq_len, dropout=dropout, no_rope=no_rope, rope_pt_seq_len=rope_pt_seq_len)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None, rope_index: Optional[torch.Tensor]=None) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope_index=rope_index)
        x = residual + self.dropout1(x)
        x = self.norm1(x)
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = residual + self.dropout3(x)
        x = self.norm2(x)
        return x

class ARCTransformerEncoder(nn.Module):

    def __init__(self, *, depth: int, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, max_seq_len: int, no_rope: int=0, rope_pt_seq_len: Optional[int]=None) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ARCTransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout, max_seq_len=max_seq_len, no_rope=no_rope, rope_pt_seq_len=rope_pt_seq_len) for _ in range(depth)])

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None, rope_index: Optional[torch.Tensor]=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope_index=rope_index)
        return x

class ARCViT(nn.Module):

    def __init__(self, num_tasks: int, image_size: int=30, num_colors: int=10, embed_dim: int=256, depth: int=6, num_heads: int=8, mlp_dim: int=512, dropout: float=0.1, num_task_tokens: int=1, patch_size: int=2, num_actions: int=0) -> None:
        super().__init__()
        if image_size <= 0:
            raise ValueError('`image_size` must be > 0.')
        if num_colors <= 0:
            raise ValueError('`num_colors` must be > 0.')
        if num_tasks <= 0:
            raise ValueError('`num_tasks` must be > 0.')
        self.image_size = image_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        if patch_size is None:
            self.seq_length = image_size * image_size
        else:
            self.seq_length = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        print(f'Patch size: {self.patch_size}, sequence length: {self.seq_length}')
        self.num_task_tokens = num_task_tokens
        self.num_actions = num_actions
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, embed_dim * self.num_task_tokens)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        if num_actions > 0:
            self.action_type_embed = nn.Embedding(num_actions, embed_dim)
            self.action_coord_embed_x = nn.Embedding(64, embed_dim // 2)
            self.action_coord_embed_y = nn.Embedding(64, embed_dim // 2)
        num_prefix = self.num_task_tokens + (1 if num_actions > 0 else 0)
        total_seq_len = num_prefix + self.seq_length
        self.positional_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        self.encoder = ARCTransformerEncoder(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, max_seq_len=total_seq_len, no_rope=num_prefix)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * (1 if patch_size is None else patch_size) ** 2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if self.num_actions > 0:
            nn.init.trunc_normal_(self.action_type_embed.weight, std=0.02)
            nn.init.trunc_normal_(self.action_coord_embed_x.weight, std=0.02)
            nn.init.trunc_normal_(self.action_coord_embed_y.weight, std=0.02)

    def forward(self, pixel_values: torch.Tensor, task_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, action_ids: Optional[torch.Tensor]=None, action_xs: Optional[torch.Tensor]=None, action_ys: Optional[torch.Tensor]=None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if pixel_values.dim() != 3:
            raise ValueError('`pixel_values` must be (batch, height, width).')
        if pixel_values.size(1) != self.image_size or pixel_values.size(2) != self.image_size:
            raise ValueError(f'`pixel_values` height/width must match configured image_size={self.image_size}. Received {pixel_values.shape[1:]}')
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        tokens = self.color_embed(pixel_values.long())
        tokens = self.patch_embed(tokens.permute((0, 3, 1, 2)))
        tokens = tokens + self.positional_embed[:, :tokens.size(1), :]
        task_tokens = self.task_token_embed(task_ids.long())
        task_tokens = task_tokens.reshape(batch_size, self.num_task_tokens, -1)
        num_prefix = self.num_task_tokens
        if self.num_actions > 0 and action_ids is not None:
            action_emb = self.action_type_embed(action_ids.long())
            if action_xs is not None and action_ys is not None:
                valid = (action_xs >= 0).float().unsqueeze(-1)
                safe_x = action_xs.clamp(min=0)
                safe_y = action_ys.clamp(min=0)
                coord_x = self.action_coord_embed_x(safe_x.long())
                coord_y = self.action_coord_embed_y(safe_y.long())
                coord_emb = torch.cat([coord_x, coord_y], dim=-1)
                action_emb = action_emb + coord_emb * valid
            action_token = action_emb.unsqueeze(1)
            hidden_states = torch.cat([task_tokens, action_token, tokens], dim=1)
            num_prefix = self.num_task_tokens + 1
        else:
            hidden_states = torch.cat([task_tokens, tokens], dim=1)
        hidden_states = self.dropout(hidden_states)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, self.image_size, self.image_size):
                raise ValueError('`attention_mask` must match pixel grid size.')
            if self.patch_size is not None:
                attention_mask = attention_mask.reshape(batch_size, self.image_size // self.patch_size, self.patch_size, self.image_size // self.patch_size, self.patch_size)
                attention_mask = torch.max(torch.max(attention_mask, dim=2)[0], dim=3)[0]
            flat_mask = attention_mask.view(batch_size, self.seq_length)
            pad_mask = ~flat_mask.bool()
            pad_mask = torch.cat([torch.zeros(batch_size, num_prefix, device=device, dtype=torch.bool), pad_mask], dim=1)
            key_padding_mask = pad_mask
        encoded = self.encoder(hidden_states, key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        pixel_states = encoded[:, num_prefix:, :]
        logits = self.head(pixel_states)
        logits = logits.reshape((-1, self.image_size // self.patch_size, self.image_size // self.patch_size, self.patch_size, self.patch_size, self.num_colors))
        logits = logits.permute((0, 1, 3, 2, 4, 5))
        logits = logits.reshape(batch_size, self.image_size, self.image_size, self.num_colors)
        logits = logits.permute(0, 3, 1, 2)
        return logits
