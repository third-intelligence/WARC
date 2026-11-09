from __future__ import annotations
import torch
from torch import nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = (x[..., 0::2], x[..., 1::2])
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class Rope2D(nn.Module):

    def __init__(self, dim_head: int, grid_size: int, theta: float=10000.0) -> None:
        super().__init__()
        if dim_head % 4 != 0:
            raise ValueError(f'dim_head must be divisible by 4, got {dim_head}')
        self.grid_size = grid_size
        half = dim_head // 2
        freqs = 1.0 / theta ** (torch.arange(0, half, 2).float() / half)
        f = torch.outer(torch.arange(grid_size).float(), freqs)
        f = f.repeat_interleave(2, dim=-1)
        fh = f[:, None, :].expand(grid_size, grid_size, half)
        fw = f[None, :, :].expand(grid_size, grid_size, half)
        table = torch.cat([fh, fw], dim=-1).reshape(grid_size * grid_size, dim_head)
        self.register_buffer('cos', table.cos(), persistent=False)
        self.register_buffer('sin', table.sin(), persistent=False)

    def forward(self, t: torch.Tensor, num_prefix: int=0) -> torch.Tensor:
        cos = self.cos.to(t.dtype)
        sin = self.sin.to(t.dtype)
        tr = t[:, :, num_prefix:]
        tr = tr * cos + rotate_half(tr) * sin
        if num_prefix == 0:
            return tr
        return torch.cat((t[:, :, :num_prefix], tr), dim=2)
