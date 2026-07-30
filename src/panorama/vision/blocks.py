from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head self-attention over a token sequence [B, N, E]."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, e = x.shape
        qkv = (self.qkv(x)
               .reshape(b, n, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))            # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Applies the 1/sqrt(head_dim) scale internally, and dispatches to
        # Flash Attention on GPU (never materializes the N x N matrix).
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)

        out = out.transpose(1, 2).reshape(b, n, e)  # merge heads
        return self.proj(out)


class Mlp(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, dim: int, hidden_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x