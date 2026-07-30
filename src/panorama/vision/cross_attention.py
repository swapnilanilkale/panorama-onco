from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from panorama.vision.blocks import Mlp


class CrossAttention(nn.Module):
    """Queries from one stream attend over keys/values from another.

    query   [B, Nq, E]  -- e.g. structural CT tokens
    context [B, Nc, E]  -- e.g. metabolic PET tokens
    output  [B, Nq, E]  -- query-shaped, context-informed
    """

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, nq, e = query.shape
        q = self._heads(self.q(query))                       # [B, H, Nq, hd]
        kv = self.kv(context).reshape(
            b, context.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                  # [B, H, Nc, hd]

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(b, nq, e)
        return self.proj(out)


class CrossModalFusionBlock(nn.Module):
    """One fusion step: query stream absorbs information from a context stream.

    The residual is GATED by whether the context stream exists for that sample,
    so an absent modality makes this block an exact no-op -- never a NaN.
    """

    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, num_heads, dropout=dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                context_present: torch.Tensor | None = None) -> torch.Tensor:
        fused = self.cross(self.norm_q(query), self.norm_c(context))
        if context_present is not None:
            fused = fused * context_present.view(-1, 1, 1)    # [B] -> [B,1,1]
        query = query + fused
        query = query + self.mlp(self.norm_mlp(query))
        return query