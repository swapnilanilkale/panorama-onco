"""A small transformer text encoder for radiology reports.

Reuses the same TransformerBlock as the vision encoder -- one attention
implementation, tested once. Swappable for a medical LLM later: anything that
maps token ids to a fixed-size embedding satisfies the same interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from panorama.vision.blocks import TransformerBlock


class ReportTextEncoder(nn.Module):
    """token ids [B, L] -> embedding [B, E]."""

    def __init__(self, vocab_size: int, embed_dim: int = 256, depth: int = 4,
                 num_heads: int = 8, max_length: int = 192,
                 dropout: float = 0.0, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, token_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (token_ids != self.pad_id)
        keep = attention_mask.bool()

        # SDPA boolean convention: True = attend, False = block.
        # [B, L] -> [B, 1, 1, L] so every query blocks the same key positions.
        attn_mask = keep[:, None, None, :]

        x = self.token_embed(token_ids) + self.pos_embed[:, :token_ids.shape[1]]
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)

        # Masked mean over real tokens only.
        weights = keep.unsqueeze(-1).float()
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)