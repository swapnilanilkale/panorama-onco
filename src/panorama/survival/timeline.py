"""Timeline encoder: a patient's sequence of studies -> one risk score.

Two things this must get right that a per-study model does not face.

TIME GAPS. Visits are irregular. The same lesion measurements over 30 days and
over 400 days imply completely different growth rates -- 4.0 vs 0.7 mm/month in
a worked example -- and sequence position alone cannot distinguish them. Elapsed
days are encoded continuously so unseen intervals interpolate.

VARIABLE LENGTH. Patients have 2-4 studies. Averaging over padded positions
pulls short timelines toward zero, so timeline LENGTH leaks into the risk score
-- and length correlates directly with the outcome, because a patient with four
visits lived long enough to have four visits. That is a confound with the label.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from panorama.vision.blocks import TransformerBlock


class TimeEncoding(nn.Module):
    """Sinusoidal encoding of elapsed days since baseline.

    Continuous rather than a learned lookup, so an interval never seen in
    training interpolates instead of falling out of vocabulary -- follow-up
    schedules vary between sites and trials.
    """

    def __init__(self, dim: int, max_period: float = 1000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"dim must be even, got {dim}")
        frequencies = torch.exp(
            torch.linspace(0.0, -math.log(max_period), dim // 2))
        self.register_buffer("frequencies", frequencies)

    def forward(self, days: torch.Tensor) -> torch.Tensor:
        angles = days.unsqueeze(-1) * self.frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class TimelineEncoder(nn.Module):
    """Study embeddings [B, T, E] + elapsed days [B, T] -> risk [B].

    Higher risk means SHORTER survival, matching the Cox partial likelihood's
    sign convention.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, depth: int = 2,
                 num_heads: int = 8, time_dim: int = 32,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.time_encoding = TimeEncoding(time_dim)
        self.project = nn.Linear(embed_dim + time_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.risk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor, days: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """`mask` is 1 for real studies, 0 for padding."""
        time = self.time_encoding(days.float())
        x = self.project(torch.cat([embeddings, time], dim=-1))

        # Padding must not reach attention, or a short timeline's risk depends
        # on how much padding follows it (see M3.8 for the same fix in text).
        keep = mask.bool()
        for block in self.blocks:
            x = block(x, keep[:, None, None, :])
        x = self.norm(x)

        weights = keep.unsqueeze(-1).float()
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.risk(pooled).squeeze(-1)

       


def collate_timelines(embeddings: list[torch.Tensor],
                      days: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Pad variable-length timelines into a batch, with a validity mask."""
    batch = len(embeddings)
    longest = max(e.shape[0] for e in embeddings)
    dim = embeddings[0].shape[1]

    padded = torch.zeros(batch, longest, dim)
    padded_days = torch.zeros(batch, longest)
    mask = torch.zeros(batch, longest)
    for i, (e, d) in enumerate(zip(embeddings, days)):
        n = e.shape[0]
        padded[i, :n] = e
        padded_days[i, :n] = d
        mask[i, :n] = 1.0
    return padded, padded_days, mask