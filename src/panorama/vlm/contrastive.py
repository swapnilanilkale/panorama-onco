"""Contrastive image-text alignment (the MedCLIP-style objective for Aim 2).

Pulls each study's imaging embedding toward its own report and pushes it away
from every other report in the batch. After training, a scan and its findings
occupy nearby points in a shared space -- which is what lets a language model
condition on imaging.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLIP's learned temperature. We learn it too, but clamp to avoid collapse.
INIT_TEMPERATURE = 0.07
MIN_TEMPERATURE = 0.01


class ProjectionHead(nn.Module):
    """Map an encoder embedding into the shared alignment space."""

    def __init__(self, in_dim: int, out_dim: int = 256,
                 hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 normalise: similarity becomes cosine, so magnitude cannot cheat.
        return F.normalize(self.norm(self.net(x)), dim=-1)


class ContrastiveAlignment(nn.Module):
    """Symmetric InfoNCE over a batch of (image, report) pairs."""

    def __init__(self, image_dim: int, text_dim: int, embed_dim: int = 256,
                 dropout: float = 0.0,
                 init_temperature: float = INIT_TEMPERATURE) -> None:
        super().__init__()
        self.image_proj = ProjectionHead(image_dim, embed_dim, dropout=dropout)
        self.text_proj = ProjectionHead(text_dim, embed_dim, dropout=dropout)
        # Parameterise as log(1/T) so the optimiser works in a well-scaled space
        # and the temperature can never become zero or negative.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    @property
    def temperature(self) -> float:
        return float(1.0 / self.logit_scale.exp().clamp(max=1.0 / MIN_TEMPERATURE))

    def encode(self, image_embed: torch.Tensor,
               text_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image_proj(image_embed), self.text_proj(text_embed)

    def forward(self, image_embed: torch.Tensor, text_embed: torch.Tensor,
                pair_ids: torch.Tensor | None = None) -> dict:
        img, txt = self.encode(image_embed, text_embed)
        scale = self.logit_scale.exp().clamp(max=1.0 / MIN_TEMPERATURE)
        logits = scale * img @ txt.t()                       # [B, B]

        b = logits.shape[0]
        targets = torch.arange(b, device=logits.device)

        if pair_ids is not None:
            # Two crops of the SAME study are not negatives of each other.
            # Mask those off-diagonal entries so they cannot be pushed apart.
            same = pair_ids[:, None] == pair_ids[None, :]
            same.fill_diagonal_(False)
            logits = logits.masked_fill(same, float("-inf"))

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)

        return {"loss": 0.5 * (loss_i2t + loss_t2i),
                "loss_i2t": loss_i2t, "loss_t2i": loss_t2i,
                "logits": logits, "image_embed": img, "text_embed": txt}


@torch.no_grad()
def retrieval_metrics(logits: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict:
    """Recall@k and median rank in both directions.

    The loss is not interpretable across batch sizes; recall is. Chance R@1
    is 1/batch_size, so always report it alongside.
    """
    b = logits.shape[0]
    targets = torch.arange(b, device=logits.device)
    out: dict[str, float] = {"chance_r1": 1.0 / b}

    for name, mat in (("i2t", logits), ("t2i", logits.t())):
        ranks = (mat > mat.gather(1, targets[:, None])).sum(dim=1)   # 0 = correct
        for k in ks:
            if k <= b:
                out[f"{name}_r{k}"] = float((ranks < k).float().mean())
        out[f"{name}_median_rank"] = float(ranks.float().median() + 1)
    return out