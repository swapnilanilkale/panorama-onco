from __future__ import annotations

import torch
import torch.nn as nn

from panorama.vision.blocks import TransformerBlock
from panorama.vision.encoder import MultiStreamViT
from panorama.vision.masking import patchify, random_token_mask


class MAEDecoder(nn.Module):
    """Lightweight per-stream decoder: tokens -> predicted voxels."""

    def __init__(self, embed_dim: int, patch_size: int, decoder_dim: int = 256,
                 depth: int = 2, num_heads: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, decoder_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size ** 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s, n, e = tokens.shape
        x = self.proj(tokens).reshape(b * s, n, -1)      # decode each stream
        for block in self.blocks:
            x = block(x)
        x = self.pred(self.norm(x))
        return x.reshape(b, s, n, -1)                    # [B, S, N, p^3]


class MultiModalMAE(nn.Module):
    """Self-supervised pretraining: hide voxels, reconstruct them."""

    def __init__(self, encoder: MultiStreamViT, patch_size: int = 16,
                 mask_ratio: float = 0.75, decoder_dim: int = 256,
                 decoder_depth: int = 2, decoder_heads: int = 8,
                 norm_pix_loss: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.decoder = MAEDecoder(encoder.embed.embed_dim, patch_size,
                                  decoder_dim, decoder_depth, decoder_heads)

    def forward(self, image: torch.Tensor, modality_mask: torch.Tensor,
                token_mask: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> dict:
        b, s = image.shape[0], image.shape[1]
        if token_mask is None:
            token_mask = random_token_mask(b, s, self.encoder.num_tokens,
                                           self.mask_ratio, image.device, generator)

        tokens, pooled = self.encoder(image, modality_mask, token_mask)
        pred = self.decoder(tokens)                              # [B, S, N, p^3]
        target = patchify(image, self.patch_size)                # [B, S, N, p^3]

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            std = target.std(dim=-1, keepdim=True)
            target = (target - mean) / (std + 1e-6)

        per_token = (pred - target).pow(2).mean(dim=-1)           # [B, S, N]

        # Score a token only if it was HIDDEN and its modality actually EXISTS.
        weight = token_mask.float() * modality_mask.unsqueeze(-1)
        loss = (per_token * weight).sum() / weight.sum().clamp(min=1.0)

        return {"loss": loss, "pred": pred, "target": target,
                "token_mask": token_mask, "pooled": pooled}