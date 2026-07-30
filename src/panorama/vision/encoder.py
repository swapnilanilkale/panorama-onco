from __future__ import annotations

import torch
import torch.nn as nn

from panorama.core.constants import Modality
from panorama.vision.blocks import TransformerBlock
from panorama.vision.cross_attention import CrossModalFusionBlock
from panorama.vision.multistream import MultiStreamPatchEmbed

STRUCTURAL = (Modality.CT, Modality.MRI)
METABOLIC = (Modality.PET,)


class MultiStreamViT(nn.Module):
    """Aim 1 encoder: per-stream self-attention interleaved with
    bidirectional structural<->metabolic cross-attention.

    Input : image [B, 3, D, H, W], modality_mask [B, 3]
    Output: tokens [B, 3, N, E]  (per-stream, per-location features)
            pooled [B, E]        (masked mean over present streams)
    """

    def __init__(self,
                 volume_shape: tuple[int, int, int] = (96, 96, 96),
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 fusion_every: int = 4,
                 share_stream_weights: bool = True) -> None:
        super().__init__()
        self.streams = Modality.imaging_streams()
        self.struct_idx = [self.streams.index(m) for m in STRUCTURAL]
        self.metab_idx = [self.streams.index(m) for m in METABOLIC]
        self.depth = depth
        self.fusion_every = fusion_every
        self.share = share_stream_weights

        self.embed = MultiStreamPatchEmbed(volume_shape, patch_size, embed_dim)
        self.num_tokens = self.embed.num_tokens

        def block() -> TransformerBlock:
            return TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)

        if share_stream_weights:
            self.blocks = nn.ModuleList([block() for _ in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                nn.ModuleList([block() for _ in self.streams]) for _ in range(depth)
            ])

        n_fusion = depth // fusion_every
        self.fuse_s2m = nn.ModuleList([                       # structural queries metabolic
            CrossModalFusionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(n_fusion)])
        self.fuse_m2s = nn.ModuleList([                       # metabolic queries structural
            CrossModalFusionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(n_fusion)])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image: torch.Tensor,
                modality_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.embed(image, modality_mask)             # [B, S, N, E]
        streams = [tokens[:, i] for i in range(len(self.streams))]

        pet_present = modality_mask[:, self.metab_idx].amax(dim=1)
        struct_present = modality_mask[:, self.struct_idx].amax(dim=1)

        fusion_i = 0
        for d in range(self.depth):
            if self.share:
                streams = [self.blocks[d](s) for s in streams]
            else:
                streams = [self.blocks[d][i](s) for i, s in enumerate(streams)]

            if (d + 1) % self.fusion_every == 0:
                metabolic = streams[self.metab_idx[0]]
                structural_ctx = torch.cat([streams[i] for i in self.struct_idx], dim=1)

                # Simultaneous update: both directions read PRE-fusion states.
                updated = list(streams)
                for i in self.struct_idx:
                    updated[i] = self.fuse_s2m[fusion_i](streams[i], metabolic, pet_present)
                for i in self.metab_idx:
                    updated[i] = self.fuse_m2s[fusion_i](streams[i], structural_ctx,
                                                         struct_present)
                streams = updated
                fusion_i += 1

        tokens = self.norm(torch.stack(streams, dim=1))        # [B, S, N, E]

        # Masked mean: absent streams must not pollute the pooled embedding.
        per_stream = tokens.mean(dim=2)                        # [B, S, E]
        weights = modality_mask.unsqueeze(-1)                  # [B, S, 1]
        denom = modality_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (per_stream * weights).sum(dim=1) / denom     # [B, E]

        return tokens, pooled