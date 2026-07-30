from __future__ import annotations

import torch
import torch.nn as nn

from panorama.core.constants import Modality
from panorama.core.exceptions import ModelBuildError
from panorama.vision.patch_embed import PatchEmbed3D


class MultiStreamPatchEmbed(nn.Module):
    """Embed CT / MRI / PET into a shared token space.

    Input : image [B, 3, D, H, W], modality_mask [B, 3]
    Output: tokens [B, 3, N, E]
    """

    def __init__(self,
                 volume_shape: tuple[int, int, int] = (96, 96, 96),
                 patch_size: int = 16,
                 embed_dim: int = 768) -> None:
        super().__init__()
        self.streams = Modality.imaging_streams()
        self.embed_dim = embed_dim

        # One projection per modality: different physics, different weights.
        self.embeds = nn.ModuleDict({
            m.value: PatchEmbed3D(1, patch_size, embed_dim) for m in self.streams
        })

        n_tokens = self.embeds[self.streams[0].value].num_patches(volume_shape)
        self.num_tokens = n_tokens

        # SHARED across modalities: token i is the same physical sub-cube in
        # every stream (world-anchored cropping guarantees this).
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, n_tokens, embed_dim))

        # Tells the model which stream a token came from.
        self.modality_embed = nn.Parameter(torch.zeros(1, len(self.streams), 1, embed_dim))

        # Substituted wholesale for an absent stream.
        self.missing_token = nn.Parameter(torch.zeros(embed_dim))

        for p in (self.pos_embed, self.modality_embed, self.missing_token):
            nn.init.trunc_normal_(p, std=0.02)

    def forward(self, image: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != len(self.streams):
            raise ModelBuildError(
                f"expected {len(self.streams)} channels (CT,MRI,PET), got {image.shape[1]}"
            )
        b = image.shape[0]

        # Embed every stream. Branch-free: samples in a batch have different
        # missing patterns, so we compute all and mask afterwards.
        per_stream = [self.embeds[m.value](image[:, i:i + 1])
                      for i, m in enumerate(self.streams)]
        tokens = torch.stack(per_stream, dim=1)              # [B, S, N, E]

        # Replace absent streams with the learnable missing token.
        present = modality_mask.view(b, len(self.streams), 1, 1)
        tokens = present * tokens + (1.0 - present) * self.missing_token

        # Where am I (shared) + what am I (per-stream).
        tokens = tokens + self.pos_embed + self.modality_embed
        return tokens


def modality_dropout(modality_mask: torch.Tensor, p: float,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly hide present streams during training; never hide the last one."""
    if p <= 0.0:
        return modality_mask
    mask = modality_mask.clone()
    drop = (torch.rand(mask.shape, generator=generator, device=mask.device) < p)
    candidate = mask * (~drop).float()
    # Any sample left with nothing keeps its original mask.
    empty = candidate.sum(dim=1, keepdim=True) == 0
    return torch.where(empty, mask, candidate)