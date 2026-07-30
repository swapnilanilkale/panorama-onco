from __future__ import annotations

import torch
import torch.nn as nn

from panorama.core.exceptions import ModelBuildError


class PatchEmbed3D(nn.Module):
    """Split a 3D volume into non-overlapping cubes and project each to a token.

    Input : [B, C, D, H, W]
    Output: [B, N, embed_dim]   where N = (D/p) * (H/p) * (W/p)
    """

    def __init__(self, in_channels: int = 1, patch_size: int = 16,
                 embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # kernel == stride  =>  each output element sees exactly one
        # non-overlapping cube. This is patchify and projection in one op.
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def grid_size(self, volume_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        p = self.patch_size
        if any(s % p for s in volume_shape):
            raise ModelBuildError(
                f"volume {volume_shape} not divisible by patch_size {p}; "
                f"pad or crop to a multiple of {p}"
            )
        return tuple(s // p for s in volume_shape)

    def num_patches(self, volume_shape: tuple[int, int, int]) -> int:
        g = self.grid_size(volume_shape)
        return g[0] * g[1] * g[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ModelBuildError(f"expected [B,C,D,H,W], got shape {tuple(x.shape)}")
        self.grid_size(tuple(x.shape[2:]))          # validate divisibility
        x = self.proj(x)                            # [B, E, D/p, H/p, W/p]
        return x.flatten(2).transpose(1, 2)         # [B, N, E]