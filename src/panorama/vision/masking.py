from __future__ import annotations

import torch


def patchify(volume: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Volume -> flat per-token voxel targets.

    [B, C, D, H, W] -> [B, C, N, patch_size**3]

    Token order MUST match PatchEmbed3D's `flatten(2)`: the depth axis varies
    slowest and the width axis fastest. Get this wrong and every prediction is
    scored against the wrong voxels -- the loss still decreases, so the bug is
    invisible.
    """
    b, c, d, h, w = volume.shape
    p = patch_size
    if d % p or h % p or w % p:
        raise ValueError(f"volume {(d, h, w)} not divisible by patch_size {p}")
    gd, gh, gw = d // p, h // p, w // p
    x = volume.reshape(b, c, gd, p, gh, p, gw, p)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)          # [B, C, gd, gh, gw, p, p, p]
    return x.reshape(b, c, gd * gh * gw, p ** 3)


def unpatchify(tokens: torch.Tensor, patch_size: int,
               grid: tuple[int, int, int]) -> torch.Tensor:
    """Inverse of `patchify` -- for visualizing reconstructions."""
    b, c, n, _ = tokens.shape
    p = patch_size
    gd, gh, gw = grid
    if n != gd * gh * gw:
        raise ValueError(f"{n} tokens does not match grid {grid}")
    x = tokens.reshape(b, c, gd, gh, gw, p, p, p)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7)
    return x.reshape(b, c, gd * p, gh * p, gw * p)


def random_token_mask(batch_size: int, num_streams: int, num_tokens: int,
                      mask_ratio: float = 0.75,
                      device: torch.device | str = "cpu",
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Boolean mask [B, S, N]; True = hidden from the encoder.

    Exactly `round(N * mask_ratio)` tokens are masked per (sample, stream), and
    each stream is masked INDEPENDENTLY -- see the module docstring rationale.
    """
    k = int(round(num_tokens * mask_ratio))
    noise = torch.rand(batch_size, num_streams, num_tokens,
                       device=device, generator=generator)
    order = noise.argsort(dim=-1)                       # random permutation per row
    mask = torch.zeros(batch_size, num_streams, num_tokens,
                       dtype=torch.bool, device=device)
    return mask.scatter(-1, order[..., :k], True)