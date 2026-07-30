import torch

from panorama.vision.encoder import MultiStreamViT
from panorama.vision.cross_attention import CrossModalFusionBlock
from panorama.vision.mae import MultiModalMAE
from panorama.vision.masking import patchify, random_token_mask, unpatchify
from panorama.vision.patch_embed import PatchEmbed3D

VOL, PATCH, DIM = (32, 32, 32), 16, 64


def tiny_encoder():
    return MultiStreamViT(volume_shape=VOL, patch_size=PATCH, embed_dim=DIM,
                          depth=2, num_heads=8, fusion_every=2)


def tiny_mae():
    return MultiModalMAE(tiny_encoder(), patch_size=PATCH, mask_ratio=0.75,
                         decoder_dim=32, decoder_depth=1, decoder_heads=4)


# ---------------------------------------------------------------- masking ----

def test_patchify_order_matches_patch_embed():
    """THE critical alignment: model's patch order == target patch order.

    If these diverge, predictions are scored against the wrong voxels. The loss
    still decreases, so nothing appears broken while nothing is learned.
    """
    p = 2
    vol = torch.arange(6 ** 3, dtype=torch.float32).reshape(1, 1, 6, 6, 6)
    emb = PatchEmbed3D(in_channels=1, patch_size=p, embed_dim=p ** 3)
    with torch.no_grad():                      # make the conv read one voxel per channel
        emb.proj.weight.zero_()
        emb.proj.bias.zero_()
        for i in range(p ** 3):
            d, h, w = i // (p * p), (i // p) % p, i % p
            emb.proj.weight[i, 0, d, h, w] = 1.0
    assert torch.allclose(emb(vol), patchify(vol, p)[:, 0])


def test_patchify_roundtrip_is_exact():
    vol = torch.arange(2 * 3 * 8 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8, 8)
    assert torch.equal(unpatchify(patchify(vol, 2), 2, (4, 4, 4)), vol)


def test_mask_count_is_exact_per_stream():
    """A varying denominator would make the loss scale jitter batch to batch."""
    m = random_token_mask(4, 3, 216, 0.75, generator=torch.Generator().manual_seed(0))
    counts = m.sum(-1).flatten().tolist()
    assert set(counts) == {162}


def test_streams_are_masked_independently():
    """Cross-modal reconstruction requires positions hidden in one stream only."""
    m = random_token_mask(1, 3, 216, 0.75, generator=torch.Generator().manual_seed(0))
    ct, pet = m[0, 0], m[0, 2]
    assert int((ct & ~pet).sum()) > 0        # hidden in CT, visible in PET
    assert int((~ct & pet).sum()) > 0        # and the reverse


# ---------------------------------------------------------------- encoder ----

def test_absent_context_makes_fusion_an_identity():
    """Gating, not attention masking -- an all-masked softmax yields NaN."""
    block = CrossModalFusionBlock(DIM, 8).eval()
    q = torch.randn(2, 8, DIM)
    ctx = torch.randn(2, 27, DIM)
    with torch.no_grad():
        gated = block(q, ctx, context_present=torch.tensor([0.0, 0.0]))
        mlp_only = q + block.mlp(block.norm_mlp(q))
    assert torch.allclose(gated, mlp_only, atol=1e-5)


def test_no_nan_for_any_modality_pattern():
    enc = tiny_encoder().eval()
    for pattern in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]):
        with torch.no_grad():
            tokens, pooled = enc(torch.randn(1, 3, *VOL),
                                 torch.tensor([pattern], dtype=torch.float32))
        assert not torch.isnan(tokens).any(), pattern
        assert not torch.isnan(pooled).any(), pattern


def test_pooling_ignores_absent_streams():
    """A CT-only study must not pool to the same embedding as a tri-modal one."""
    enc = tiny_encoder().eval()
    image = torch.randn(1, 3, *VOL)
    with torch.no_grad():
        _, ct_only = enc(image, torch.tensor([[1., 0., 0.]]))
        _, tri = enc(image, torch.tensor([[1., 1., 1.]]))
    assert not torch.allclose(ct_only, tri, atol=1e-3)


def test_absent_stream_content_cannot_affect_output():
    """If MRI is flagged absent, its voxels must be entirely ignored."""
    enc = tiny_encoder().eval()
    mask = torch.tensor([[1., 0., 1.]])
    a = torch.randn(1, 3, *VOL)
    b = a.clone()
    b[:, 1] = 999.0                                  # garbage in the absent channel
    with torch.no_grad():
        _, pooled_a = enc(a, mask)
        _, pooled_b = enc(b, mask)
    assert torch.allclose(pooled_a, pooled_b, atol=1e-5)


# -------------------------------------------------------------------- MAE ----

def test_loss_excludes_absent_modalities():
    """Scoring an absent stream trains the model to predict zeros."""
    mae = tiny_mae().eval()
    mask = torch.tensor([[1., 0., 1.]])
    tm = random_token_mask(1, 3, mae.encoder.num_tokens, 0.75,
                           generator=torch.Generator().manual_seed(3))
    a = torch.randn(1, 3, *VOL)
    b = a.clone()
    b[:, 1] = 50.0                                   # wildly different MRI content
    with torch.no_grad():
        loss_a = mae(a, mask, token_mask=tm)["loss"]
        loss_b = mae(b, mask, token_mask=tm)["loss"]
    assert torch.allclose(loss_a, loss_b, atol=1e-4)


def test_every_parameter_receives_gradient():
    mae = tiny_mae()
    out = mae(torch.randn(2, 3, *VOL), torch.tensor([[1., 0., 1.], [1., 1., 1.]]),
              generator=torch.Generator().manual_seed(0))
    out["loss"].backward()
    missing = [n for n, p in mae.named_parameters() if p.grad is None]
    assert missing == [], missing


def test_loss_decreases_on_a_fixed_batch():
    """Integration smoke test: the whole pipeline can actually learn."""
    torch.manual_seed(0)
    mae = tiny_mae()
    image = torch.randn(2, 3, *VOL)
    mask = torch.tensor([[1., 0., 1.], [1., 1., 1.]])
    tm = random_token_mask(2, 3, mae.encoder.num_tokens, 0.75,
                          generator=torch.Generator().manual_seed(1))
    opt = torch.optim.AdamW(mae.parameters(), lr=1e-3)

    first = None
    for step in range(30):
        opt.zero_grad()
        loss = mae(image, mask, token_mask=tm)["loss"]
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
    assert float(loss.detach()) < first