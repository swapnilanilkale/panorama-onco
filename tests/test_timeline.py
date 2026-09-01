"""Timeline encoder guards."""
import numpy as np
import pytest
import torch

from panorama.survival.cox import cox_partial_likelihood_loss
from panorama.survival.timeline import TimeEncoding, TimelineEncoder, collate_timelines

EMBED = 32


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return TimelineEncoder(embed_dim=EMBED, hidden_dim=32, depth=2, num_heads=4)


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    lengths = [2, 3, 4, 2, 4]
    embeddings = [torch.randn(n, EMBED) for n in lengths]
    days = [torch.tensor(np.cumsum([0.0] + list(rng.uniform(30, 200, n - 1))),
                         dtype=torch.float32) for n in lengths]
    return (*collate_timelines(embeddings, days), lengths)


def test_padding_does_not_change_the_risk_score(encoder, batch):
    """Padded positions must reach neither attention nor pooling.

    Timeline LENGTH correlates directly with survival -- a patient with four
    visits lived long enough to have four visits -- so a length leak is a
    confound with the label itself, not merely a nuisance.
    """
    x, days, mask, lengths = batch
    encoder.eval()
    with torch.no_grad():
        for i, n in enumerate(lengths):
            padded = encoder(x[i:i + 1], days[i:i + 1], mask[i:i + 1])
            trimmed = encoder(x[i:i + 1, :n], days[i:i + 1, :n], mask[i:i + 1, :n])
            assert torch.allclose(padded, trimmed, atol=1e-5), i


def test_elapsed_time_changes_the_output(encoder, batch):
    """Identical scans over 30 days and 400 days imply different growth rates."""
    x, days, mask, _ = batch
    encoder.eval()
    with torch.no_grad():
        original = encoder(x[:1], days[:1], mask[:1])
        stretched = encoder(x[:1], days[:1] * 4.0, mask[:1])
    assert not torch.allclose(original, stretched, atol=1e-4)


def test_risk_depends_on_the_embeddings(encoder, batch):
    """A model returning a constant would still minimise a shift-invariant loss."""
    x, days, mask, _ = batch
    encoder.eval()
    with torch.no_grad():
        risk = encoder(x, days, mask)
    assert risk.std() > 1e-4
    assert len(torch.unique(risk)) == x.shape[0]


def test_single_patient_forward_is_not_degenerate(encoder, batch):
    """Regression guard: centring risk INSIDE the model made a batch of one
    return exactly 0.0 for every input. Shift-invariance belongs in the loss."""
    x, days, mask, _ = batch
    encoder.eval()
    with torch.no_grad():
        a = encoder(x[0:1], days[0:1], mask[0:1])
        b = encoder(x[1:2], days[1:2], mask[1:2])
    assert float(a) != 0.0 or float(b) != 0.0
    assert not torch.allclose(a, b)


def test_collate_marks_real_positions(batch):
    x, days, mask, lengths = batch
    assert x.shape[1] == max(lengths)
    for i, n in enumerate(lengths):
        assert mask[i, :n].all()
        assert not mask[i, n:].any()
        assert (x[i, n:] == 0).all()

def test_time_encoding_is_continuous():
    """Small changes in elapsed days give bounded changes in the encoding.

    Continuity matters so an unseen follow-up interval interpolates rather than
    falling out of vocabulary -- schedules differ between sites. Note the
    highest-frequency component has a period of ~2pi days, so sub-day steps DO
    move it appreciably; the guarantee is boundedness, not imperceptibility.
    """
    encoding = TimeEncoding(dim=16)
    days = torch.arange(0.0, 400.0, 5.0)
    codes = encoding(days)

    # No jumps: consecutive 5-day steps stay within the Lipschitz bound.
    steps = (codes[1:] - codes[:-1]).norm(dim=-1)
    assert float(steps.max()) < 3.0
    assert torch.isfinite(codes).all()

    # Distinctness: no two timepoints collide.
    similarity = torch.nn.functional.cosine_similarity(
        codes[:, None, :], codes[None, :, :], dim=-1)
    off_diagonal = similarity - torch.eye(len(days)) * 2
    assert float(off_diagonal.max()) < 0.999


def test_low_frequency_components_vary_slowly():
    """The multi-scale design: low-frequency dims separate months and years
    while high-frequency dims resolve days."""
    encoding = TimeEncoding(dim=16, max_period=1000.0)
    near = encoding(torch.tensor([100.0, 105.0]))
    far = encoding(torch.tensor([100.0, 700.0]))
    # Last dims are the lowest frequencies -- they should barely move over 5
    # days and move clearly over 600.
    assert (near[0, -2:] - near[1, -2:]).abs().max() < 0.05
    assert (far[0, -2:] - far[1, -2:]).abs().max() > 0.1


def test_time_encoding_requires_even_dim():
    with pytest.raises(ValueError, match="even"):
        TimeEncoding(dim=7)


def test_trains_with_the_cox_loss(encoder, batch):
    """Integration: encoder plus partial likelihood must reduce the loss."""
    x, days, mask, _ = batch
    rng = np.random.default_rng(0)
    duration = torch.tensor(rng.exponential(300, x.shape[0]), dtype=torch.float32)
    event = torch.ones(x.shape[0], dtype=torch.bool)

    opt = torch.optim.AdamW(encoder.parameters(), lr=3e-3)
    first = None
    for step in range(60):
        opt.zero_grad()
        loss = cox_partial_likelihood_loss(encoder(x, days, mask), duration, event)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.detach())
    assert float(loss.detach()) < first