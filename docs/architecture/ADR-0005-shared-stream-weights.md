# ADR-0005: Share self-attention weights across modality streams

- **Status:** Accepted
- **Date:** 2026-07-30

## Context
Each stream (CT/MRI/PET) needs a transformer encoder. Weights can be separate
per stream or shared. At embed_dim=768, depth=12: shared = 85M params,
separate = 255M (+170M). More importantly, with separate weights each stream's
encoder only receives gradient from studies where that modality exists --
often ~1/3 of a cohort, and far less for MRI.

## Decision
**Share** self-attention blocks across streams by default
(`share_stream_weights=True`), disambiguated by the learned modality embedding
from M2.2. Patch-embedding projections remain per-modality (different physics).
Cross-modal fusion blocks are shared between the two structural streams.
The flag allows switching to per-stream weights as an experiment.

## Consequences
- 3x fewer parameters -- important given medical cohorts are small.
- Every study trains all encoder weights, whatever modalities it has.
- Matches the "generalist medical AI" framing: one encoder, many modalities.
- Risk: shared weights may underfit genuinely modality-specific features.
  Mitigation: the flag makes this a measurable experiment, not an assumption.