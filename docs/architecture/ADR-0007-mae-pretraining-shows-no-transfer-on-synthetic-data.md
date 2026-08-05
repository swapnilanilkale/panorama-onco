# ADR-0007: MAE pretraining shows no measurable transfer on the synthetic cohort

- **Status:** Accepted (finding, not a decision to change course)
- **Date:** 2026-08-05

## Context
Aim 1 claims self-supervised MAE pretraining produces clinically meaningful
representations. We tested this by comparing a pretrained frozen encoder
against an architecturally identical randomly-initialised frozen encoder.

## Evidence
| measurement (n) | pretrained | scratch | diff |
|---|---|---|---|
| retrieval R@1 (141) | 0.0780 | 0.0851 | -0.007 |
| RECIST balanced acc (141) | 0.4785 | 0.4805 | -0.002 |
| crop-local lesion diameter R^2, 1.5k steps (58) | 0.4323 | 0.4589 | -0.027 |
| crop-local lesion diameter R^2, 15k steps (250) | 0.3917 | 0.3429 | +0.049 |
| study-level tumour burden R^2 (141) | 0.3659 | 0.3640 | +0.002 |

Pretrained leads in 2 of 5. The largest gap (0.049) is 0.55 sd of the
sampling distribution of R^2 at n=250 -- not significant.

A 10x pretraining sweep raised variance-explained 0.248 -> 0.288 and retrieval
R@1 0.0709 -> 0.0922, but moved no probe.

## Negative controls
- Shuffling image embeddings drops retrieval to exactly chance (0.0071),
  confirming retrieval is genuinely multimodal.
- Crop-local R^2 of ~0.39 confirms the probe measures real signal.
- The two encoders' embeddings have cosine similarity 0.029 -- they are
  genuinely different feature spaces that perform identically.

## Interpretation
The synthetic lesions are bright spheres on a smooth background. A random
nonlinear projection preserves enough structure to solve these tasks, so the
benchmark cannot discriminate between representations. This is a statement
about the benchmark, not about MAE.

## Consequences
- Do not claim pretraining benefit on synthetic data.
- The representation claim must be tested on real TCIA data, where texture and
  anatomical variation cannot be captured by a random projection.
- Two methodological errors found and fixed along the way: RECIST classification
  is underspecified from a single timepoint, and study-level targets require
  multi-crop pooling. Both are recorded in the probe module's docstrings.