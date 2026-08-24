# ADR-0009: MAE pretraining collapses the representation on this cohort

- **Status:** Accepted (finding)
- **Date:** 2026-08-24

## Context
ADR-0007 recorded no measurable transfer from MAE pretraining on synthetic data
and concluded the benchmark could not discriminate. This tests the same claim on
real QIN-BREAST PET/CT (105 studies, 38 patients) with a probe target that is
demonstrably non-trivial.

## Method
Frozen encoder, linear ridge probe, target = peak normalised PET (SUV) in the
crop. Predicted twice: from CT alone (PET channel zeroed AND its presence bit
cleared, so the missing-modality token is substituted) and from CT+PET.
Control encoder is architecturally identical with random weights.

An earlier probe target -- mean CT intensity -- gave R^2 0.998 for BOTH
encoders. It was discarded: mean pooling of a linear patch projection makes the
input mean linearly recoverable by construction, so any encoder solves it.

## Results
| measurement (n=84 val) | pretrained | scratch |
|---|---|---|
| peak PET R^2, from CT alone | 0.4628 | 0.4517 |
| peak PET R^2, from CT+PET | 0.8551 | 0.8826 |
| effective rank (95% var) | **7 / 128** | **10 / 128** |

The 0.40 gap between CT-alone and CT+PET confirms the probe is discriminating,
not saturated.

## Finding
Pretraining shows no transfer, and the pretrained representation has FEWER
effective dimensions than random initialisation. MAE reconstruction at this
scale (73 train studies, 41 epochs) appears to reward encoding a few dominant
modes -- body position, tissue/air, gross intensity -- and discarding the
detail downstream tasks need.

This explains the ADR-0007 nulls mechanistically: the benchmarks were adequate;
the representation was impoverished.

## Caveat
R^2 0.46 from CT alone may largely reflect anatomical LOCATION (brain and
bladder are always hot, lung always cold) rather than lesion-level
structure-to-metabolism inference. Testing that requires region-matched crops.

## Consequences
- Aim 1's representation claim is NOT supported at this scale. Do not report
  pretraining benefit.
- Candidate causes to test next: model capacity (2.3M params is very small),
  cohort size (73 studies), and the masking ratio. Representation collapse
  under reconstruction objectives is a known failure mode addressed in the
  literature by contrastive or distillation auxiliaries.
- Effective rank should be logged during pretraining, not only after. A metric
  that reveals collapse mid-run is worth more than one discovered at evaluation.