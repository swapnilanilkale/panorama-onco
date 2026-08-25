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

## Finding (revised 2026-08-24)

An initial reading attributed the null transfer to representation collapse
(effective rank 7 vs 10). Logging rank DURING training falsified that: rank held
at 9-10 from step 99 to 1999 while loss improved monotonically. The two rank
figures were also measured at different sample ceilings (84 vs 21) and are not
comparable.

The actual finding is simpler and more serious. Validation variance explained is
**0.005** against a training value of 0.068:

| step | val loss | val variance explained |
|---|---|---|
| 99 | 1.2066 | -0.2066 |
| 999 | 1.0008 | -0.0008 |
| 1999 | 0.9950 | 0.0050 |

The model takes ~1100 steps merely to beat the trivial predictor on held-out
data, and ends barely above it. MAE pretraining at this scale does not learn
generalisable structure. A frozen encoder that has learned nothing performs like
a random one because it functionally is one.

## Consequences (revised)
- Aim 1's representation claim is not supported, and the reason is
  under-learning, not collapse.
- Report VALIDATION variance explained, never training. The earlier figure of
  0.230 was training-only and measured memorisation over 41 epochs.
- Candidate causes, in order of likelihood: model capacity (2.3M parameters,
  4 layers, 128 dims is very small for volumetric data); cohort size (73 train
  studies); crop size (32^3 at 2mm sees 64mm of a 700mm field of view, so most
  crops contain little structure to reconstruct).
- The methodological lesson is general: a metric that is only logged at the end
  of training cannot distinguish "never learned" from "learned then lost".

## Caveat
R^2 0.46 from CT alone may largely reflect anatomical LOCATION (brain and
bladder are always hot, lung always cold) rather than lesion-level
structure-to-metabolism inference. Testing that requires region-matched crops.