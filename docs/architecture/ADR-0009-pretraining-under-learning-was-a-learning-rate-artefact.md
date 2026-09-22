# ADR-0009: Pretraining under-learning was a learning-rate artefact

- **Status:** Accepted
- **Date:** 2026-09-21
- **Supersedes:** three earlier readings of the same question under this number
  (representation collapse; under-learning; capacity).

## The question
MAE-pretrained encoders tied randomly initialised ones across seven downstream
measurements (ADR-0007). Why?

## History: four readings

1. **Representation collapse** (2026-08-24). Frozen effective rank 7 vs 10 for
   a random encoder. Falsified the same day: logged during training, rank held
   flat at 9-10, and the two figures had different sample ceilings (84 vs 21).
2. **Under-learning** (2026-08-24). Validation variance explained 0.005 against
   training 0.068. Correct as a description; the cause was not identified.
3. **Capacity** (2026-09-11). A 184M-parameter model reached 0.13. Accepted
   prematurely: that run differed from the baseline in batch size (8 vs 2),
   effective learning rate (3.1e-5 vs 7.8e-6), steps (3,000 vs 2,000) and
   samples seen (24,000 vs 4,000), not only in model size.
4. **Learning rate** (2026-09-21). A 2.3M model at the 184M run's exact
   training configuration reached 0.11. This reading.

## Evidence

All three runs: QIN-BREAST PET/CT, 73 train / 21 val studies, identical split,
MAE objective, mask ratio 0.75, base_lr 1e-3 under the linear scaling rule
lr = base_lr x batch / 256.

| run | params | batch | effective lr | steps | samples seen | val variance explained |
|---|---|---|---|---|---|---|
| original baseline | 2.3M | 2 | 7.8e-6 | 2,000 | 4,000 | 0.005 |
| **controlled** | **2.3M** | **8** | **3.1e-5** | **3,000** | **24,000** | **0.110** |
| large | 184M | 8 | 3.1e-5 | 3,000 | 24,000 | 0.130 |

Plateau values are the mean of the last 20 validation checks (steps 2303-2987);
sd 0.0035 (184M) and 0.0038 (2.3M).

Decomposition of the gain from 0.005 to 0.130:

| factor | contribution | share |
|---|---|---|
| training configuration (batch / lr / steps) | +0.105 | 84% |
| model capacity (2.3M -> 184M) | +0.021 | 16% |

Evidence files: `docs/evidence/qin-large-184M-metrics.csv`, and the 2.3M run's
metrics exported alongside it.

## Capacity: small but consistent

At the matched configuration the 184M model is ahead at 20 of 20 plateau checks
(gap +0.014 to +0.024) and converges faster:

| threshold | 184M first reaches it at | 2.3M first reaches it at |
|---|---|---|
| 0.00 | step 71 | step 251 |
| 0.05 | step 179 | step 539 |
| 0.10 | step 719 | step 1223 |

Both runs share the seed, data order and validation crops, so their
fluctuations are correlated (r = +0.56 on step-to-step changes; both dip at
step 2411). The per-check comparison is therefore paired. It is still one seed
per arm, so the +0.021 has no confidence interval.

## Effective rank at scale
At 184M, rank fell from 10 to 4 within ~150 steps, then recovered to 6 while
validation variance explained kept rising: compression that accompanies
learning rather than failure. The rank ceiling is 21 (21 validation studies at
one crop each), which limits how much this can say.

## Root cause
The linear scaling rule is calibrated for large-batch training, where MAE
reference settings use batches in the thousands. At batch 2 it reduced a
base_lr of 1e-3 to 7.8e-6. Every pretraining configuration in the project used
batch 2, including `pretrain_smoke.yaml` and the original `pretrain_qin.yaml`.
No error or warning was raised; loss still decreased on the training set.

## Consequences
- **All seven downstream nulls (ADR-0007) used encoders trained at a near-zero
  learning rate.** They compared two representations that had both learned
  very little, and must be rerun with properly trained encoders.
- **Reruns do not require the 184M model.** The 2.3M model reaches 85% of its
  validation performance at the corrected configuration and trains on CPU.
- **The configuration is not yet optimised.** 3.1e-5 was not chosen; it is
  simply what batch 8 produced. A learning-rate sweep may move the 2.3M model
  further, and could close the capacity gap entirely.
- Configs must set the effective learning rate explicitly, or the validator
  must reject an effective lr below a sane floor.

## Methodological findings

> **Negative controls on the evaluation do not control the training.** Seven
> measurements, four task redesigns, a shuffle control and a matched
> random-weight arm were all correct, and none could detect an encoder trained
> at 7.8e-6.

> **A confirming experiment must change one variable.** The capacity
> explanation was accepted after a run that changed four. The controlled rerun
> attributed 84% of the effect to the variables that were not supposed to be
> under test.