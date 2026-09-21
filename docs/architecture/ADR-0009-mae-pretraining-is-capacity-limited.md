# ADR-0009: MAE pretraining is capacity-limited, not ineffective

- **Status:** Accepted (supersedes two earlier readings of the same question)
- **Date:** 2026-09-11
- **Supersedes:** the "representation collapse" and "under-learning" findings
  previously recorded under this number.

## History of this decision

This ADR has been wrong twice, and the record of how is more useful than the
conclusion.

**First reading (2026-08-24): representation collapse.** A frozen pretrained
encoder showed effective rank 7 of 128 against a random encoder's 10, and this
was read as MAE narrowing the feature space.

**Second reading (2026-08-24, same day): under-learning.** Logging effective
rank DURING training falsified the collapse story -- rank held flat at 9-10 from
step 99 to 1999 while loss improved monotonically, and the two figures compared
had different sample ceilings (84 vs 21). The revised finding was that
validation variance explained sat at 0.005 against a training value of 0.068:
the model was not learning anything that generalised. Capacity was named as the
most likely cause but not tested.

**Third reading (2026-09-11): capacity.** Tested. Confirmed.

## Evidence

Identical data, objective, schedule and split; only model size differs.

| | 2.3M params | 184M params |
|---|---|---|
| embed_dim / depth / heads | 128 / 4 / 8 | 768 / 12 / 12 |
| decoder dim / depth | 128 / 2 | 512 / 4 |
| train variance explained | ~0.23 | 0.20 |
| **validation variance explained** | **0.005** | **0.135** |

The 184M run is converged, not a lucky checkpoint: validation variance explained
rises steadily from step 143, reaches ~0.13 by step 1800, and holds there for
the remaining 1,200 steps (82 validation checks in total).

### Effective rank behaves differently at scale

| step range | effective rank (ceiling 21) |
|---|---|
| 35 | 10 |
| 143-575 | 4-5 |
| 611-1835 | 5 |
| 1871-2987 | 6 |

Rank drops sharply in the first ~150 steps, bottoms at 4, then recovers slowly
to 6 and stabilises -- while validation variance explained climbs throughout.

So compression DOES occur at 184M parameters, where it did not at 2.3M. But it
coincides with the model learning, not failing: a rank of 6 is the
low-dimensional structure the encoder found, not a degenerate space. The first
reading of this ADR was directionally right about the phenomenon and wrong about
its meaning.

## What was wrong, and why it took three attempts

The null result was robust across seven measurements and four evaluation-task
redesigns. Two probe targets were discarded for being underspecified or
trivially recoverable; a shuffle control confirmed retrieval was genuinely
multimodal; an architecturally matched random-weight arm was used throughout.
Every control was correct.

**None of that detected a wrong hyperparameter.** Negative controls verify that
an evaluation can discriminate; they cannot tell you that the thing being
evaluated was never given the capacity to differ from its control. Comparing a
2.3M-parameter pretrained encoder against a 2.3M-parameter random one was a fair
comparison between two representations that had both learned almost nothing.

## Consequences

**Every downstream null computed with the 2.3M encoder is now suspect** and
should be rerun with this checkpoint:

- Aim 1 retrieval R@1 (0.0780 pretrained vs 0.0851 scratch)
- Aim 1 crop-local and peak-PET probes (0.3917/0.3429 and 0.4628/0.4517)
- Aim 2 RECIST balanced accuracy (0.6154 tuned vs 0.5926 frozen)
- Aim 3 embedding quality: burden recovery R^2 0.373 bounded the achievable
  C-index at 0.659 (ADR-0013). Better embeddings should raise that ceiling
  toward the 0.8168 oracle.

**ADR-0007's conclusion** -- that the synthetic benchmark could not discriminate
between representations -- may also have been an artefact. It was inferred from
trained and untrained encoders performing identically, which is now explicable
by the trained encoder having learned little.

## Caveats on this run

- Lightning used both T4s (`CUDA_VISIBLE_DEVICES: [0,1]`) despite `devices: 1`,
  so the effective batch was 16 rather than 8 and the epoch was 36 batches. The
  2.3M baseline was run on CPU at batch 8. The comparison is therefore not
  perfectly controlled on batch size, though a 27-fold difference in validation
  variance explained is far larger than batch-size effects of that scale.
- Effective rank is measured against a ceiling of 21, because the validation
  split has 21 studies at one crop each. A higher `patches_per_study` at
  validation would make the rank figure more informative.
- Only one seed. The result is a large effect on a converged curve, but a
  repeated run would make it a measurement rather than an observation.

## Methodological finding (for any write-up)

> A null result can survive every negative control you can devise and still be
> an artefact of a single hyperparameter. Controls establish that an evaluation
> is capable of detecting a difference; they say nothing about whether the
> system was configured to produce one.

This is a stronger claim than the original negative result and belongs in the
paper's framing rather than as a footnote to it.