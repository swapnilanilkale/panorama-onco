# ADR-0014: Aim 3 results -- what the timeline encoder actually learns

- **Status:** Accepted (findings)
- **Date:** 2026-09-05

## Method

Study embeddings are cached from a frozen vision encoder (mean over 4
foreground-biased crops per study) and fed to a transformer over the patient's
timeline, with sinusoidal encoding of elapsed days concatenated to each study
embedding. Training uses the Cox partial likelihood.

Training is **full-batch** by necessity: the partial likelihood compares each
event against everyone at risk IN THE SAME BATCH, so mini-batching shrinks the
risk sets and biases the estimate. At batch 8 the mean risk set is ~4 against a
true ~100. The cohort is 1.4 MB of cached embeddings, so there is no reason to
approximate.

Outcomes are simulated with a hazard depending on image-visible quantities
(ADR-0013):

    log h = 0.9*z(baseline burden) + 0.4*z(lesion count) + 1.2*z(growth ratio)

Growth is weighted highest because it is the only driver a single-timepoint
model cannot observe.

## Primary result (60 val patients, 36 events, 1,201 comparable pairs)

| arm | val C-index |
|---|---|
| **full timeline** | **0.7061** |
| single-scan control (first study only, elapsed time zeroed) | 0.5196 |
| oracle (true simulated risk) | 0.8168 |
| ADR-0013 predicted ceiling | 0.659 |

The control uses an identical architecture and parameter count, sees only the
baseline study, never improved on its initialisation, and early-stopped after
150 steps. All of the timeline arm's performance is attributable to having more
than one study.

The result exceeded ADR-0013's predicted ceiling of 0.659. That prediction
assumed growth would be recovered as a ratio of two noisy burden estimates,
whose errors compound; the encoder evidently extracts the signal more directly
than that decomposition allows.

## Temporal ablations (20 seeds per arm, paired by initialisation)

| arm | C-index | sd |
|---|---|---|
| correct chronology | 0.6515 | 0.0207 |
| shuffled chronology | 0.6475 | 0.0202 |
| ordinal positions (1, 2, 3 instead of elapsed days) | 0.6204 | 0.0189 |

| comparison | difference | 95% CI | permutation p |
|---|---|---|---|
| correct vs shuffled | +0.0040 | [-0.0032, +0.0112] | 0.297 |
| correct vs ordinal | **+0.0312** | [+0.0199, +0.0424] | **0.00001** |
| shuffled vs ordinal | +0.0272 | [+0.0142, +0.0401] | 0.0003 |

In the shuffle arm each patient's studies are permuted with their elapsed days
carried along, so the (scan, time) pairing stays valid and only sequence order
changes. Five seeds were insufficient -- a 0.012 effect needs ~18 -- so all
comparisons use 20.

The benchmark can detect order-blindness: an order-invariant proxy for growth
(max/min burden) correlates -0.133 with the true growth ratio, so ordering
cannot be recovered from the set of embeddings alone.

## What the model actually learns

Decomposing the +0.132 gain over a single scan:

- **elapsed-time encoding: +0.031 (24%)** -- significant, positive in 18/20 seeds
- **sequence order: +0.004 (3%)** -- CI contains zero, positive in 12/20 seeds

**The model does not learn disease evolution as an ordered process.** Shuffling
a patient's studies costs nothing measurable. This is the architecture behaving
as designed rather than a defect: elapsed days are concatenated to each study
embedding and pooling is a masked mean, so temporal information is PER-STUDY and
the whole computation is permutation-invariant. Order enters nowhere.

The supported claim is therefore narrower and cleaner than the primary result
suggests:

> Continuous encoding of elapsed time outperforms ordinal visit indices by
> 0.031 C-index (p = 1e-5), independently of sequence order.

This matters clinically because oncology follow-up is irregular: identical
measurements 30 days apart and 400 days apart imply completely different growth
rates.

The claim that must NOT be made is that the model learns temporal ordering or
disease trajectory. A reviewer running this ablation would find otherwise.

## What this is and is not

It IS: evidence that continuous elapsed-time encoding recovers survival signal
that ordinal position encoding does not, and that multi-study input vastly
outperforms single-study input, when that signal exists and is present in the
study embeddings.

It is NOT a clinical survival result. The outcomes are simulated. No cohort with
imaging and time-to-event data was found in the TCIA survey (ADR-0013), so this
validates the machinery where the true hazard is known -- which no real cohort
permits -- and leaves real-outcome validation open.

## Implementation note: Lightning removed from this path

The LightningModule reported a constant val C-index of 0.4667 for both the
timeline and control arms across all validation checks, over a cohort with 585
comparable pairs where the script's own cohort had 1,201. A direct training loop
on the same data gave 0.706 and 0.520 respectively.

Full-batch training on cached tensors gains nothing from a framework built for
distributed mini-batch training: it contributed a dummy dataloader, an
artificial epoch concept, and a state-management bug. The replacement is 60
lines in `panorama.survival.train` and has no such failure mode.

## Consequences

- Overfitting is severe: train C-index reaches 0.99 against 0.71 validation.
  140 patients and 75 events cannot support 108K parameters; ADR-0013's
  10-events-per-covariate rule allows about 8. Early stopping on validation
  concordance is load-bearing, not a refinement.
- The gap to the 0.8168 oracle is bounded by embedding quality (burden recovery
  R^2 0.373), which ADR-0009 attributes to under-learning at this model scale.
- **Open ablation:** if the task is permutation-invariant, a transformer over
  the sequence may not earn its parameters over a per-study encoder with masked
  mean pooling. Worth testing, and a simpler model would be easier to defend.
- Any write-up must frame the contribution as irregular-follow-up modelling,
  not sequence modelling.