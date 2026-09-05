# ADR-0014: Aim 3 results -- temporal modelling recovers survival signal

- **Status:** Accepted (findings)
- **Date:** 2026-09-05

## Method
Study embeddings cached from a frozen vision encoder (mean over 4 crops), fed to
a transformer over the patient's timeline with sinusoidal encoding of elapsed
days, trained with the Cox partial likelihood. Full-batch, because the partial
likelihood compares each event against everyone at risk IN THE SAME BATCH -- at
batch 8 the mean risk set is ~4 against a true ~100.

Outcomes are simulated with a hazard depending on image-visible quantities
(ADR-0013): baseline burden, lesion count, and growth ratio, with growth
weighted highest because it is the only driver a single-timepoint model cannot
see.

Control arm: identical architecture and parameter count, fed only the FIRST
study with elapsed time zeroed.

## Results (60 val patients, 36 events, 1,201 comparable pairs)
| arm | val C-index |
|---|---|
| **full timeline** | **0.7061** |
| baseline only (control) | 0.5196 |
| oracle (true simulated risk) | 0.8168 |
| ADR-0013 predicted ceiling | 0.659 |

The control never improved on its initialisation and early-stopped after 150
steps. All of the timeline arm's performance is attributable to temporal
information.

The result EXCEEDED the ceiling predicted in ADR-0013. That prediction assumed
growth would be recovered as a ratio of two noisy burden estimates, whose errors
compound; the encoder evidently extracts temporal signal more directly than that
decomposition allows.

## What this is and is not
It IS: evidence that the timeline architecture recovers temporal signal when
that signal exists and is encoded in the study embeddings.

It is NOT a clinical survival result. The outcomes are simulated. No cohort with
imaging and time-to-event data was found in the TCIA survey (ADR-0013), so this
validates the machinery where the true hazard is known -- which no real cohort
permits -- and leaves real-outcome validation open.

## Implementation note
Lightning was removed from this training path. Full-batch training on 1.4 MB of
cached tensors gains nothing from a framework designed for distributed
mini-batch training, and a state-management bug in the LightningModule caused
`validation_step` to report a constant 0.4667 for both arms over a val cohort
with a different comparable-pair count (585) than the script's (1,201). A plain
loop is 60 lines and has no such failure mode.

## Consequences
- Overfitting is severe: train C-index reaches 0.99 against 0.71 validation.
  140 patients and 75 events cannot support 108K parameters; ADR-0013's
  10-events-per-covariate rule allows about 8. Early stopping on validation
  concordance is therefore load-bearing, not a refinement.
- The gap to the 0.8168 oracle is bounded by embedding quality (burden recovery
  R^2 0.373), which ADR-0009 attributes to under-learning at this model scale.