# ADR-0011: Aim 2 results -- structured report generation on synthetic imaging

- **Status:** Accepted (findings)
- **Date:** 2026-08-25

## Method
Study pairs (prior, current) -> a `StructuredReportHead` predicting lesion count,
per-lesion diameters, per-lesion CHANGE from prior, organ, and new-lesion flag.
RECIST is DERIVED from predicted diameters by `panorama.clinical.recist.classify`,
never predicted directly, so the model cannot state a category contradicting its
own measurements. Synthetic cohort per ADR-0010 (200 patients, 702 studies,
496/141/65 patient-level split).

## Results (validation, n=141 examples)
| | fine-tuned | frozen random |
|---|---|---|
| diameter MAE | **5.48 mm** | 5.73 mm |
| change MAE | **3.39 mm** | 3.87 mm |
| RECIST accuracy | 0.688 | 0.667 |
| RECIST balanced accuracy | 0.615 | 0.593 |
| PD recall | 0.381 | 0.333 |

Chance balanced accuracy is 0.333; always-predict-SD scores 0.55 raw accuracy.
The predict-zero baseline for change is 4.56 mm.

## The methodological finding
The first version predicted only ABSOLUTE diameters. An ablation showed zeroing
the prior study cost 0.06 mm -- the prior was unused. The reason is structural:
absolute diameter is a function of the current scan alone, so the loss had no
reason to use the other input.

Adding a per-lesion CHANGE target -- not computable without both scans -- raised
the cost of removing the prior to 0.52 mm (8x) and produced the FIRST measurable
benefit from learned over random features anywhere in this project (change MAE
3.39 vs 3.87). Every prior comparison, across seven measurements, had been null.

**If you want a model to use an input, the target must be impossible to predict
without it.**

## Limitation: a confound in the synthetic data
`corr(current diameter, true change) = +0.634`. Trajectory and size are
entangled by construction -- a progressor is large *because* it grew -- so change
is partly inferable from the current scan alone. Zeroing the current image still
costs 3x more than zeroing the prior (1.56 vs 0.52 mm).

The change result is therefore real but INFLATED: it does not isolate temporal
comparison. Fixing it means drawing size and trajectory independently and
regenerating the cohort, which would invalidate all results above. Deferred in
favour of route (c) in ADR-0010 -- real annotated data, where trajectories are
not formula-generated.

## Honest scope
- Generation is demonstrated on SYNTHETIC imaging only (ADR-0010).
- PD recall of 0.38 means the system misses ~60% of progression events, the
  category that changes clinical management. Not deployable.
- No medical LLM is integrated. What exists is structured field prediction plus
  deterministic rendering, which is a defensible design (see the docstring in
  `report_head.py`) but is not the brief's "align visual tokens with a medical
  LLM".
- No RexRank, RadGraph-F1, or expert review.