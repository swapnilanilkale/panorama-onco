# ADR-0013: Aim 3 scope, benchmark design, and the expected ceiling

- **Status:** Accepted
- **Date:** 2026-09-01

## Context
Aim 3 requires overall and progression-free survival prediction. Neither is
available in the collections used: QIN-BREAST's endpoint is pathologic complete
response (a binary label determined at surgery, with no follow-up time), and
HCC-TACE-Seg carries no outcome data. No cohort with imaging AND time-to-event
data was found in the TCIA survey.

The synthetic imaging cohort cannot substitute directly: its studies fall on a
fixed 90-day schedule, so derived PFS takes only three distinct durations,
identical between events and censored patients (both median 180 days, range
180-270). Survival models rank by risk, which requires duration to carry
information.

## Decision
Validate the survival machinery on SIMULATED outcomes whose hazard depends on
image-visible quantities:

    log h = 0.9*z(baseline burden) + 0.4*z(lesion count) + 1.2*z(growth ratio)

Growth is weighted highest deliberately -- it is the only driver a
single-timepoint model cannot see, so it separates a sequence encoder from a
per-study baseline. Verified: adding growth raises the oracle C-index from
0.705 (burden + count) to 0.837.

## Expected results, fixed BEFORE the experiment
Cached embeddings recover baseline tumour burden at R^2 0.373 (MAE 12.3 mm
against a target sd of 21.0 mm) -- a 28% relative error. Growth is a RATIO, so
that error compounds: simulated growth correlation falls to +0.12.

The achievable C-index given these embeddings is therefore:

| | C-index |
|---|---|
| oracle, true risk | 0.790 |
| **achievable ceiling from embeddings** | **0.659** |
| burden only, no temporal signal | 0.612 |

A result near 0.66 means the timeline encoder extracted essentially all
available signal. A result near 0.61 means it recovered nothing temporal.
Stating this in advance prevents fitting the interpretation to the outcome.

## Consequences
- Aim 3 has NO real-data component. The machinery is validated where the true
  hazard is known -- which no real cohort permits -- and applying it to real
  outcomes remains open work.
- The narrow 0.61-0.66 window is a consequence of embedding quality, which
  ADR-0009 already attributes to under-learning at this model scale. A stronger
  vision encoder would widen it.
- Two prior probe designs failed by not asking this question first: the RECIST
  probe measured a target the input did not determine, and the tissue-fraction
  probe measured one any encoder could solve. Establishing the achievable
  ceiling before running is now standard practice in this project.