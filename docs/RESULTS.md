# PANORAMA: consolidated results

Status as of 2026-09-05. Every number below is traceable to an ADR and a script
in `scripts/`. Claims are separated by evidential strength: results with
confidence intervals, results without, and open questions.

## 1. What was built

| aim | component | state |
|---|---|---|
| 1 | Multi-stream 3D ViT, CT/MRI/PET, cross-attention fusion, MAE pretraining | built, tested, runs on real clinical PET/CT |
| 1 | World-coordinate patch sampling (no voxel registration required) | built, tested |
| 2 | RECIST 1.1 including nadir-referenced progression | built, 13 property tests |
| 2 | Structured report head; RECIST derived, not predicted | built, evaluated |
| 2 | Contrastive image-text alignment | built, evaluated |
| 3 | Cox partial likelihood, concordance, Breslow baseline hazard | built, validated against a known hazard |
| 3 | Timeline encoder with continuous elapsed-time encoding | built, evaluated with ablations |

Real-data pipeline: TCIA REST download with resume, DICOM series reading with
position-based sorting and spacing validation, PET Bq/mL to SUV with decay
correction, DICOM SEG reading with position-matched frames, NIfTI conversion
with round-trip geometry verification. 94 tests, 14 ADRs.

## 2. Results WITH uncertainty quantification

### Temporal ablations (ADR-0014) -- 20 seeds, paired by initialisation

| comparison | difference | 95% CI | permutation p |
|---|---|---|---|
| correct vs shuffled chronology | +0.0040 | [-0.0032, +0.0112] | 0.297 |
| **correct vs ordinal positions** | **+0.0312** | **[+0.0199, +0.0424]** | **0.00001** |
| shuffled vs ordinal positions | +0.0272 | [+0.0142, +0.0401] | 0.0003 |

**Supported claim:** continuous encoding of elapsed time outperforms ordinal
visit indices by 0.031 C-index, independently of sequence order.

**Refuted claim:** that the model learns disease evolution as an ordered
process. Shuffling each patient's studies costs nothing measurable. The
architecture is permutation-invariant by construction -- elapsed days are
concatenated per study and pooling is a masked mean -- so order enters nowhere.

### Cox implementation correctness (ADR-0014)

Fitted coefficients [0.741, -0.497, 0.289] against a simulated truth of
[0.8, -0.5, 0.3], with all five distractor coefficients under 0.05. Fitted
C-index 0.7333 against the true hazard's 0.7330 -- the model recovers
essentially all available signal. Validation impossible on real data, where the
true hazard is unobservable.

## 3. Results WITHOUT uncertainty quantification

These are single-run point estimates. They are reported as such and should not
be treated as measurements until bootstrapped.

| result | value | n | bootstrap sd (est.) |
|---|---|---|---|
| MAE validation variance explained | 0.005 | -- | unknown |
| Retrieval R@1, pretrained vs scratch | 0.0780 / 0.0851 | 141 | binomial p only |
| Crop-local R^2, pretrained vs scratch | 0.3917 / 0.3429 | 250 | ~0.12 |
| Peak-PET R^2, pretrained vs scratch | 0.4628 / 0.4517 | 84 | ~0.20 |
| RECIST balanced accuracy, tuned vs frozen | 0.6154 / 0.5926 | 141 | unknown |
| Timeline vs single-scan C-index | 0.7061 / 0.5196 | 60 | ~0.05 |

The Aim 1 pretrained-versus-scratch gaps (0.049 and 0.011) sit well inside their
estimated intervals. The null conclusion is very likely correct, but **without
the interval it is an assertion rather than a measurement.** Bootstrapping these
is the highest-priority outstanding work.

The timeline-versus-single-scan gap (+0.187) is roughly 4x its estimated sd and
is unlikely to be noise, but should be bootstrapped for the same reason.

## 4. Negative controls and methodological findings

The distinctive contribution. Each is a way a plausible-looking result can be
wrong, with the control that caught it.

**Self-supervised pretraining tied random initialisation** (ADR-0007, ADR-0009).
Seven measurements across four task types, with an architecturally identical
random-weight control. Diagnosed as under-learning, not representation
collapse: validation variance explained 0.005 against training 0.068, and
effective rank held at 9-10 throughout training rather than falling.

**A probe target must be determined by the input.** The RECIST-category probe
was underspecified -- progression is defined by change between timepoints and
the probe saw one -- so PD recall was under 15% for reasons that had nothing to
do with the encoder (ADR-0009).

**A probe target must not be trivially recoverable.** Mean CT intensity gave
R^2 0.998 for both pretrained and random encoders, because mean pooling of a
linear patch projection makes the input mean linearly recoverable by
construction (ADR-0009).

**An absolute-measurement target permits shortcut learning.** Predicting
absolute lesion diameters left the prior scan unused: zeroing it cost 0.06 mm.
Adding a per-lesion CHANGE target -- not computable without both scans -- raised
that to 0.52 mm and produced the first measurable benefit from learned over
random features anywhere in the project (ADR-0011).

**A benchmark must be able to discriminate.** The synthetic imaging cohort's
lesions are bright spheres on smooth backgrounds; a random nonlinear projection
preserves enough structure to solve the probe tasks. Four independent
evaluations gave the same answer for trained and untrained encoders (ADR-0007).

**A metric can report a confident constant.** A LightningModule reported val
C-index 0.4667 for two architecturally different arms across every check, over a
cohort with a different comparable-pair count than the script's. A direct
training loop on the same data gave 0.706 and 0.520 (ADR-0014).

**Generated artifacts must not be versioned.** Committing synthetic cohorts grew
the repository to 2.38 GiB and made `git push` fail; history rewriting recovered
it to 80 KiB (ADR-0006).

## 5. Real-data findings

**PET is stored in Bq/mL, not SUV** (ADR-0008). Feeding raw values to an
SUV-calibrated normaliser saturates 25% of every volume. Conversion requires
patient weight, injected dose, and decay correction; omitting decay inflates
every SUV by ~1.8x at a typical 95-minute uptake delay, and variably so, making
scan timing a confound. Verified against normal liver at SUV 2.6.

**Slice spacing must be derived, not read.** CT `SliceThickness` said 3.75 mm
where the true derived spacing was 3.27 mm.

**A DICOM series may contain several acquisitions.** HCC-TACE-Seg's multiphase
CTs hold two contrast phases at identical positions; read as one volume they
interleave anatomically incoherent slices.

**Longitudinal annotated data is scarce.** A survey of all 156 TCIA collections
found 47 with SEG or RTSTRUCT. Of the two inspected in detail: ISPY1's
segmentation is a 70%-threshold enhancement map with empty structured reports,
and HCC-TACE-Seg has expert multi-structure contours but only at baseline
(25 of 25 sampled patients have exactly one SEG). No collection with imaging
AND time-to-event outcomes was found.

## 6. Honest scope

- **Aim 1's representation claim is not supported** at the scale tested. A
  184M-parameter run is in progress; ADR-0009 predicts capacity as the most
  likely cause.
- **Aim 2's generation is demonstrated on synthetic imaging only.** Real-data
  work validates measurement (HCC-TACE-Seg expert contours), not change or
  RECIST derivation.
- **Aim 3 has no real-data component.** Outcomes are simulated because no
  suitable cohort exists in the archives surveyed.
- **No medical LLM is integrated.** The report system predicts structured fields
  and renders deterministically -- defensible, but not the brief's "align visual
  tokens with a medical LLM".
- **The MRI stream is untested on real data**, because no collection surveyed
  has same-date tri-modal studies.

## 7. Outstanding work, in priority order

1. Bootstrap confidence intervals on every result in section 3.
2. Architecture ablation: does the transformer beat a per-study MLP with masked
   mean pooling, given the task is permutation-invariant?
3. Capacity experiment at 184M parameters (in progress) -- determines whether
   Aim 1's null is a scale artefact.
4. Real time-to-event cohort. TCGA-linked TCIA collections carry `days_to_death`
   and `vital_status`; this is the binding constraint on any clinical claim.
5. Medical LLM integration for Aim 2.