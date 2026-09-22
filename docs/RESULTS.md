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

### 2.1 Temporal ablations (ADR-0014) -- 20 seeds, paired by initialisation

| arm | C-index | sd |
|---|---|---|
| correct chronology | 0.6515 | 0.0207 |
| shuffled chronology | 0.6475 | 0.0202 |
| ordinal positions (1, 2, 3 instead of elapsed days) | 0.6204 | 0.0189 |

| comparison | difference | 95% CI | permutation p |
|---|---|---|---|
| correct vs shuffled chronology | +0.0040 | [-0.0032, +0.0112] | 0.297 |
| **correct vs ordinal positions** | **+0.0312** | **[+0.0199, +0.0424]** | **0.00001** |
| shuffled vs ordinal positions | +0.0272 | [+0.0142, +0.0401] | 0.0003 |

**Supported claim:** continuous encoding of elapsed time outperforms ordinal
visit indices by 0.031 C-index, independently of sequence order.

**Refuted claim:** that the model learns disease evolution as an ordered
process. Shuffling each patient's studies costs nothing measurable (positive in
only 12 of 20 seeds). The architecture is permutation-invariant by construction
-- elapsed days are concatenated per study and pooling is a masked mean -- so
order enters nowhere.

Decomposing the +0.132 gain over a single scan: elapsed-time encoding accounts
for +0.031 (24%), sequence order for +0.004 (3%).

The benchmark can detect order-blindness: an order-invariant proxy for growth
(max/min burden) correlates -0.133 with the true growth ratio, so ordering
cannot be recovered from the set of embeddings alone.

### 2.2 Timeline versus single-scan (ADR-0014) -- patient-level bootstrap, 4000 replicates

| comparison | difference | 95% CI | bootstrap p |
|---|---|---|---|
| full timeline vs single-scan control | **+0.1865** | **[+0.0300, +0.3446]** | **0.022** |

Point estimates: full timeline 0.7061, single-scan control 0.5196, oracle (true
simulated risk) 0.8168.

The control uses an identical architecture and parameter count, sees only the
baseline study with elapsed time zeroed, never improved on its initialisation,
and early-stopped after 150 steps.

The interval is wide because the validation split has 60 patients and 36 events.
Resampling is at the PATIENT level: the C-index is computed over 1,201
comparable pairs, but pairs sharing a patient are not independent, and
resampling pairs would give an interval roughly 4.5x too narrow.

### 2.3 Cox implementation correctness (ADR-0014)

Fitted coefficients [0.741, -0.497, 0.289] against a simulated truth of
[0.8, -0.5, 0.3], with all five distractor coefficients under 0.05. Fitted
C-index 0.7333 against the true hazard's 0.7330 -- the model recovers
essentially all available signal. This validation is impossible on real data,
where the true hazard is unobservable.

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
| Aim 2 prior-ablation, change MAE | 3.34 -> 3.86 mm | 141 | unknown |

The Aim 1 pretrained-versus-scratch gaps (0.049 and 0.011) sit well inside their
estimated intervals. The null conclusion is very likely correct, but **without
the interval it is an assertion rather than a measurement.** Bootstrapping these
is the highest-priority outstanding work, because they are the basis for the
project's central negative findings (ADR-0007, ADR-0009).

## 4. Negative controls and methodological findings

The distinctive contribution. Each is a way a plausible-looking result can be
wrong, with the control that caught it.

**A null result survived every control and was still an artefact** (ADR-0009).
MAE pretraining tied random initialisation across seven measurements and four
evaluation-task redesigns. Two probe targets were discarded for being
underspecified or trivially recoverable; a shuffle control confirmed retrieval
was genuinely multimodal; an architecturally matched random-weight arm was used
throughout. Every control was correct.

The finding was nonetheless wrong. At 2.3M parameters, validation variance
explained was 0.005; at 184M, on identical data, objective, schedule and split,
it is **0.135** -- a 27-fold difference, converged over 82 validation checks.
The comparison had been between two representations that had both learned
almost nothing.

> Negative controls establish that an evaluation is capable of detecting a
> difference. They say nothing about whether the system was configured to
> produce one.

Effective rank behaves differently at scale: flat at 9-10 throughout training at
2.3M, but dropping 10 -> 4 in the first 150 steps at 184M and recovering to 6 --
compression that coincides with learning rather than with failure.

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
C-index 0.4667 for two architecturally different arms across every validation
check, over a cohort with 585 comparable pairs where the script's own cohort had
1,201. A direct training loop on the same data gave 0.706 and 0.520 (ADR-0014).

**A confidence interval must resample the independent unit.** Resampling the
1,201 comparable pairs rather than the 60 patients gives an interval 4.5x too
narrow, because pairs sharing a patient are not independent.

**Generated artifacts must not be versioned.** Committing synthetic cohorts grew
the repository to 2.38 GiB and made `git push` fail; history rewriting recovered
it to 80 KiB (ADR-0006).

## 5. Real-data findings

**PET is stored in Bq/mL, not SUV** (ADR-0008). Feeding raw values to an
SUV-calibrated normaliser saturates 25% of every volume. Conversion requires
patient weight, injected dose, and decay correction; omitting decay inflates
every SUV by ~1.8x at a typical 95-minute uptake delay, and variably so, making
scan timing a confound. Verified against normal liver at SUV 2.6.

**Slice spacing must be derived, not read.** CT `SliceThickness` reported
3.75 mm where the true derived spacing was 3.27 mm.

**A DICOM series may contain several acquisitions.** HCC-TACE-Seg's multiphase
CTs hold two contrast phases at identical slice positions; read as one volume
they interleave anatomically incoherent slices. Detected by duplicate positions
and split on `AcquisitionNumber`.

**Series must be resolved by UID, not by description.** In HCC-TACE-Seg the
segmentation's referenced CT is `Recon 2` for two patients, `Recon 3` for
another, and an unnumbered series for a fourth. A description-based rule would
have selected the wrong series for two of five patients.

**Longitudinal annotated data is scarce.** A survey of all 156 TCIA collections
found 47 with SEG or RTSTRUCT. Of the two inspected in detail: ISPY1's
segmentation is a 70%-threshold enhancement map with empty structured reports,
and HCC-TACE-Seg has expert multi-structure contours but only at baseline
(25 of 25 sampled patients have exactly one SEG). No collection with imaging
AND time-to-event outcomes was found (ADR-0012, ADR-0013).

## 6. Honest scope

- **Aim 1's representation claim is supported at 184M parameters** and was not
  at 2.3M (ADR-0009). Validation variance explained 0.135 vs 0.005. Caveats:
  one seed, and Lightning used both T4s despite `devices: 1`, so the effective
  batch was 16 against the baseline's 8.
- **All downstream comparisons predate this checkpoint** and are being rerun.
  The Aim 1 probes, Aim 2's tuned-vs-frozen arms, and Aim 3's embedding quality
  (burden recovery R^2 0.373, which bounded the achievable C-index at 0.659)
  were computed with the 2.3M encoder.
- **Aim 2's generation is demonstrated on synthetic imaging only** (ADR-0010).
  Real-data work validates lesion MEASUREMENT against expert contours
  (HCC-TACE-Seg), not change tracking or RECIST derivation.
- **Aim 3 has no real-data component.** Outcomes are simulated because no
  cohort with imaging and time-to-event data was found in the archives surveyed.
- **No medical LLM is integrated.** The report system predicts structured fields
  and renders deterministically -- defensible, since a language model can emit a
  fluent wrong measurement and this structurally cannot -- but it is not the
  brief's "align visual tokens with a medical LLM".
- **The MRI stream is untested on real data**, because no surveyed collection
  has same-date tri-modal studies.
- **Aim 2's synthetic benchmark carries a confound**: lesion size and trajectory
  are entangled by construction (corr 0.634), so change is partly inferable from
  the current scan alone (ADR-0011).

## 7. Outstanding work, in priority order

1. Rerun all downstream evaluations with the 184M checkpoint. Aim 1 probes
   first -- `scripts/ablate_pretraining.py` already exists and takes a
   `--checkpoint` argument.
2. Recompute Aim 3's embedding cache and achievable ceiling. Better embeddings
   should raise the 0.659 bound toward the 0.8168 oracle.
3. Bootstrap confidence intervals on whatever the rerun produces. Section 3's
   estimates are superseded, not fixed.
4. A second seed at 184M, to make the capacity result a measurement.
5. Real time-to-event cohort -- still the binding constraint on any clinical
   claim.
6. Medical LLM integration for Aim 2.