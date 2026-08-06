# ADR-0008: QIN-BREAST collection properties and the CT+PET subset decision

- **Status:** Accepted
- **Date:** 2026-08-05

## Context
QIN-BREAST (TCIA, DOI 10.7937/K9/TCIA.2016.21JUEBH0, CC-BY 3.0) was selected as
the first real cohort: longitudinal, multi-modal, and small enough to work with
on CPU. Inspecting its metadata via the public NBIA REST API before downloading
revealed several properties that are not evident from the collection
description and that materially affect how it can be used.

## Observations

**Cohort shape.** 68 patients, but PET/CT timepoints are unevenly distributed:

| PET/CT timepoints | patients |
|---|---|
| 0 (MR only) | 25 |
| 1 | 5 |
| 2 | 9 |
| 3 | 29 |

38 patients have >= 2 PET/CT timepoints (~4.9 GB). We downloaded 10 for
development.

**PET is stored in BQML, not SUV.** `Units = "BQML"`, values reaching ~21,000.
Feeding this to `IntensityNorm.PET_SUV_CLIP` (which clips to [0, 25]) would
saturate 25% of every volume. Conversion is implemented in `panorama.data.pet`;
all required tags (PatientWeight, RadionuclideTotalDose, half-life, injection
and scan times) are present for all 27 series inspected.

**The CT is CTAC, not diagnostic.** `SeriesDescription = "CTAC"` -- a low-dose
CT acquired for PET attenuation correction. It is adequate for anatomical
localisation but is NOT a contrast-enhanced diagnostic CT, so texture-based
findings drawn from it should be qualified.

**MR is quantitative, not anatomical.** Three sequences per study:
`DWI_EPI_MPS_smartTX` (108 images), `multi-flip_T1-map_smartTX` (200), and
`dynamic_smartTX` (500). The last is 4D DCE -- the same slices repeated over
time as contrast washes in -- and cannot be treated as a single 3D volume.

**CT/PET and MR are acquired on different dates**, typically 3-5 days apart.
Under our `Study` schema (one acquisition date per study) these are separate
studies, so genuinely tri-modal same-date studies are rare in this collection.

**Series descriptions are perfectly consistent** (`CTAC` x27, `PET AC 3DWB`
x27). This is single-site, single-protocol data, so string-matching selection
rules suffice here and will NOT generalise to multi-site collections.

**CT padding values reach -3024 HU**, outside the physical Hounsfield range.
This is the scanner's fill value outside the reconstruction circle. The fixed
HU window clips it harmlessly, but a meaningful fraction of each CT is
fabricated padding rather than anatomy.

## Decision
Use **CT + PET only** for the first real-data experiments:

- they are same-date acquisitions, so true multi-modal studies exist;
- they exercise the structural<->metabolic cross-attention that is Aim 1's
  actual claim;
- they avoid the 4D DCE problem entirely.

MR is deferred as a later extension, which will require choosing a
representative timepoint from the dynamic series.

## Consequences
- The usable cohort is 38 patients rather than 68.
- PET normalisation depends on the SUV conversion being correct; it is guarded
  by `tests/test_pet_suv.py`, anchored on normal liver falling in SUV 2-3.
- Aim 1's tri-modal claim is tested on synthetic data and on CT+PET here; a
  genuinely tri-modal real cohort remains future work.
- Selection rules written against this collection must be revisited before
  applying them to any multi-site data.