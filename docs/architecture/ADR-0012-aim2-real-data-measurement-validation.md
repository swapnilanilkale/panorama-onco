# ADR-0012: Aim 2 real-data arm validates MEASUREMENT, not change

- **Status:** Accepted
- **Date:** 2026-08-25

## Context
ADR-0010 deferred a real-data arm for Aim 2 because QIN-BREAST has no lesion
annotations. A survey of all 156 TCIA collections (`scripts/find_annotated_
collections.py`) found 47 carrying SEG or RTSTRUCT alongside imaging. Two were
inspected in detail.

## What the candidates actually contain

**ISPY1** (222 patients, MR, 100% longitudinal, SEG at every timepoint):
one segment, `'PE Tumor'`, SEMIAUTOMATIC, produced by thresholding percent
enhancement at 70%. Connected-component analysis gives 21 pieces of which one
holds 98.5% of voxels -- a single thresholded region, not independently
contoured target lesions. Its Structured Report objects are empty templates:
every measurement field is None. The trial's own endpoint is functional tumour
volume, not RECIST.

**HCC-TACE-Seg** (105 patients, CT, 100% longitudinal): four named segments per
SEG -- `Liver`, `Mass`, `Portal vein`, `Abdominal aorta` -- a proper
multi-structure expert delineation. `Mass` is a genuine tumour contour. BUT the
SEG appears only at the FIRST timepoint: 25 of 25 sampled patients have exactly
one SEG series.

## Decision
Use **HCC-TACE-Seg** to validate the MEASUREMENT half of Aim 2 on real annotated
CT. Longitudinal change and RECIST derivation remain on synthetic imaging, where
per-lesion trajectories are ground truth.

The real-data claim is therefore: *lesion measurement from real clinical CT with
expert contours.* The synthetic claim is: *change tracking and RECIST derivation
across a treatment course.* Two clearly-scoped results, neither overstated.

## Options rejected
- **ISPY1 with functional tumour volume.** Would give a longitudinal real-data
  result, but substitutes a different endpoint for the RECIST 1.1 the project
  brief specifies, and rests on a 70%-threshold map rather than expert contours.
- **ISPY1 with RECIST approximated from the dominant PE component.** Presenting a
  single-lesion threshold-derived diameter as RECIST would misrepresent a
  multi-lesion criterion built on radiologist measurement.

## Consequences
- Aim 2 has a real-data component that did not previously exist.
- CT retains the existing preprocessing, HU windowing and DICOM reader unchanged.
- DICOM SEG objects pack multiple segments as consecutive frames; per-segment
  masks must be unpacked via `PerFrameFunctionalGroupsSequence` rather than read
  directly from `pixel_array` (observed shape (436, 512, 512) for four segments).
- Change validation on real data remains open, and would require a collection
  with per-timepoint expert contours. None was found in this survey.