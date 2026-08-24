# ADR-0010: Aim 2 generation is developed on synthetic imaging

- **Status:** Accepted
- **Date:** 2026-08-24

## Context
Aim 2 requires a model that ingests current and prior scans and emits a
RECIST-correct structured report. Training that needs per-lesion measurements.
QIN-BREAST supplies neither segmentations nor diameters (ADR-0008), and its CT
is CTAC -- low-dose attenuation-correction, not diagnostic quality -- so
deriving lesion contours from it is unreliable.

## Options considered
- **(a) Generate on synthetic, align on real.** Synthetic lesions have known
  geometry, so measurements are ground truth. Alignment needs no annotations
  and runs on real data unchanged.
- **(b) Add segmentation.** TotalSegmentator/nnU-Net on real CT. Substantial
  subsystem, GPU-dependent, and breast lesions on CTAC are hard; poor masks
  would silently corrupt every downstream measurement.
- **(c) Find an annotated collection.** TCIA collections shipping SEG/RTSTRUCT
  give real images with expert contours, at the cost of another
  download-and-inspect cycle and possibly losing longitudinal structure.

## Decision
(a) now, (c) as the documented follow-up once the generator works.

## Consequences
- The generation capability is demonstrated on synthetic imaging only. This is
  a materially weaker claim than clinical validation and must be stated as such
  in any write-up, not glossed.
- The alignment result on real QIN-BREAST data is the evidence that the pipeline
  transfers; it is reported separately and honestly.
- All generator code is annotation-source agnostic: it consumes
  `StructuredReport` objects, so swapping synthetic geometry for expert contours
  is a data change, not a code change.