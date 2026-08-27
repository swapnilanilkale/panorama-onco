"""Read DICOM SEG objects into per-structure masks aligned to the source volume.

Two things make this harder than reading pixel_array:

1. Multiple segments are packed as CONSECUTIVE FRAMES in one array. Four
   segments over 109 slices gives shape (436, 512, 512). Which frame belongs to
   which segment, and to which slice, comes from
   PerFrameFunctionalGroupsSequence -- the standard does not guarantee that
   frames are grouped segment-by-segment.

2. A SEG covers only the slices containing its structures, so its frame count is
   usually smaller than the source series. Masks must be placed into a
   volume-shaped array by matching ImagePositionPatient, not by index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama.core.exceptions import DataIngestionError
from panorama.core.logging import get_logger
from panorama.data.volume import MedicalVolume

log = get_logger(__name__)


def read_segmentation(path: Path | str,
                      reference: MedicalVolume,
                      tolerance_mm: float = 1.0) -> dict[str, np.ndarray]:
    """Read a DICOM SEG into `{segment_label: bool mask}` shaped like `reference`.

    `reference` is the source imaging volume the SEG was drawn on; masks are
    placed by matching each frame's physical position to a reference slice.
    """
    import pydicom

    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:
        raise DataIngestionError(f"cannot read SEG {path!r}: {exc}") from exc

    segments = getattr(ds, "SegmentSequence", None)
    if not segments:
        raise DataIngestionError(f"{path!r} has no SegmentSequence")
    labels = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", f"segment{i}"))
              for i, s in enumerate(segments, start=1)}
    log.info("SEG %s: %d segments %s", Path(path).name, len(labels),
             list(labels.values()))

    frames = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not frames:
        raise DataIngestionError(
            f"{path!r} has no PerFrameFunctionalGroupsSequence; cannot map "
            f"frames to segments or slice positions")

    pixels = ds.pixel_array
    if pixels.ndim == 2:                       # a single-frame SEG
        pixels = pixels[None]
    if len(frames) != pixels.shape[0]:
        raise DataIngestionError(
            f"{path!r}: {len(frames)} frame descriptors but "
            f"{pixels.shape[0]} frames")

    # Reference slice positions along the slice normal, for matching.
    slice_dir = reference.affine[:3, 2]
    slice_dir = slice_dir / np.linalg.norm(slice_dir)
    origin = reference.affine[:3, 3]
    n_slices = reference.shape[2]
    reference_depths = np.array([
        float(np.dot(origin + slice_dir * reference.spacing_mm[2] * k, slice_dir))
        for k in range(n_slices)])

    masks = {label: np.zeros(reference.shape, dtype=bool)
             for label in labels.values()}
    unmatched = 0

    for i, frame in enumerate(frames):
        try:
            number = int(frame.SegmentIdentificationSequence[0]
                         .ReferencedSegmentNumber)
            position = np.asarray(
                frame.PlanePositionSequence[0].ImagePositionPatient, dtype=float)
        except (AttributeError, IndexError) as exc:
            raise DataIngestionError(
                f"{path!r} frame {i}: missing segment or position info") from exc

        depth = float(np.dot(position, slice_dir))
        k = int(np.argmin(np.abs(reference_depths - depth)))
        if abs(reference_depths[k] - depth) > tolerance_mm:
            unmatched += 1
            continue

        label = labels.get(number)
        if label is not None:
            masks[label][:, :, k] |= pixels[i].astype(bool)

    if unmatched:
        log.warning("%s: %d of %d frames had no reference slice within %.1f mm",
                    Path(path).name, unmatched, len(frames), tolerance_mm)

    for label, mask in masks.items():
        log.info("  %-18s %8d voxels", label, int(mask.sum()))
    return masks


def longest_axial_diameter_mm(mask: np.ndarray,
                              volume: MedicalVolume) -> tuple[float, int]:
    """RECIST-style longest diameter: the longest in-plane extent on ANY slice.

    RECIST measures on a single axial slice, as a radiologist does. Taking the 3D
    bounding diagonal instead would systematically over-measure -- for a lesion
    spanning several thick slices the through-plane extent can exceed the in-plane
    one, which is not what the criterion asks for.

    Returns (diameter_mm, slice_index).
    """
    if mask.shape != volume.shape:
        raise ValueError(f"mask {mask.shape} != volume {volume.shape}")

    row_mm, col_mm = volume.spacing_mm[0], volume.spacing_mm[1]
    best_mm, best_slice = 0.0, -1

    for k in range(mask.shape[2]):
        rows, cols = np.nonzero(mask[:, :, k])
        if rows.size < 2:
            continue
        # Exact longest chord needs all pairwise distances; on a convex-ish
        # lesion the extremes lie on the bounding box corners, which is a close
        # and far cheaper approximation.
        points = np.column_stack([rows * row_mm, cols * col_mm])
        if points.shape[0] > 2000:              # subsample very large slices
            index = np.random.default_rng(0).choice(points.shape[0], 2000,
                                                    replace=False)
            points = points[index]
        spread = points.max(axis=0) - points.min(axis=0)
        diameter = float(np.hypot(*spread))
        if diameter > best_mm:
            best_mm, best_slice = diameter, k

    return best_mm, best_slice


def mask_volume_ml(mask: np.ndarray, volume: MedicalVolume) -> float:
    """Segmented volume in millilitres (1 mL = 1000 mm^3)."""
    return float(mask.sum()) * volume.voxel_volume_mm3 / 1000.0

def is_measurable_lesion(mass: np.ndarray, organ: np.ndarray | None = None,
                         max_organ_fraction: float = 1.0) -> tuple[bool, str]:
    """Reject annotations that are not measurable target lesions.

    RECIST 1.1 excludes non-measurable disease. A `Mass` segmentation larger
    than the organ containing it describes diffuse or multifocal involvement,
    not a discrete lesion whose longest diameter means anything -- observed in
    1 of 5 HCC-TACE-Seg patients (ADR-0012).
    """
    if not mass.any():
        return False, "mask is empty"
    if organ is not None and organ.any():
        fraction = float(mass.sum()) / float(organ.sum())
        if fraction > max_organ_fraction:
            return False, (f"mass is {fraction:.2f}x the organ volume -- "
                           f"diffuse involvement, not a measurable lesion")
    return True, "measurable"