"""Read a DICOM series into a MedicalVolume.

Real DICOM is a directory of per-slice files with no reliable ordering and
several tags that must be applied before the pixel values mean anything. Every
step here corresponds to a way silent corruption gets into medical pipelines.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from panorama.core.constants import Modality
from panorama.core.exceptions import DataIngestionError
from panorama.core.logging import get_logger
from panorama.data.volume import MedicalVolume

log = get_logger(__name__)

# Fraction of the median gap by which slice spacing may vary before we refuse.
MAX_SPACING_IRREGULARITY = 0.05

def read_series(paths: Sequence[Path | str],
                modality: Modality,
                allow_irregular: bool = False,
                to_suv: bool = True) -> MedicalVolume:

    """Assemble per-slice DICOM files into one volume with a correct affine."""
    import pydicom

    if not paths:
        raise DataIngestionError("no DICOM files given")

    slices = []
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path))
        except Exception as exc:
            raise DataIngestionError(f"cannot read DICOM {path!r}: {exc}") from exc
        if not hasattr(ds, "ImagePositionPatient"):
            raise DataIngestionError(f"{path!r} has no ImagePositionPatient")
        slices.append(ds)

    # Direction cosines: rows and columns of the image plane. The slice axis is
    # their cross product -- DICOM does not store it.
    iop = np.asarray(slices[0].ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    slice_dir = np.cross(row_dir, col_dir)

    # Sort by position ALONG THE SLICE NORMAL. Filenames, InstanceNumber and
    # SliceLocation are all unreliable; the patient-space position is not.
    def depth(ds) -> float:
        return float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float),
                            slice_dir))

    slices.sort(key=depth)
    depths = np.array([depth(s) for s in slices])

    if len(slices) < 2:
        raise DataIngestionError(f"series has only {len(slices)} slice(s)")

    gaps = np.diff(depths)
    dz = float(np.median(gaps))
    if dz <= 0:
        raise DataIngestionError(f"non-increasing slice positions (median gap {dz})")
    irregularity = float(np.abs(gaps - dz).max() / dz)
    if irregularity > MAX_SPACING_IRREGULARITY:
        message = (f"irregular slice spacing: median {dz:.3f} mm, worst deviation "
                   f"{irregularity:.1%} -- likely a missing slice, which would "
                   f"corrupt every physical measurement")
        if not allow_irregular:
            raise DataIngestionError(message)
        log.warning("%s (proceeding because allow_irregular=True)", message)

    # Stack in the sorted order, as [rows, cols, slices].
    array = np.stack([s.pixel_array for s in slices], axis=-1).astype(np.float32)

    # Stored values are NOT physical units until rescaled.
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        array = array * slope + intercept

    # PET is stored as activity concentration (Bq/mL). Convert to SUV so the
    # per-modality normalisation in M1.3 operates on the scale it assumes.
    if modality is Modality.PET and to_suv:
        from panorama.data.pet import suv_bw_factor
        array = array * suv_bw_factor(slices[0])

    row_mm, col_mm = (float(v) for v in slices[0].PixelSpacing)
    affine = np.eye(4)
    affine[:3, 0] = row_dir * row_mm
    affine[:3, 1] = col_dir * col_mm
    affine[:3, 2] = slice_dir * dz
    affine[:3, 3] = np.asarray(slices[0].ImagePositionPatient, dtype=float)

    log.info("series: %d slices, shape %s, spacing (%.2f, %.2f, %.2f) mm",
             len(slices), array.shape, row_mm, col_mm, dz)
    return MedicalVolume(array=array, affine=affine, modality=modality)