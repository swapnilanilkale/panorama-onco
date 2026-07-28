from __future__ import annotations

from dataclasses import replace

import numpy as np

from panorama.core.constants import IntensityNorm, Modality
from panorama.core.exceptions import DataIngestionError
from panorama.data.volume import MedicalVolume

EPS = 1e-8


def normalize_ct(array: np.ndarray, window=IntensityNorm.CT_HU_WINDOW) -> np.ndarray:
    """CT: FIXED Hounsfield window -> [0,1]. Same HU always maps to same value."""
    lo, hi = window
    x = np.clip(array.astype(np.float32), lo, hi)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def normalize_mri(array: np.ndarray, percentiles=IntensityNorm.MRI_CLIP_PERCENTILES) -> np.ndarray:
    """MRI: no absolute units -> clip outliers, then PER-VOLUME z-score."""
    x = array.astype(np.float32)
    lo, hi = np.percentile(x, percentiles)
    x = np.clip(x, lo, hi)
    return ((x - x.mean()) / (x.std() + EPS)).astype(np.float32)


def normalize_pet(array: np.ndarray, clip=IntensityNorm.PET_SUV_CLIP) -> np.ndarray:
    """PET (already SUV): fixed clip -> [0,1], preserving cross-patient meaning."""
    lo, hi = clip
    x = np.clip(array.astype(np.float32), lo, hi)
    return ((x - lo) / (hi - lo)).astype(np.float32)


_NORMALIZERS = {
    Modality.CT: normalize_ct,
    Modality.MRI: normalize_mri,
    Modality.PET: normalize_pet,
}


def normalize(volume: MedicalVolume) -> MedicalVolume:
    """Dispatch to the right normalizer for this volume's modality."""
    fn = _NORMALIZERS.get(volume.modality)
    if fn is None:
        raise DataIngestionError(f"No normalizer for modality {volume.modality.value!r}")
    return replace(volume, array=fn(volume.array))