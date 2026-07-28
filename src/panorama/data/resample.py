from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage

from panorama.core.constants import Modality
from panorama.data.volume import MedicalVolume

DEFAULT_SPACING_MM = (1.5, 1.5, 1.5)


def resample(volume: MedicalVolume, target_spacing=DEFAULT_SPACING_MM) -> MedicalVolume:
    """Resample onto an isotropic grid so 1 voxel = the same mm in every stream."""
    old = np.asarray(volume.spacing_mm, dtype=np.float64)
    new = np.asarray(target_spacing, dtype=np.float64)
    zoom = old / new                       # >1 = upsample, <1 = downsample

    # Labels must use NEAREST: interpolating label 1 and 3 would invent label 2.
    is_label = volume.modality is Modality.SEG
    order = 0 if is_label else 1           # 0 = nearest, 1 = trilinear

    out = ndimage.zoom(volume.array.astype(np.float32), zoom, order=order,
                       mode="nearest", prefilter=False)
    if is_label:
        out = out.round().astype(volume.array.dtype)
    else:
        out = out.astype(np.float32)

    # The affine MUST be updated too, or the geometry silently becomes a lie.
    affine = volume.affine.copy()
    for i in range(3):
        affine[:3, i] = volume.affine[:3, i] * (new[i] / old[i])

    return replace(volume, array=out, affine=affine)