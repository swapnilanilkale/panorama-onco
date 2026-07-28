from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panorama.core.constants import Modality
from panorama.core.exceptions import DataIngestionError


@dataclass
class MedicalVolume:
    """Voxels plus the geometry needed to interpret them in millimetres."""

    array: np.ndarray      # 3D voxel data
    affine: np.ndarray     # 4x4 matrix: voxel index -> physical position (mm)
    modality: Modality

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        """Physical size of one voxel along each axis."""
        return tuple(float(np.linalg.norm(self.affine[:3, i])) for i in range(3))

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        """Physical size of the whole scanned box."""
        return tuple(s * n for s, n in zip(self.spacing_mm, self.shape))

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing_mm
        return sx * sy * sz

    def tumor_volume_ml(self, mask: np.ndarray) -> float:
        """Volume of a segmented region, in millilitres (1 mL = 1000 mm^3)."""
        if mask.shape != self.array.shape:
            raise ValueError(f"mask shape {mask.shape} != volume shape {self.array.shape}")
        return float(mask.sum()) * self.voxel_volume_mm3 / 1000.0


def load_nifti(path: Path | str, modality: Modality) -> MedicalVolume:
    """Read a .nii / .nii.gz file into a MedicalVolume."""
    import nibabel as nib

    try:
        img = nib.load(str(path))
    except Exception as exc:
        raise DataIngestionError(f"Could not read NIfTI {path!r}: {exc}") from exc

    # asanyarray(dataobj) keeps the on-disk dtype (int16 for CT).
    # get_fdata() would silently upcast to float64 -- 4x the RAM, for nothing.
    array = np.asanyarray(img.dataobj)
    if array.ndim != 3:
        raise DataIngestionError(f"Expected a 3D volume, got shape {array.shape} in {path!r}")

    return MedicalVolume(array=array, affine=np.asarray(img.affine), modality=modality)