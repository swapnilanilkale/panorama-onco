from __future__ import annotations

from panorama.core.constants import Modality
from panorama.data.normalize import normalize
from panorama.data.resample import DEFAULT_SPACING_MM, resample
from panorama.data.volume import MedicalVolume


def preprocess(volume: MedicalVolume, target_spacing=DEFAULT_SPACING_MM) -> MedicalVolume:
    """Canonical preprocessing: geometry first, then intensity.

    Order matters. We resample BEFORE normalizing so that intensity statistics
    (the MRI mean/std) are computed on exactly the grid the model will see,
    and so interpolation happens in native physical units (HU, SUV).
    """
    out = resample(volume, target_spacing)
    if volume.modality is Modality.SEG:
        return out                      # labels are categories -- never normalized
    return normalize(out)


def preprocess_study(volumes: dict[Modality, MedicalVolume],
                     target_spacing=DEFAULT_SPACING_MM) -> dict[Modality, MedicalVolume]:
    """Apply preprocessing to every stream present in a study."""
    return {m: preprocess(v, target_spacing) for m, v in volumes.items()}