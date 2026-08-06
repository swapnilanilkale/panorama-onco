from __future__ import annotations

import numpy as np

from panorama.core.constants import Modality
from panorama.data.volume import MedicalVolume
from panorama.core.exceptions import MissingModalityError

DEFAULT_PATCH = (96, 96, 96)


def voxel_to_world(affine: np.ndarray, voxel: np.ndarray) -> np.ndarray:
    """Voxel index -> physical position in millimetres."""
    return (affine @ np.array([*voxel, 1.0]))[:3]


def world_to_voxel(affine: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Physical position in millimetres -> (fractional) voxel index."""
    return (np.linalg.inv(affine) @ np.array([*world, 1.0]))[:3]


def crop_with_pad(array: np.ndarray, origin, patch_size, pad_value: float = 0.0) -> np.ndarray:
    """Crop [origin, origin+patch_size), zero-padding wherever we fall outside."""
    out = np.full(patch_size, pad_value, dtype=array.dtype)
    src_lo = np.maximum(origin, 0)
    src_hi = np.minimum(np.array(origin) + patch_size, array.shape)
    if np.any(src_hi <= src_lo):
        return out                                    # patch entirely outside
    dst_lo = src_lo - origin
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        array[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def sample_center_voxel(array: np.ndarray, rng: np.random.Generator,
                        fg_threshold: float | None = None,
                        fg_probability: float = 0.8) -> np.ndarray:
    """Pick a crop centre, biased toward foreground so we don't train on air."""
    if fg_threshold is not None and rng.random() < fg_probability:
        fg = np.argwhere(array > fg_threshold)
        if len(fg) > 0:
            return fg[rng.integers(len(fg))].astype(float)
    return np.array([rng.integers(s) for s in array.shape], dtype=float)


def crop_around_world_point(volume: MedicalVolume, centre_world: np.ndarray,
                            patch_size=DEFAULT_PATCH) -> np.ndarray:
    """Crop a patch centred on a PHYSICAL point, whatever this volume's grid is."""
    centre_vox = world_to_voxel(volume.affine, centre_world)
    origin = np.round(centre_vox - np.array(patch_size) / 2.0).astype(int)
    return crop_with_pad(volume.array, origin, patch_size)


def sample_study_patch(volumes: dict[Modality, MedicalVolume],
                       rng: np.random.Generator,
                       patch_size=DEFAULT_PATCH,
                       fg_threshold: float | None = None,
                       reference: Modality | None = None
                       ) -> tuple[dict[Modality, np.ndarray], np.ndarray]:
    """Sample ONE physical location and crop every stream around it.

    The crop centre is chosen from a REFERENCE modality. This must be explicit:
    `fg_threshold` is calibrated against one modality's normalised range, and
    silently using whichever stream came first in the dict makes foreground
    sampling depend on insertion order.
    """
    if reference is not None and reference in volumes:
        ref_volume = volumes[reference]
    else:
        # Prefer CT: its fixed HU window gives a stable, scanner-independent
        # scale for the threshold. Fall back to whatever is present.
        ref_volume = next((volumes[m] for m in Modality.imaging_streams()
                           if m in volumes), None)
        if ref_volume is None:
            raise MissingModalityError("study has no imaging streams to sample from")

    centre_vox = sample_center_voxel(ref_volume.array, rng, fg_threshold)
    centre_world = voxel_to_world(ref_volume.affine, centre_vox)
    patches = {m: crop_around_world_point(v, centre_world, patch_size)
               for m, v in volumes.items()}
    return patches, centre_world