"""Memory-mapped access to preprocessed volumes.

Resampling a 512x512x83 CT to a 350x350x136 grid costs ~0.8s. Doing it inside
`__getitem__` means paying it on every batch of every epoch: measured at 12s per
batch of 8 two-modality studies, which pinned a T4 at 0.05 it/s -- slower than
CPU-only training, because the GPU sat idle waiting.

Precomputing once and memory-mapping reduces the same batch to ~3ms. `mmap_mode`
matters: a 350x350x136 float32 volume is 65 MB, and loading it fully to take one
32^3 crop reads 500x more than needed. Memory mapping pages in only the touched
blocks and lets the OS page cache keep hot studies resident.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama.core.constants import Modality
from panorama.core.exceptions import DataIngestionError
from panorama.data.schema import Study
from panorama.data.volume import MedicalVolume


def volume_paths(root: Path | str, study: Study,
                 modality: Modality) -> tuple[Path, Path]:
    directory = (Path(root) / study.patient_id / study.acquired_on.isoformat())
    return (directory / f"{modality.value}.npy",
            directory / f"{modality.value}_affine.npy")


def has_precomputed(root: Path | str, study: Study) -> bool:
    return all(volume_paths(root, study, m)[0].is_file() for m in study.volumes)


def load_precomputed(root: Path | str, study: Study,
                     modality: Modality) -> MedicalVolume:
    """Memory-mapped volume. The array is NOT read until it is indexed."""
    array_path, affine_path = volume_paths(root, study, modality)
    if not array_path.is_file():
        raise DataIngestionError(f"no precomputed volume at {array_path}")
    return MedicalVolume(
        array=np.load(array_path, mmap_mode="r"),
        affine=np.load(affine_path),
        modality=modality)