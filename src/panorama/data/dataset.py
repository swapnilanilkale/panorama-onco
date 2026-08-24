from __future__ import annotations

from collections.abc import Sequence
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from panorama.core.constants import Modality
from panorama.data.patches import DEFAULT_PATCH, sample_study_patch
from panorama.data.pipeline import preprocess
from panorama.data.resample import DEFAULT_SPACING_MM
from panorama.data.schema import Study
from panorama.data.volume import load_nifti


def stack_modalities(patches: dict[Modality, np.ndarray],
                     patch_size) -> tuple[np.ndarray, np.ndarray]:
    """dict-of-present-streams -> fixed [3,D,H,W] array + [3] presence mask."""
    streams = Modality.imaging_streams()
    image = np.zeros((len(streams), *patch_size), dtype=np.float32)
    mask = np.zeros(len(streams), dtype=np.float32)
    for i, m in enumerate(streams):
        if m in patches:
            image[i] = patches[m]
            mask[i] = 1.0
    return image, mask


class MultiModalPatchDataset(Dataset):
    """Yields one multi-modal 3D patch per index, as model-ready tensors."""
    def __init__(self,
                 studies: Sequence[Study],
                 crop_size=DEFAULT_PATCH,
                 target_spacing=DEFAULT_SPACING_MM,
                 fg_threshold: float | None = None,
                 patches_per_study: int = 1,
                 seed: int = 1337,
                 cache_volumes: bool = True,
                 cache_size: int = 16) -> None:
        self.studies = list(studies)
        self.crop_size = tuple(crop_size)
        self.target_spacing = tuple(target_spacing)
        self.fg_threshold = fg_threshold
        self.patches_per_study = patches_per_study
        self.seed = seed
        self.epoch = 0
        self.cache_size = cache_size
        # OrderedDict as an LRU: a whole-body study is ~130 MB preprocessed, so
        # an unbounded cache would grow to ~10 GB over an epoch. Bounding it
        # keeps the reuse that matters (consecutive crops of the same study)
        # without the memory.
        self._cache: OrderedDict[str, dict] | None = (
            OrderedDict() if cache_volumes else None)

    
    def __len__(self) -> int:
        return len(self.studies) * self.patches_per_study

    def set_epoch(self, epoch: int) -> None:
        """Call each epoch so the same index yields a different patch."""
        self.epoch = epoch

    def _volumes(self, study: Study) -> dict:
        if self._cache is not None and study.study_id in self._cache:
            self._cache.move_to_end(study.study_id)     # mark as recently used
            return self._cache[study.study_id]
        vols = {m: preprocess(load_nifti(p, m), self.target_spacing)
                for m, p in study.volumes.items()}
        if self._cache is not None:
            self._cache[study.study_id] = vols
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)          # evict least recent
        return vols

    def __getitem__(self, idx: int) -> dict:
        study = self.studies[idx // self.patches_per_study]

        # Deterministic per (epoch, index): reproducible, yet varies across epochs,
        # and each worker computes the same value without coordination.
        rng = np.random.default_rng((self.seed, self.epoch, idx))

        volumes = self._volumes(study)
        patches, centre = sample_study_patch(volumes, rng, self.crop_size, self.fg_threshold)
        image, mask = stack_modalities(patches, self.crop_size)
        

        return {
            "image": torch.from_numpy(image),          # [3, D, H, W]
            "modality_mask": torch.from_numpy(mask),   # [3]
            "centre_mm": torch.from_numpy(centre.astype(np.float32)),
            "patient_id": study.patient_id,
            "study_id": study.study_id,
        }