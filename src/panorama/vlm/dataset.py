"""Pair each imaging patch with its study's radiology report."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from panorama.core.constants import RECISTResponse
from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.dataset import MultiModalPatchDataset
from panorama.data.patches import DEFAULT_PATCH
from panorama.data.resample import DEFAULT_SPACING_MM
from panorama.data.schema import Study
from panorama.vlm.tokenizer import ReportTokenizer

log = get_logger(__name__)

RESPONSE_INDEX = {r: i for i, r in enumerate(RECISTResponse)}


class ImageReportDataset(Dataset):
    """Wraps MultiModalPatchDataset, attaching the study's tokenized report."""

    def __init__(self, studies: Sequence[Study], corpus: dict[str, dict],
                 tokenizer: ReportTokenizer,
                 lesions: dict[str, list] | None = None, **patch_kwargs) -> None:
        paired = [s for s in studies if s.study_id in corpus]
        if not paired:
            raise ConfigError("no study in this split has a report in the corpus")
        if len(paired) < len(studies):
            log.warning("%d of %d studies have no report and were dropped",
                        len(studies) - len(paired), len(studies))

        self.patches = MultiModalPatchDataset(paired, **patch_kwargs)
        self.corpus = corpus
        self.tokenizer = tokenizer
        # World coordinates of each lesion, for crop-local probe targets.
        self.lesions = lesions or {}
        # Half the crop's physical extent, in mm -- computed from the kwargs we
        # were given rather than reaching into the patch dataset's internals.
        crop = patch_kwargs.get("crop_size", DEFAULT_PATCH)
        spacing = patch_kwargs.get("target_spacing", DEFAULT_SPACING_MM)
        self.half_extent_mm = tuple(c * s / 2.0 for c, s in zip(crop, spacing))
        # Stable integer id per study: lets the collated batch mark which
        # samples are crops of the SAME study (see ContrastiveAlignment).
        self.study_index = {s.study_id: i for i, s in enumerate(paired)}

    def __len__(self) -> int:
        return len(self.patches)

    def set_epoch(self, epoch: int) -> None:
        self.patches.set_epoch(epoch)

    def __getitem__(self, idx: int) -> dict:
        sample = self.patches[idx]
        study_id = sample["study_id"]
        record = self.corpus[study_id]

        ids, mask = self.tokenizer.encode(record["report"])
        sample["token_ids"] = torch.tensor(ids, dtype=torch.long)
        sample["attention_mask"] = torch.tensor(mask, dtype=torch.long)
        sample["pair_id"] = torch.tensor(self.study_index[study_id], dtype=torch.long)
        sample["response"] = torch.tensor(
            RESPONSE_INDEX[RECISTResponse(record["response"])], dtype=torch.long)

        # Study-level targets: properties of the WHOLE study, so a single crop
        # cannot determine them (see extract_study_features for the fix).
        sample["sld_mm"] = torch.tensor(record["sld_mm"], dtype=torch.float32)
        sample["n_lesions"] = torch.tensor(float(record["n_lesions"]),
                                           dtype=torch.float32)

        # Crop-local target: the diameter of the lesion nearest this crop's
        # centre, and whether that centre falls inside the crop. This IS
        # determined by what the encoder sees.
        centre = sample["centre_mm"].numpy()
        nearest_mm, in_view = 0.0, 0.0
        study_lesions = self.lesions.get(study_id, [])
        if study_lesions:
            offsets = [np.abs(centre - np.asarray(l.centre_mm))
                       for l in study_lesions]
            dists = [float(np.linalg.norm(o)) for o in offsets]
            i = int(np.argmin(dists))
            nearest_mm = float(study_lesions[i].longest_diameter_mm)
            # A crop is a BOX: the lesion centre is inside iff every axis is.
            in_view = float(all(o <= h for o, h in
                                zip(offsets[i], self.half_extent_mm)))
        sample["local_lesion_mm"] = torch.tensor(nearest_mm, dtype=torch.float32)
        sample["local_in_view"] = torch.tensor(in_view, dtype=torch.float32)
        return sample