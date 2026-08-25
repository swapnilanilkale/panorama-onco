"""Study pairs with nadir-aware RECIST targets.

Each example is (prior study, current study) plus the structured fields a report
must contain. Baselines are included with no prior -- they are roughly a third
of all studies, and excluding them would also leave the head's `no_prior` path
untrained.

The target is computed ONCE PER TIMELINE, not per pair. RECIST progression is
measured against the running nadir, which may lie at an intermediate timepoint
the pair cannot see; labelling from (prior, current) alone calls a rebounding
patient stable.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from panorama.clinical.recist import TimepointAssessment, assess_course
from panorama.core.constants import RECISTResponse
from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.dataset import MultiModalPatchDataset
from panorama.data.schema import Study
from panorama.data.splits import build_timelines
from panorama.vlm.report_head import MAX_LESIONS

log = get_logger(__name__)

RESPONSE_INDEX = {r: i for i, r in enumerate(RECISTResponse)}


class StudyPairDataset(Dataset):
    """One example per study: its own crop, its prior's crop, and the targets."""

    def __init__(self, studies: Sequence[Study],
                 lesions_by_study: dict[str, list],
                 organs: Sequence[str],
                 max_lesions: int = MAX_LESIONS,
                 **patch_kwargs) -> None:
        known = [s for s in studies if s.study_id in lesions_by_study]
        if not known:
            raise ConfigError("no study in this split has lesion ground truth")

        self.lesions = lesions_by_study
        self.organ_index = {name: i for i, name in enumerate(organs)}
        self.max_lesions = max_lesions

        # Timelines are built from THIS split only, so a patient's studies
        # never span splits (they cannot -- splits are patient-level -- but
        # building here keeps the invariant local and obvious).
        self.examples: list[tuple[Study, Study | None, dict]] = []
        for timeline in build_timelines(known):
            course = assess_course([
                TimepointAssessment(s.study_id, lesions_by_study[s.study_id])
                for s in timeline.studies])
            for i, (study, assessment) in enumerate(zip(timeline.studies, course)):
                prior = timeline.studies[i - 1] if i > 0 else None
                self.examples.append((study, prior, self._targets(assessment)))

        n_baseline = sum(1 for _, prior, _ in self.examples if prior is None)
        log.info("%d examples (%d baselines, %d with prior) from %d studies",
                 len(self.examples), n_baseline,
                 len(self.examples) - n_baseline, len(known))

        self.patches = MultiModalPatchDataset(known, **patch_kwargs)
        self.study_row = {s.study_id: i for i, s in enumerate(known)}

    def _targets(self, assessment: TimepointAssessment) -> dict:
        lesions = assessment.lesions[:self.max_lesions]
        diameters = np.zeros(self.max_lesions, dtype=np.float32)
        organs = np.zeros(self.max_lesions, dtype=np.int64)
        for i, lesion in enumerate(lesions):
            diameters[i] = lesion.longest_diameter_mm
            organs[i] = self.organ_index.get(lesion.organ, 0)
        return {
            "lesion_count": len(lesions),
            "diameters_mm": diameters,
            "organs": organs,
            "new_lesion": float(assessment.new_lesion),
            # Derived from the WHOLE course by assess_course, so the nadir is
            # the true running minimum.
            "response": RESPONSE_INDEX[assessment.response],
            "sld_mm": float(assessment.sld_mm),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.patches.set_epoch(epoch)

    def _crop(self, study: Study, offset: int) -> dict:
        """Sample a crop for one study. `offset` varies the RNG per role so a
        study used as both current and prior does not get identical crops."""
        row = self.study_row[study.study_id]
        return self.patches[row * self.patches.patches_per_study + offset]

    def __getitem__(self, idx: int) -> dict:
        study, prior, targets = self.examples[idx]

        current_sample = self._crop(study, 0)
        sample = {
            "image": current_sample["image"],
            "modality_mask": current_sample["modality_mask"],
            "study_id": study.study_id,
            "patient_id": study.patient_id,
        }

        if prior is None:
            # Shape-compatible placeholder; the gate makes its content irrelevant.
            sample["prior_image"] = torch.zeros_like(current_sample["image"])
            sample["prior_modality_mask"] = torch.zeros_like(
                current_sample["modality_mask"])
            sample["prior_present"] = torch.tensor(0.0)
            sample["prior_study_id"] = ""
        else:
            prior_sample = self._crop(prior, 1)
            sample["prior_image"] = prior_sample["image"]
            sample["prior_modality_mask"] = prior_sample["modality_mask"]
            sample["prior_present"] = torch.tensor(1.0)
            sample["prior_study_id"] = prior.study_id

        sample["lesion_count"] = torch.tensor(targets["lesion_count"],
                                              dtype=torch.long)
        sample["diameters_mm"] = torch.from_numpy(targets["diameters_mm"])
        sample["organs"] = torch.from_numpy(targets["organs"])
        sample["new_lesion"] = torch.tensor(targets["new_lesion"])
        sample["response"] = torch.tensor(targets["response"], dtype=torch.long)
        sample["sld_mm"] = torch.tensor(targets["sld_mm"])
        return sample