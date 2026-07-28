from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from panorama.core.constants import Modality
from panorama.core.exceptions import MissingModalityError


@dataclass
class Study:
    """One patient, one timepoint, one or more co-acquired scans."""

    patient_id: str
    study_id: str
    acquired_on: date
    volumes: dict[Modality, Path] = field(default_factory=dict)

    @property
    def present(self) -> tuple[Modality, ...]:
        return tuple(m for m in Modality.imaging_streams() if m in self.volumes)

    def modality_mask(self) -> tuple[int, ...]:
        """1 = stream present, 0 = absent, in canonical CT/MRI/PET order."""
        return tuple(int(m in self.volumes) for m in Modality.imaging_streams())

    def path(self, modality: Modality) -> Path:
        if modality not in self.volumes:
            raise MissingModalityError(
                f"{modality.value} missing for study {self.study_id!r} "
                f"(patient {self.patient_id!r}); present: {[m.value for m in self.present]}"
            )
        return self.volumes[modality]


@dataclass
class PatientTimeline:
    """All studies for one patient, ordered in time."""

    patient_id: str
    studies: list[Study] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.studies.sort(key=lambda s: s.acquired_on)

    def __len__(self) -> int:
        return len(self.studies)

    @property
    def baseline(self) -> Study:
        if not self.studies:
            raise ValueError(f"No studies for patient {self.patient_id!r}")
        return self.studies[0]

    def days_since_baseline(self, study: Study) -> int:
        return (study.acquired_on - self.baseline.acquired_on).days

    def pairs(self) -> list[tuple[Study, Study]]:
        """Consecutive (prior, current) pairs -- the unit of RECIST comparison."""
        return list(zip(self.studies, self.studies[1:]))