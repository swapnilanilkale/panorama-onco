from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from panorama.core.exceptions import ConfigError
from panorama.data.schema import PatientTimeline, Study


@dataclass
class CohortSplit:
    """Train/val/test study lists guaranteed disjoint at the PATIENT level."""

    train: list[Study]
    val: list[Study]
    test: list[Study]

    def patient_ids(self) -> dict[str, set[str]]:
        return {name: {s.patient_id for s in getattr(self, name)}
                for name in ("train", "val", "test")}

    def summary(self) -> str:
        ids = self.patient_ids()
        rows = [f"{name:5} {len(getattr(self, name)):>5} studies "
                f"{len(ids[name]):>4} patients" for name in ("train", "val", "test")]
        return "\n".join(rows)


def patient_level_split(studies: Sequence[Study],
                        val_fraction: float = 0.15,
                        test_fraction: float = 0.15,
                        seed: int = 1337) -> CohortSplit:
    """Split a cohort by PATIENT, never by study.

    A patient contributes ALL of their timepoints to exactly one split. Study-level
    splitting would place a patient's baseline in train and their follow-up in val:
    same tumour, same anatomy, same scanner -- so the model can score well by
    recognising the patient rather than the disease.
    """
    if not 0.0 <= val_fraction + test_fraction < 1.0:
        raise ConfigError(
            f"val_fraction + test_fraction must be in [0, 1), got "
            f"{val_fraction} + {test_fraction}")

    patients = sorted({s.patient_id for s in studies})   # sorted => deterministic
    if not patients:
        raise ConfigError("no studies provided")

    order = np.random.default_rng(seed).permutation(patients)
    n_test = int(round(len(patients) * test_fraction))
    n_val = int(round(len(patients) * val_fraction))

    test_ids = set(order[:n_test])
    val_ids = set(order[n_test:n_test + n_val])
    train_ids = set(order[n_test + n_val:])

    def take(ids: set[str]) -> list[Study]:
        return [s for s in studies if s.patient_id in ids]

    split = CohortSplit(train=take(train_ids), val=take(val_ids), test=take(test_ids))
    assert_no_patient_leakage(split)
    return split


def assert_no_patient_leakage(split: CohortSplit) -> None:
    """Raise if any patient appears in more than one split."""
    ids = split.patient_ids()
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise ConfigError(
                f"patient leakage between {a} and {b}: {len(overlap)} patients, "
                f"e.g. {sorted(overlap)[:3]}")


def build_timelines(studies: Sequence[Study]) -> list[PatientTimeline]:
    """Group studies into per-patient chronological timelines (for Aims 2 and 3)."""
    by_patient: dict[str, list[Study]] = {}
    for s in studies:
        by_patient.setdefault(s.patient_id, []).append(s)
    return [PatientTimeline(pid, sorted_studies)
            for pid, sorted_studies in sorted(by_patient.items())]