"""RECIST 1.1 response assessment.

The standard by which oncologists decide, reproducibly, whether a treatment is
working. Everything in Aims 2 and 3 depends on getting this right: the report
generator must state these categories correctly, and progression-free survival
is defined by the first PD event.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from panorama.core.constants import (
    RECIST_PD_ABSOLUTE_MM,
    RECIST_PD_GROWTH_FRACTION,
    RECIST_PR_SHRINK_FRACTION,
    RECISTResponse,
)


@dataclass(frozen=True)
class Lesion:
    """A measurable target lesion. RECIST measures the LONGEST diameter."""

    lesion_id: str
    longest_diameter_mm: float
    organ: str = "unspecified"

    @property
    def is_resolved(self) -> bool:
        return self.longest_diameter_mm <= 0.0


@dataclass
class TimepointAssessment:
    """Measurements at one visit, plus the derived response."""

    study_id: str
    lesions: list[Lesion] = field(default_factory=list)
    new_lesion: bool = False
    response: RECISTResponse | None = None
    rationale: str = ""

    @property
    def sld_mm(self) -> float:
        """Sum of longest diameters -- the quantity RECIST tracks."""
        return sum(l.longest_diameter_mm for l in self.lesions)

    @property
    def all_resolved(self) -> bool:
        return bool(self.lesions) and all(l.is_resolved for l in self.lesions)


def classify(sld_mm: float, baseline_mm: float, nadir_mm: float,
             new_lesion: bool = False,
             all_resolved: bool = False) -> tuple[RECISTResponse, str]:
    """Assign a RECIST 1.1 category to one timepoint.

    Note the asymmetry, which is the crux of the standard:
      * RESPONSE (PR) is measured against the BASELINE.
      * PROGRESSION (PD) is measured against the NADIR -- the smallest sum
        recorded so far. A tumour that shrank then regrew is progressing even
        while still below its baseline.
    """
    if new_lesion:
        return RECISTResponse.PD, "new lesion appeared"
    if all_resolved:
        return RECISTResponse.CR, "all target lesions resolved"

    if nadir_mm > 0:
        growth_fraction = (sld_mm - nadir_mm) / nadir_mm
        growth_mm = sld_mm - nadir_mm
        # BOTH thresholds required: the 5mm floor rejects measurement noise.
        if (growth_fraction >= RECIST_PD_GROWTH_FRACTION
                and growth_mm >= RECIST_PD_ABSOLUTE_MM):
            return RECISTResponse.PD, (
                f"sum increased {growth_fraction:.0%} ({growth_mm:+.1f} mm) "
                f"from nadir of {nadir_mm:.1f} mm")

    if baseline_mm > 0:
        shrink_fraction = (baseline_mm - sld_mm) / baseline_mm
        if shrink_fraction >= RECIST_PR_SHRINK_FRACTION:
            return RECISTResponse.PR, (
                f"sum decreased {shrink_fraction:.0%} from baseline of "
                f"{baseline_mm:.1f} mm")

    change = (sld_mm - baseline_mm) / baseline_mm if baseline_mm > 0 else 0.0
    return RECISTResponse.SD, (
        f"sum changed {change:+.0%} from baseline, meeting neither the "
        f"{RECIST_PR_SHRINK_FRACTION:.0%} response nor the "
        f"{RECIST_PD_GROWTH_FRACTION:.0%} progression threshold")


def assess_course(timepoints: list[TimepointAssessment]) -> list[TimepointAssessment]:
    """Classify a whole treatment course in order, tracking the nadir."""
    if not timepoints:
        return []

    baseline = timepoints[0].sld_mm
    nadir = baseline
    for i, tp in enumerate(timepoints):
        if i == 0:
            tp.response, tp.rationale = RECISTResponse.SD, "baseline assessment"
        else:
            tp.response, tp.rationale = classify(
                tp.sld_mm, baseline, nadir,
                new_lesion=tp.new_lesion, all_resolved=tp.all_resolved)
        nadir = min(nadir, tp.sld_mm)
    return timepoints


def first_progression(timepoints: list[TimepointAssessment]) -> int | None:
    """Index of the first PD -- the event that defines progression-free survival."""
    for i, tp in enumerate(timepoints):
        if tp.response is RECISTResponse.PD:
            return i
    return None