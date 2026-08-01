from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

from panorama.clinical.recist import Lesion, TimepointAssessment
from panorama.core.constants import Modality, RECISTResponse

TECHNIQUE = {
    Modality.CT: "contrast-enhanced CT",
    Modality.MRI: "multiparametric MRI",
    Modality.PET: "FDG PET",
}
IMPRESSION_LABEL = {
    RECISTResponse.CR: "complete response",
    RECISTResponse.PR: "partial response",
    RECISTResponse.SD: "stable disease",
    RECISTResponse.PD: "progressive disease",
    RECISTResponse.NE: "not evaluable",
}
LESION_OPENER = ["There is a", "A", "Noted is a", "Again seen is a"]
MEASURE_PHRASE = ["measuring {d:.0f} mm in longest diameter",
                  "which measures {d:.0f} mm",
                  "{d:.0f} mm in greatest dimension"]
TREND_PHRASE = {
    "up":   ["increased from {p:.0f} mm", "larger than the prior {p:.0f} mm",
             "grown from {p:.0f} mm"],
    "down": ["decreased from {p:.0f} mm", "smaller than the prior {p:.0f} mm",
             "reduced from {p:.0f} mm"],
    "same": ["unchanged from {p:.0f} mm", "stable at the prior {p:.0f} mm"],
}


@dataclass
class StructuredReport:
    """The FACTS. Prose is rendered from this; evaluation compares against it."""

    study_id: str
    patient_id: str
    acquired_on: date
    modalities: list[Modality]
    lesions: list[Lesion]
    sld_mm: float
    response: RECISTResponse
    rationale: str
    baseline_sld_mm: float | None = None
    nadir_sld_mm: float | None = None
    prior_sld_mm: float | None = None
    prior_study_id: str | None = None
    prior_lesion_mm: dict[str, float] = field(default_factory=dict)
    new_lesion: bool = False


def _trend(current: float, prior: float | None) -> str | None:
    if prior is None:
        return None
    if current > prior + 1e-6:
        return "up"
    if current < prior - 1e-6:
        return "down"
    return "same"


def render(report: StructuredReport, rng: random.Random | None = None) -> str:
    """Render a structured report as radiology prose, with phrasing variation."""
    rng = rng or random.Random(0)
    pick = rng.choice

    mods = ", ".join(TECHNIQUE[m] for m in report.modalities)
    lines = [f"TECHNIQUE: {mods} of the chest, abdomen and pelvis.", ""]

    if report.prior_study_id:
        lines += [f"COMPARISON: prior study {report.prior_study_id}.", ""]
    else:
        lines += ["COMPARISON: none available; this is the baseline study.", ""]


    lines.append("FINDINGS:")
    for lesion in report.lesions:
        d = lesion.longest_diameter_mm
        if d <= 0:
            lines.append(f"- The previously noted {lesion.organ} lesion has resolved.")
            continue
        organ = lesion.organ
        noun = organ if organ.endswith(("node", "mass", "nodule")) else f"{organ} lesion"
        sentence = (f"{pick(LESION_OPENER)} {noun} "
                    f"{pick(MEASURE_PHRASE).format(d=d)}")
        prior = report.prior_lesion_mm.get(lesion.lesion_id)
        direction = _trend(d, prior)
        if direction:
            sentence += f", {pick(TREND_PHRASE[direction]).format(p=prior)}"
        lines.append(sentence + ".")


    if report.new_lesion:
        lines.append("- A new lesion is identified, not present on the prior study.")

    lines += ["", f"Sum of longest diameters: {report.sld_mm:.0f} mm."]
    if report.baseline_sld_mm is not None:
        lines[-1] = (lines[-1][:-1] +
                     f" (baseline {report.baseline_sld_mm:.0f} mm, "
                     f"nadir {report.nadir_sld_mm:.0f} mm).")

    lines += ["", "IMPRESSION:",
              f"{IMPRESSION_LABEL[report.response].capitalize()} by RECIST 1.1 -- "
              f"{report.rationale}."]
    return "\n".join(lines)


def build_report(patient_id: str, tp: TimepointAssessment, acquired_on: date,
                 modalities: list[Modality],
                 baseline_sld: float | None = None,
                 nadir_sld: float | None = None,
                 prior: TimepointAssessment | None = None) -> StructuredReport:
    return StructuredReport(
        study_id=tp.study_id, patient_id=patient_id, acquired_on=acquired_on,
        modalities=list(modalities), lesions=list(tp.lesions), sld_mm=tp.sld_mm,
        response=tp.response, rationale=tp.rationale,
        baseline_sld_mm=baseline_sld, nadir_sld_mm=nadir_sld,
        prior_sld_mm=prior.sld_mm if prior else None,
        prior_study_id=prior.study_id if prior else None,
        prior_lesion_mm={l.lesion_id: l.longest_diameter_mm for l in prior.lesions}
                        if prior else {},
        new_lesion=tp.new_lesion)
