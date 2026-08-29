"""Survival outcomes for the synthetic imaging cohort.

The cohort's own PFS is unusable for survival modelling: studies fall on a fixed
90-day schedule, so durations take three values and are identical between events
and censored patients (ADR-0013). Survival models rank by risk, which needs
duration to carry information.

This generates outcomes whose hazard depends on quantities VISIBLE in the images
-- baseline tumour burden, lesion count, and growth rate -- so an encoder that
fails has failed at something achievable, and a null result is interpretable.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from panorama.core.logging import get_logger
from panorama.data.schema import PatientTimeline
from panorama.survival.data import SurvivalOutcome

log = get_logger(__name__)

# Growth is weighted highest deliberately: it is the only driver a
# single-timepoint model cannot see, so it separates a sequence encoder from a
# per-study baseline.
COEFFICIENTS = {"burden": 0.9, "count": 0.4, "growth": 1.2}
MEDIAN_SURVIVAL_DAYS = 500.0


def _features(timeline: PatientTimeline,
              lesions_by_study: dict[str, list]) -> tuple[float, float, float] | None:
    """Baseline burden, lesion count, and growth ratio -- all image-derived."""
    known = [s for s in timeline.studies if s.study_id in lesions_by_study]
    if len(known) < 2:
        return None

    first = lesions_by_study[known[0].study_id]
    last = lesions_by_study[known[-1].study_id]
    burden = sum(l.longest_diameter_mm for l in first)
    final = sum(l.longest_diameter_mm for l in last)
    if burden <= 0:
        return None
    return burden, float(len(first)), final / burden


def simulate_outcomes(timelines: Sequence[PatientTimeline],
                      lesions_by_study: dict[str, list],
                      censoring_range_days: tuple[float, float] = (90.0, 1000.0),
                      seed: int = 1337) -> tuple[dict[str, SurvivalOutcome], dict]:
    """Generate (time, event) pairs from image-derived risk.

    Returns (outcomes by patient, ground truth) -- the ground truth carries the
    true risk scores and coefficients so a fitted model can be compared against
    the hazard that actually generated the data, which no real cohort permits.
    """
    rng = np.random.default_rng(seed)

    usable, rows = [], []
    for timeline in timelines:
        features = _features(timeline, lesions_by_study)
        if features is not None:
            usable.append(timeline.patient_id)
            rows.append(features)
    if not rows:
        raise ValueError("no timeline has two timepoints with lesion data")

    table = np.asarray(rows)                       # [n, 3]

    def standardise(column: np.ndarray) -> np.ndarray:
        return (column - column.mean()) / (column.std() + 1e-8)

    risk = (COEFFICIENTS["burden"] * standardise(table[:, 0])
            + COEFFICIENTS["count"] * standardise(table[:, 1])
            + COEFFICIENTS["growth"] * standardise(table[:, 2]))

    # Exponential survival under proportional hazards.
    true_time = rng.exponential(MEDIAN_SURVIVAL_DAYS / np.exp(risk))
    censor_at = rng.uniform(*censoring_range_days, size=len(risk))
    observed = np.minimum(true_time, censor_at)
    event = true_time <= censor_at

    outcomes = {
        pid: SurvivalOutcome(pid, int(round(t)), bool(e), endpoint="OS")
        for pid, t, e in zip(usable, observed, event)
    }
    log.info("simulated outcomes for %d patients: %d events (%.0f%%), "
             "median observed %.0f days, %d distinct durations",
             len(outcomes), int(event.sum()), 100 * event.mean(),
             float(np.median(observed)), len(np.unique(np.round(observed))))

    return outcomes, {
        "patient_ids": usable,
        "true_risk": risk,
        "features": table,
        "coefficients": COEFFICIENTS,
        "feature_names": ("baseline_burden_mm", "lesion_count", "growth_ratio"),
    }