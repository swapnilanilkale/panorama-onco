"""Survival outcomes: (time, event) pairs with censoring.

Censoring is what makes survival analysis different from classification. A
patient who has not progressed by their last visit is NOT a negative example --
they are an unknown whose event time is somewhere after that visit. Treating
them as negative teaches the model that short follow-up means low risk, which
is a property of the study, not the patient. 

Note on the synthetic imaging cohort: its studies fall on a fixed 90-day
schedule, so PFS durations take only three distinct values and are identical
between events and censored patients (both median 180 days, range 180-270).
Survival models rank by risk, which requires duration to carry information --
so that cohort cannot validate a survival model, and `simulate_cohort` is used
instead. See ADR-0013.
"""
from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from panorama.clinical.recist import TimepointAssessment, assess_course, first_progression
from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.schema import PatientTimeline

log = get_logger(__name__)


@dataclass(frozen=True)
class SurvivalOutcome:
    """One patient's time-to-event record.

    `duration` is time from baseline to the event OR to last contact.
    `event` is True only when the event was OBSERVED.
    """

    patient_id: str
    duration_days: int
    event: bool
    endpoint: str = "PFS"

    def __post_init__(self) -> None:
        if self.duration_days < 0:
            raise ValueError(f"{self.patient_id}: negative duration")

    @property
    def duration_months(self) -> float:
        return self.duration_days / 30.44


def pfs_from_timeline(timeline: PatientTimeline,
                      lesions_by_study: dict[str, list]) -> SurvivalOutcome | None:
    """Progression-free survival derived from a RECIST course.

    The event is the FIRST progressive disease assessment, whose timing depends
    on the running nadir -- so it must come from `assess_course` over the whole
    timeline, not from comparing consecutive pairs.

    A patient with no PD is CENSORED at their last scan: we know they were
    progression-free up to that point and nothing after it.
    """
    known = [s for s in timeline.studies if s.study_id in lesions_by_study]
    if len(known) < 2:
        return None                     # a single scan gives no follow-up interval

    course = assess_course([
        TimepointAssessment(s.study_id, lesions_by_study[s.study_id]) for s in known])
    index = first_progression(course)
    baseline = known[0].acquired_on

    if index is None:
        last = known[-1].acquired_on
        return SurvivalOutcome(timeline.patient_id, (last - baseline).days,
                               event=False)
    return SurvivalOutcome(timeline.patient_id,
                           (known[index].acquired_on - baseline).days, event=True)


def read_outcomes(path: Path | str,
                  patient_column: str = "patient_id",
                  duration_column: str = "duration_days",
                  event_column: str = "event",
                  endpoint: str = "OS") -> dict[str, SurvivalOutcome]:
    """Load outcomes from a clinical CSV.

    Real survival data comes from a trial's follow-up records, not from imaging.
    The event column must distinguish OBSERVED events from censoring; a column
    that only says "alive/dead at last contact" without a date is not usable.
    """
    outcomes: dict[str, SurvivalOutcome] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            missing = [c for c in (patient_column, duration_column, event_column)
                       if not row.get(c)]
            if missing:
                raise ConfigError(f"{path}:{line_no} missing columns {missing}")
            flag = str(row[event_column]).strip().lower()
            outcomes[row[patient_column]] = SurvivalOutcome(
                patient_id=row[patient_column],
                duration_days=int(float(row[duration_column])),
                event=flag in ("1", "true", "yes", "event", "dead", "progressed"),
                endpoint=endpoint)
    return outcomes


def cohort_summary(outcomes: Sequence[SurvivalOutcome]) -> dict:
    """Descriptive statistics -- always report these alongside any C-index.

    A cohort with few events cannot support a survival model however large it
    is: the effective sample size is the NUMBER OF EVENTS, not the number of
    patients. The usual rule of thumb is at least 10 events per covariate.
    """
    if not outcomes:
        return {"n": 0, "n_events": 0, "event_rate": 0.0}

    durations = np.array([o.duration_days for o in outcomes], dtype=float)
    events = np.array([o.event for o in outcomes], dtype=bool)
    return {
        "n": len(outcomes),
        "n_events": int(events.sum()),
        "n_censored": int((~events).sum()),
        "event_rate": float(events.mean()),
        "median_followup_days": float(np.median(durations)),
        "median_event_days": (float(np.median(durations[events]))
                              if events.any() else None),
        "max_covariates": int(events.sum() // 10),   # 10-events-per-covariate rule
    }


def simulate_cohort(n: int = 400, n_features: int = 8,
                    baseline_hazard: float = 1 / 24.0,
                    censoring_rate: float = 0.4,
                    seed: int = 1337) -> tuple[np.ndarray, list[SurvivalOutcome]]:
    """Synthetic survival data with a KNOWN true hazard.

    The module can then be validated where the answer is known: a correct Cox
    implementation must recover the simulated coefficients, and the C-index must
    approach the concordance achievable given the noise. Real cohorts cannot
    test this, because their true hazard is unobservable.

    Returns (features [n, n_features], outcomes).
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, n_features))
    # Only the first three features carry signal -- the rest are distractors.
    coefficients = np.zeros(n_features)
    coefficients[:3] = [0.8, -0.5, 0.3]

    risk = features @ coefficients
    # Exponential survival with proportional hazards: T ~ Exp(h0 * exp(risk))
    true_time = rng.exponential(1.0 / (baseline_hazard * np.exp(risk)))
    # Administrative censoring: uniform follow-up windows.
    scale = np.quantile(true_time, 1.0 - censoring_rate) * 1.5
    follow_up = rng.uniform(0.0, scale, n)

    observed = np.minimum(true_time, follow_up)
    events = true_time <= follow_up

    outcomes = [SurvivalOutcome(f"SIM{i:04d}", int(round(t * 30.44)), bool(e))
                for i, (t, e) in enumerate(zip(observed, events))]
    log.info("simulated %d patients: %d events (%.0f%%), true coefficients %s",
             n, int(events.sum()), 100 * events.mean(), coefficients[:3])
    return features, outcomes