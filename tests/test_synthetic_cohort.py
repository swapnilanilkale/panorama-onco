"""Guards on the synthetic cohort's *labels*, not just its files.

A generator that produces plausible volumes but a degenerate label
distribution is worse than no generator: training curves look normal while the
model never sees the clinically important category.
"""
from collections import Counter

import pytest

from panorama.clinical.recist import TimepointAssessment, assess_course
from panorama.core.constants import RECISTResponse
from panorama.data.synthetic import TRAJECTORIES, read_lesions, write_cohort

R = RECISTResponse


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("cohort")
    lesions_csv = root / "lesions.csv"
    write_cohort(root / "raw", n_patients=16, max_studies=4, seed=0,
                 lesion_manifest=lesions_csv)
    return read_lesions(lesions_csv)


def courses(cohort):
    """patient_id -> assessed RECIST course."""
    by_patient: dict[str, list[str]] = {}
    for study_id in sorted(cohort):
        by_patient.setdefault(study_id.split("_")[0], []).append(study_id)
    return {pid: assess_course([TimepointAssessment(s, cohort[s]) for s in studies])
            for pid, studies in by_patient.items()}


def test_every_study_has_lesion_ground_truth(cohort):
    assert cohort
    assert all(lesions for lesions in cohort.values())


def test_lesions_are_large_enough_to_express_progression(cohort):
    """Below ~15mm SLD the 5mm absolute floor suppresses PD entirely."""
    for study_id, lesions in cohort.items():
        sld = sum(l.longest_diameter_mm for l in lesions)
        assert sld >= 15.0, f"{study_id} SLD {sld:.1f}mm is too small"


def test_cohort_contains_all_response_categories(cohort):
    """The whole point of the trajectory profiles."""
    seen = Counter(tp.response for course in courses(cohort).values()
                   for tp in course)
    for expected in (R.SD, R.PR, R.PD):
        assert seen[expected] > 0, f"no {expected.value} in cohort: {dict(seen)}"


def test_progression_is_not_vanishingly_rare(cohort):
    """A handful of PD examples trains a model that never predicts it."""
    seen = Counter(tp.response for course in courses(cohort).values()
                   for tp in course)
    total = sum(seen.values())
    assert seen[R.PD] / total >= 0.05, f"PD is only {seen[R.PD]}/{total}"


def test_all_anatomical_regions_occur(cohort):
    """Region bins must span the range centres are actually sampled from."""
    organs = {l.organ for lesions in cohort.values() for l in lesions}
    assert "unspecified" not in organs
    assert len(organs) >= 4, organs


def test_some_progression_is_detectable_only_from_the_nadir(cohort):
    """The rebound trajectory must actually appear: PD while BELOW baseline.

    This is the clinically hardest case and the one a baseline-referenced
    implementation gets wrong. If the cohort never contains it, the model
    can neither learn it nor be tested on it.
    """
    found = []
    for pid, course in courses(cohort).items():
        baseline = course[0].sld_mm
        for tp in course:
            if tp.response is R.PD and tp.sld_mm < baseline:
                found.append((pid, tp.study_id, tp.sld_mm, baseline))
    assert found, "no below-baseline PD in cohort -- the rebound case is missing"


def test_trajectories_are_declared_for_all_four_profiles():
    assert set(TRAJECTORIES) == {"responder", "stable", "progressor", "rebound"}
    assert all(len(v) == 4 for v in TRAJECTORIES.values())