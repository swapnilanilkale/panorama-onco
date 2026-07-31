from panorama.clinical.recist import (
    Lesion, TimepointAssessment, assess_course, classify, first_progression)
from panorama.core.constants import RECISTResponse

R = RECISTResponse


def course(*sums, new_at=None):
    tps = [TimepointAssessment(f"S{i}", [Lesion(f"L{i}", s)],
                               new_lesion=(new_at == i))
           for i, s in enumerate(sums)]
    return assess_course(tps)


# ------------------------------------------------- the nadir rule (the big one)

def test_progression_is_measured_from_nadir_not_baseline():
    """A tumour that shrank then regrew is PD even while below baseline."""
    tps = course(100.0, 55.0, 90.0)
    assert tps[1].response is R.PR
    assert tps[2].response is R.PD          # 90 < 100 baseline, but +64% on nadir
    assert "nadir" in tps[2].rationale


def test_baseline_referenced_logic_would_get_it_wrong():
    """Documents WHY the nadir rule exists -- guards against a naive 'fix'."""
    sld, baseline, nadir = 90.0, 100.0, 55.0
    assert (sld - baseline) / baseline < 0                    # looks like improvement
    assert classify(sld, baseline, nadir)[0] is R.PD          # but is progression


def test_nadir_tracks_the_running_minimum():
    tps = course(100.0, 70.0, 50.0, 62.0)
    # 62 vs nadir 50 = +24%, +12mm -> PD, despite being far below baseline
    assert tps[3].response is R.PD


# ---------------------------------------------------------------- boundaries

def test_pr_threshold_is_inclusive_at_exactly_30_percent():
    assert classify(70.0, 100.0, 100.0)[0] is R.PR      # exactly -30%
    assert classify(70.1, 100.0, 100.0)[0] is R.SD      # a hair under


def test_pd_requires_both_percent_and_absolute():
    # exactly +20% and exactly +5mm -> PD
    assert classify(30.0, 200.0, 25.0)[0] is R.PD
    # +25% but only +2.5mm -> NOT PD (inside measurement error)
    assert classify(12.5, 200.0, 10.0)[0] is not R.PD
    # +4mm absolute but only +4% -> NOT PD
    assert classify(104.0, 200.0, 100.0)[0] is not R.PD


# ------------------------------------------------------------ special cases

def test_new_lesion_is_pd_however_much_the_sum_shrank():
    resp, why = classify(10.0, 100.0, 10.0, new_lesion=True)
    assert resp is R.PD and "new lesion" in why


def test_complete_response_when_all_lesions_resolve():
    tps = assess_course([TimepointAssessment("S0", [Lesion("L1", 30.0)]),
                         TimepointAssessment("S1", [Lesion("L1", 0.0)])])
    assert tps[1].response is R.CR


def test_baseline_timepoint_is_never_a_response():
    tps = course(100.0)
    assert tps[0].response is R.SD
    assert tps[0].rationale == "baseline assessment"


def test_stable_disease_sits_between_the_thresholds():
    for sld in (75.0, 100.0, 115.0):
        assert classify(sld, 100.0, 100.0)[0] is R.SD


# ------------------------------------------------------------------- course

def test_first_progression_returns_the_earliest_pd():
    tps = course(100.0, 60.0, 78.0, 95.0)
    idx = first_progression(tps)
    assert idx == 2                                   # not 3
    assert tps[idx].response is R.PD


def test_no_progression_returns_none():
    assert first_progression(course(100.0, 80.0, 75.0)) is None


def test_empty_course_is_handled():
    assert assess_course([]) == []
    assert first_progression([]) is None


def test_every_assessment_carries_a_rationale():
    """Reports must be able to state WHY, not just the label."""
    for tp in course(100.0, 60.0, 80.0, 120.0):
        assert tp.rationale and isinstance(tp.rationale, str)