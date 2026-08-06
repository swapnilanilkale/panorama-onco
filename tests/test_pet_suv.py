"""SUV conversion guards.

`suv_bw_factor` reads its inputs with getattr, so a plain namespace stands in
for a pydicom Dataset -- these tests need no DICOM files and no downloaded data.
"""
from types import SimpleNamespace

import pytest

from panorama.core.exceptions import DataIngestionError
from panorama.data.pet import suv_bw_factor

# Real tag values from QIN-BREAST-01-0001, 1991-09-01.
REAL = dict(Units="BQML", PatientWeight=77, SeriesTime="172300",
            dose=432151424, half_life=6588, start="154800.00")


def dataset(**overrides):
    values = {**REAL, **overrides}
    rpis = SimpleNamespace(RadionuclideTotalDose=values["dose"],
                           RadionuclideHalfLife=values["half_life"],
                           RadiopharmaceuticalStartTime=values["start"])
    return SimpleNamespace(
        Units=values["Units"], PatientWeight=values["PatientWeight"],
        SeriesTime=values["SeriesTime"],
        RadiopharmaceuticalInformationSequence=[rpis])


def test_known_series_gives_physiological_suv():
    """Anchor on a value clinicians recognise: normal liver is SUV 2-3.

    Checking a derived quantity against a PHYSIOLOGICAL reference is a far
    stronger test than checking it does not crash.
    """
    factor = suv_bw_factor(dataset())
    liver_bqml = 8000.0
    assert 2.0 <= liver_bqml * factor <= 3.0


def test_factor_is_stable():
    """Exact regression guard on the arithmetic."""
    assert suv_bw_factor(dataset()) == pytest.approx(3.2457e-04, rel=1e-4)


def test_decay_correction_is_applied():
    """Omitting decay inflates every SUV by ~1.8x at a 95-minute delay.

    A scan 95 min after injection has lost nearly one F-18 half-life (110 min).
    Using the INJECTED dose instead of the decayed dose is the single most
    common SUV implementation error.
    """
    factor = suv_bw_factor(dataset())
    naive = 77 * 1000.0 / 432151424          # no decay correction
    assert factor / naive == pytest.approx(1.82, abs=0.02)


def test_longer_uptake_delay_gives_a_larger_factor():
    """More decay -> less activity remaining -> bigger multiplier."""
    early = suv_bw_factor(dataset(SeriesTime="164800"))   # 60 min
    late = suv_bw_factor(dataset(SeriesTime="174800"))    # 120 min
    assert late > early


def test_scan_crossing_midnight():
    """Injection at 23:30, scan at 00:45 is 75 minutes, not negative."""
    factor = suv_bw_factor(dataset(start="233000", SeriesTime="004500"))
    assert factor > 0


def test_implausible_delay_is_rejected():
    with pytest.raises(DataIngestionError, match="implausible"):
        suv_bw_factor(dataset(start="000000", SeriesTime="230000"))


def test_missing_weight_is_rejected():
    ds = dataset()
    ds.PatientWeight = None
    with pytest.raises(DataIngestionError, match="PatientWeight"):
        suv_bw_factor(ds)


def test_missing_radiopharmaceutical_sequence_is_rejected():
    ds = dataset()
    ds.RadiopharmaceuticalInformationSequence = None
    with pytest.raises(DataIngestionError, match="Radiopharmaceutical"):
        suv_bw_factor(ds)


def test_missing_dose_names_the_tag():
    ds = dataset()
    ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose = None
    with pytest.raises(DataIngestionError, match="RadionuclideTotalDose"):
        suv_bw_factor(ds)


def test_heavier_patient_gives_proportionally_larger_factor():
    """SUV is body-weight normalised: double the weight, double the factor."""
    assert (suv_bw_factor(dataset(PatientWeight=154))
            == pytest.approx(2 * suv_bw_factor(dataset(PatientWeight=77))))