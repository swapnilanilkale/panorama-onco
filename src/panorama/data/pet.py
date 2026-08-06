"""Convert PET activity concentration to SUV.

PET pixel data is stored as Becquerels per millilitre -- raw radioactivity,
which depends on how much tracer was injected, how long ago, and how big the
patient is. SUV (Standardized Uptake Value) normalises all three, which is what
makes ~2.5 mean "suspicious" across patients and scanners.

Without this conversion the per-modality normalisation from M1.3 is meaningless:
it clips to an SUV range of [0, 25] while the data is in the thousands.
"""
from __future__ import annotations

from panorama.core.exceptions import DataIngestionError
from panorama.core.logging import get_logger

log = get_logger(__name__)


def _hhmmss_to_seconds(value: str | float) -> int:
    """DICOM TM values look like '154800.00' -- HHMMSS with optional fraction."""
    text = str(value).split(".")[0].zfill(6)
    try:
        return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:6])
    except ValueError as exc:
        raise DataIngestionError(f"cannot parse DICOM time {value!r}") from exc


def suv_bw_factor(dataset) -> float:
    """Multiplier converting Bq/mL to body-weight SUV (g/mL) for one series.

    SUV_bw = concentration [Bq/mL] * body_weight [g] / decayed_dose [Bq]

    The decay correction is the step implementations most often omit. F-18 has a
    110-minute half-life and scans typically start 60-90 minutes after
    injection, so the activity remaining is roughly half the injected dose --
    omitting it inflates every SUV by nearly 2x.
    """
    units = str(getattr(dataset, "Units", "")).upper()
    if units not in ("BQML", "CNTS", ""):
        log.warning("unexpected PET Units %r -- SUV conversion assumes BQML", units)

    weight_kg = getattr(dataset, "PatientWeight", None)
    if not weight_kg:
        raise DataIngestionError("PET series has no PatientWeight; cannot compute SUV")

    sequence = getattr(dataset, "RadiopharmaceuticalInformationSequence", None)
    if not sequence:
        raise DataIngestionError(
            "PET series has no RadiopharmaceuticalInformationSequence; "
            "cannot compute SUV")
    info = sequence[0]

    dose_bq = getattr(info, "RadionuclideTotalDose", None)
    half_life = getattr(info, "RadionuclideHalfLife", None)
    start_time = getattr(info, "RadiopharmaceuticalStartTime", None)
    scan_time = (getattr(dataset, "SeriesTime", None)
                 or getattr(dataset, "AcquisitionTime", None))

    missing = [name for name, value in
               (("RadionuclideTotalDose", dose_bq),
                ("RadionuclideHalfLife", half_life),
                ("RadiopharmaceuticalStartTime", start_time),
                ("SeriesTime/AcquisitionTime", scan_time)) if value is None]
    if missing:
        raise DataIngestionError(f"cannot compute SUV; missing tags: {missing}")

    elapsed_s = _hhmmss_to_seconds(scan_time) - _hhmmss_to_seconds(start_time)
    if elapsed_s < 0:                       # scan crossed midnight
        elapsed_s += 24 * 3600
    if elapsed_s > 6 * 3600:
        raise DataIngestionError(
            f"implausible injection-to-scan delay of {elapsed_s / 3600:.1f} h")

    decayed_bq = float(dose_bq) * 2 ** (-elapsed_s / float(half_life))
    if decayed_bq <= 0:
        raise DataIngestionError("decayed dose is not positive")

    factor = float(weight_kg) * 1000.0 / decayed_bq
    log.info("SUV factor %.4e (weight %.0f kg, dose %.0f MBq, %.0f min delay, "
             "%.0f%% remaining)", factor, float(weight_kg), float(dose_bq) / 1e6,
             elapsed_s / 60, 100 * decayed_bq / float(dose_bq))
    return factor