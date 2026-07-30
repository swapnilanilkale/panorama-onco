from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from panorama.core.constants import Modality
from panorama.core.exceptions import DataIngestionError
from panorama.core.logging import get_logger
from panorama.data.schema import Study

log = get_logger(__name__)

COLUMNS = ("patient_id", "study_id", "acquired_on", "modality", "relative_path")
VOLUME_SUFFIXES = (".nii", ".nii.gz")


def _parse_date(text: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise DataIngestionError(
        f"cannot parse study date from {text!r} (want YYYY-MM-DD or YYYYMMDD)")


def _modality_from(filename: str) -> Modality | None:
    key = filename.split(".")[0].upper()
    try:
        return Modality(key)
    except ValueError:
        return None


def scan_directory(root: Path | str) -> list[Study]:
    """Scan root/<patient_id>/<YYYY-MM-DD>/<MODALITY>.nii.gz into Study objects."""
    root = Path(root)
    if not root.is_dir():
        raise DataIngestionError(f"not a directory: {root}")

    studies: list[Study] = []
    skipped = 0
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for study_dir in sorted(d for d in patient_dir.iterdir() if d.is_dir()):
            acquired = _parse_date(study_dir.name)
            volumes: dict[Modality, Path] = {}
            for f in sorted(study_dir.iterdir()):
                if not f.name.endswith(VOLUME_SUFFIXES):
                    continue
                modality = _modality_from(f.name)
                if modality is None:
                    skipped += 1
                    log.warning("unrecognised modality in %s -- skipped", f)
                    continue
                volumes[modality] = f.relative_to(root)
            if not volumes:
                continue
            studies.append(Study(patient_dir.name,
                                 f"{patient_dir.name}_{study_dir.name}",
                                 acquired, volumes))
    log.info("scanned %d studies (%d files skipped)", len(studies), skipped)
    return studies


def write_manifest(studies: list[Study], path: Path | str) -> Path:
    """One row per (study, modality). Paths stay RELATIVE to the data root."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"patient_id": s.patient_id, "study_id": s.study_id,
         "acquired_on": s.acquired_on.isoformat(),
         "modality": m.value, "relative_path": str(p).replace("\\", "/")}
        for s in studies
        for m, p in sorted(s.volumes.items(), key=lambda kv: kv[0].value)
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_manifest(path: Path | str, data_root: Path | str) -> list[Study]:
    """Rebuild Study objects, re-anchoring relative paths to THIS machine's root."""
    path, data_root = Path(path), Path(data_root)
    by_study: dict[str, Study] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            missing = [c for c in COLUMNS if not row.get(c)]
            if missing:
                raise DataIngestionError(f"{path}:{line_no} missing columns {missing}")
            sid = row["study_id"]
            if sid not in by_study:
                by_study[sid] = Study(row["patient_id"], sid,
                                      _parse_date(row["acquired_on"]), {})
            by_study[sid].volumes[Modality(row["modality"])] = \
                data_root / row["relative_path"]
    return [by_study[k] for k in sorted(by_study)]