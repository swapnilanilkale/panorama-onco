"""Convert downloaded DICOM series to the NIfTI layout `scan_directory` reads.

    data/tcia/qin-breast/<patient>/<date>/CT_2/*.dcm        (input, from download)
    data/tcia/qin-breast-nifti/<patient>/<date>/CT.nii.gz   (output)

Input directories are named <MODALITY>_<SeriesNumber> since M1.22, because a
study can contain several series of one modality and keying on modality alone
made them collide.

Conversion happens ONCE, to disk. The alternative -- parsing 83 DICOM files per
study on every epoch -- would make the DataLoader the bottleneck.

    python scripts/convert_dicom.py
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import nibabel as nib
import numpy as np

from panorama.core.constants import Modality
from panorama.core.exceptions import DataIngestionError
from panorama.core.logging import configure_logging, get_logger
from panorama.data.dicom import read_series

log = get_logger(__name__)

# DICOM modality codes -> our vocabulary. PT is the standard code for PET.
DICOM_TO_MODALITY = {"CT": Modality.CT, "PT": Modality.PET, "MR": Modality.MRI}


def modality_prefix(directory: Path) -> str | None:
    """'CT_2' -> 'CT'. Returns None for directories we do not handle."""
    prefix = directory.name.split("_")[0]
    return prefix if prefix in DICOM_TO_MODALITY else None


def convert_series(series_dir: Path, out_path: Path,
                   allow_irregular: bool = False) -> dict:
    """Read one DICOM series and write it as NIfTI, verifying the geometry."""
    prefix = modality_prefix(series_dir)
    if prefix is None:
        raise DataIngestionError(f"unrecognised modality directory: {series_dir}")
    modality = DICOM_TO_MODALITY[prefix]

    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise DataIngestionError(f"no DICOM files in {series_dir}")

    volume = read_series(files, modality, allow_irregular=allow_irregular)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # nibabel stores (array, affine) with the SAME axis correspondence our
    # affine already uses, so no transpose. A transpose here would silently
    # turn a 700x700x271mm body into 113x700x1674mm.
    nib.save(nib.Nifti1Image(volume.array.astype(np.float32), volume.affine),
             str(out_path))

    # Round-trip check: physical extent must survive the write.
    reloaded = nib.load(str(out_path))
    back_spacing = tuple(float(np.linalg.norm(reloaded.affine[:3, i]))
                         for i in range(3))
    back_extent = tuple(s * n for s, n in zip(back_spacing, reloaded.shape))
    if not np.allclose(back_extent, volume.extent_mm, rtol=1e-4):
        raise DataIngestionError(
            f"geometry lost writing {out_path.name}: "
            f"{tuple(round(e) for e in volume.extent_mm)} mm became "
            f"{tuple(round(e) for e in back_extent)} mm")

    return {"modality": modality.value, "shape": volume.shape,
            "spacing": tuple(round(s, 3) for s in volume.spacing_mm),
            "extent": tuple(round(e) for e in volume.extent_mm),
            "range": (float(volume.array.min()), float(volume.array.max())),
            "mb": out_path.stat().st_size / 1e6}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/tcia/qin-breast"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/tcia/qin-breast-nifti"))
    parser.add_argument("--allow-irregular", action="store_true",
                        help="convert series with irregular slice spacing anyway")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")

    series_dirs = sorted(p for p in args.src.rglob("*")
                         if p.is_dir() and modality_prefix(p)
                         and list(p.glob("*.dcm")))
    log.info("%d series to convert", len(series_dirs))

    converted = skipped = 0
    failures: list[tuple[Path, str]] = []
    collisions: list[tuple[Path, Path]] = []
    stats: dict[str, list] = collections.defaultdict(list)
    claimed: dict[Path, Path] = {}

    for series_dir in series_dirs:
        date_dir = series_dir.parent
        patient_dir = date_dir.parent
        prefix = modality_prefix(series_dir)
        name = "PET" if prefix == "PT" else prefix
        out_path = args.out / patient_dir.name / date_dir.name / f"{name}.nii.gz"

        # Several series of one modality in a study would map to the same file.
        # QIN-BREAST has exactly one per modality; collections like HCC-TACE-Seg
        # do not, so report rather than silently overwrite.
        if out_path in claimed:
            collisions.append((series_dir, claimed[out_path]))
            log.warning("COLLISION %s and %s both map to %s -- keeping the first",
                        series_dir.relative_to(args.src),
                        claimed[out_path].relative_to(args.src), out_path.name)
            continue
        claimed[out_path] = series_dir

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            info = convert_series(series_dir, out_path, args.allow_irregular)
        except DataIngestionError as exc:
            failures.append((series_dir.relative_to(args.src), str(exc)[:90]))
            log.warning("SKIPPED %s: %s", series_dir.relative_to(args.src), exc)
            continue

        converted += 1
        stats[info["modality"]].append(info)
        log.info("%s/%s %s -> %s %s %s mm, range [%.1f, %.1f], %.1f MB",
                 patient_dir.name, date_dir.name, series_dir.name,
                 info["shape"], info["spacing"], info["extent"],
                 *info["range"], info["mb"])

    log.info("converted %d, skipped %d already present, %d failed, %d collisions",
             converted, skipped, len(failures), len(collisions))
    for path, why in failures:
        log.info("  failed: %s -- %s", path, why)

    for modality, rows in sorted(stats.items()):
        ranges = np.array([r["range"] for r in rows])
        log.info("%s: %d volumes, value range across cohort [%.2f, %.2f], %.0f MB total",
                 modality, len(rows), ranges[:, 0].min(), ranges[:, 1].max(),
                 sum(r["mb"] for r in rows))


if __name__ == "__main__":
    main()