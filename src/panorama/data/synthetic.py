"""Synthetic oncology cohort generator.

Lets the whole pipeline be exercised end to end without TCIA access, and gives
a controlled testbed where the cross-modal correspondence is KNOWN: lesions are
dense on CT and hypermetabolic on PET at identical world coordinates, on
deliberately different voxel grids.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from panorama.core.constants import Modality
from panorama.core.logging import get_logger

log = get_logger(__name__)

# Native geometry per modality -- deliberately mismatched, as in real acquisitions.
GEOMETRY = {
    Modality.CT: ((96, 96, 96), (1.0, 1.0, 1.0)),
    Modality.MRI: ((64, 64, 48), (1.5, 1.5, 2.0)),
    Modality.PET: ((24, 24, 24), (4.0, 4.0, 4.0)),
}
PATTERNS = ([Modality.CT, Modality.PET],                       # PET/CT staging
            [Modality.CT],                                     # CT-only follow-up
            [Modality.CT, Modality.MRI, Modality.PET])         # full tri-modal


# Multiplicative lesion-size factors per timepoint, by clinical trajectory.
# Chosen so the cohort contains all RECIST categories -- including a rebound
# course whose progression only the nadir rule detects.
TRAJECTORIES = {
    "responder":  (1.00, 0.62, 0.55, 0.60),
    "stable":     (1.00, 1.05, 0.97, 1.08),
    "progressor": (1.00, 1.12, 1.35, 1.70),
    "rebound":    (1.00, 0.58, 0.66, 0.95),
}
TRAJECTORY_NAMES = tuple(TRAJECTORIES)

# Anatomical region by z position (mm). Bins must span the range that
# `write_cohort` actually samples centres from, or some regions never occur.
REGIONS = ((0.0, 35.0, "right lower lobe"),
           (35.0, 50.0, "left hilum"),
           (50.0, 65.0, "hepatic segment VII"),
           (65.0, 1e9, "retroperitoneal node"))


def _region_for(centre_mm) -> str:
    z = float(centre_mm[2])
    for lo, hi, name in REGIONS:
        if lo <= z < hi:
            return name
    return "unspecified"


def _distance_field(shape, spacing, centre_mm) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)],
                        indexing="ij")
    return sum((g - c) ** 2 for g, c in zip(grids, centre_mm)) ** 0.5


def synth_volume(modality: Modality, lesions_mm, radii_mm,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One volume with lesions at the given PHYSICAL positions."""
    shape, spacing = GEOMETRY[modality]
    affine = np.diag([*spacing, 1.0])

    if modality is Modality.PET:
        # SUV-like: smooth hot spots on a low background.
        vol = rng.gamma(1.5, 0.8, shape).astype(np.float32)
        for centre, radius in zip(lesions_mm, radii_mm):
            d = _distance_field(shape, spacing, centre)
            vol += (8.0 * np.exp(-(d ** 2) / (2 * (radius * 1.2) ** 2))).astype(np.float32)
        return vol, affine

    if modality is Modality.CT:
        vol = rng.normal(-50.0, 30.0, shape).astype(np.float32)   # HU-like
        bright = (120.0, 15.0)
    else:                                                          # MRI
        vol = rng.normal(300.0, 60.0, shape).astype(np.float32)    # arbitrary units
        bright = (700.0, 80.0)

    for centre, radius in zip(lesions_mm, radii_mm):
        mask = _distance_field(shape, spacing, centre) <= radius
        vol[mask] = rng.normal(*bright, int(mask.sum())).astype(np.float32)
    return vol, affine



def write_cohort(root: Path | str, n_patients: int = 20,
                 max_studies: int = 3, seed: int = 0,
                 lesion_manifest: Path | str | None = None) -> Path:
    """Write volumes AND the lesion ground truth that describes them."""
    import csv

    import nibabel as nib

    root = Path(root)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    n_studies = 0

    for p in range(n_patients):
        patient = f"PAT{p:04d}"
        modalities = PATTERNS[p % len(PATTERNS)]
        trajectory = TRAJECTORY_NAMES[p % len(TRAJECTORY_NAMES)]
        factors = TRAJECTORIES[trajectory]
        baseline = date(2024, 1, 15)


        n_lesions = int(rng.integers(2, 4))              
        centres = rng.uniform(25.0, 71.0, (n_lesions, 3))
        radii = rng.uniform(8.0, 16.0, n_lesions)        

        for k in range(int(rng.integers(3, max_studies + 1))):
            acquired = baseline + timedelta(days=90 * k)
            drift = centres + rng.normal(0.0, 1.5, centres.shape)
            scaled = radii * factors[min(k, len(factors) - 1)]

            out_dir = root / patient / acquired.isoformat()
            out_dir.mkdir(parents=True, exist_ok=True)
            for modality in modalities:
                vol, affine = synth_volume(modality, drift, scaled, rng)
                nib.save(nib.Nifti1Image(vol, affine),
                         out_dir / f"{modality.value}.nii.gz")

            study_id = f"{patient}_{acquired.isoformat()}"
            for i, (centre, radius) in enumerate(zip(drift, scaled)):
                rows.append({
                    "patient_id": patient, "study_id": study_id,
                    "trajectory": trajectory, "lesion_id": f"L{i + 1}",
                    "organ": _region_for(centre),
                    "longest_diameter_mm": round(float(2.0 * radius), 1),
                    "centre_x_mm": round(float(centre[0]), 1),
                    "centre_y_mm": round(float(centre[1]), 1),
                    "centre_z_mm": round(float(centre[2]), 1),
                })
            n_studies += 1

    log.info("wrote %d studies for %d patients to %s", n_studies, n_patients, root)

    if lesion_manifest:
        path = Path(lesion_manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        log.info("lesion ground truth: %s (%d lesions)", path, len(rows))

    return root


def read_lesions(path: Path | str) -> dict[str, list]:
    """study_id -> [Lesion], the ground truth the reports describe."""
    import csv

    from panorama.clinical.recist import Lesion

    out: dict[str, list] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["study_id"], []).append(
                Lesion(row["lesion_id"], float(row["longest_diameter_mm"]),
                       row["organ"]))
    return out

