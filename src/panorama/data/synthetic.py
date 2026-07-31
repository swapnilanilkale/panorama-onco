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
                 max_studies: int = 3, seed: int = 0) -> Path:
    """Write root/<patient>/<date>/<MODALITY>.nii.gz for a whole cohort."""
    import nibabel as nib

    root = Path(root)
    rng = np.random.default_rng(seed)
    n_studies = 0

    for p in range(n_patients):
        patient = f"PAT{p:04d}"
        modalities = PATTERNS[p % len(PATTERNS)]
        baseline = date(2024, 1, 15)
        # A patient's lesions persist across timepoints, drifting and growing.
        n_lesions = int(rng.integers(1, 4))
        centres = rng.uniform(20.0, 76.0, (n_lesions, 3))
        radii = rng.uniform(5.0, 12.0, n_lesions)

        for k in range(int(rng.integers(1, max_studies + 1))):
            acquired = baseline + timedelta(days=90 * k)
            drift = centres + rng.normal(0.0, 1.5, centres.shape)
            grown = radii * (1.0 + 0.15 * k)
            out_dir = root / patient / acquired.isoformat()
            out_dir.mkdir(parents=True, exist_ok=True)
            for modality in modalities:
                vol, affine = synth_volume(modality, drift, grown, rng)
                nib.save(nib.Nifti1Image(vol, affine),
                         out_dir / f"{modality.value}.nii.gz")
            n_studies += 1

    log.info("wrote %d studies for %d patients to %s", n_studies, n_patients, root)
    return root