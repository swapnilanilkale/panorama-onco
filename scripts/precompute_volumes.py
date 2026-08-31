"""Preprocess volumes once to .npy, so training reads instead of recomputing.

Resampling a 512x512x83 CT to a 350x350x136 grid takes ~2 seconds. Doing it
inside __getitem__ means paying that on every batch of every epoch, which
leaves the GPU idle: on a T4 this pinned throughput at 0.05 it/s -- slower
than CPU-only training.

    python scripts/precompute_volumes.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panorama.core.constants import Modality
from panorama.core.logging import configure_logging, get_logger
from panorama.data.manifest import read_manifest
from panorama.data.pipeline import preprocess
from panorama.data.volume import load_nifti

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/tcia/manifests/qin-breast.csv"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/tcia/qin-breast-nifti"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/tcia/qin-breast-preprocessed"))
    parser.add_argument("--spacing", type=float, nargs=3, default=(2.0, 2.0, 2.0))
    args = parser.parse_args()

    configure_logging("INFO")
    studies = read_manifest(args.manifest, args.data_root)
    log.info("preprocessing %d studies to %s", len(studies), args.out)

    index = {}
    for i, study in enumerate(studies, start=1):
        for modality, path in study.volumes.items():
            out_dir = args.out / study.patient_id / study.acquired_on.isoformat()
            out_dir.mkdir(parents=True, exist_ok=True)
            array_path = out_dir / f"{modality.value}.npy"
            affine_path = out_dir / f"{modality.value}_affine.npy"
            if array_path.exists():
                continue
            volume = preprocess(load_nifti(path, modality),
                                target_spacing=tuple(args.spacing))
            # float16 halves the storage. Normalised values live in [0,1] (CT,
            # PET) or are z-scored (MRI), so ~3 decimal digits of precision is
            # far more than the data carries.
            np.save(array_path, volume.array.astype(np.float16))
            np.save(affine_path, volume.affine)
    
            index[f"{study.study_id}/{modality.value}"] = str(
                array_path.relative_to(args.out)).replace("\\", "/")
        if i % 10 == 0:
            log.info("  %d/%d studies", i, len(studies))

    (args.out / "index.json").write_text(json.dumps(index), encoding="utf-8")
    total = sum(p.stat().st_size for p in args.out.rglob("*.npy")) / 1e9
    log.info("done: %d arrays, %.1f GB", len(list(args.out.rglob('*.npy'))), total)


if __name__ == "__main__":
    main()