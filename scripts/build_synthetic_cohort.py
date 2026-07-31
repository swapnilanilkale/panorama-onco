"""Generate a synthetic cohort and its manifest, for end-to-end smoke runs.

    python scripts/build_synthetic_cohort.py --patients 24
"""
from __future__ import annotations

import argparse
from pathlib import Path

from panorama.core.logging import configure_logging, get_logger
from panorama.data.manifest import scan_directory, write_manifest
from panorama.data.synthetic import write_cohort

log = get_logger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/synthetic/raw"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/synthetic/manifests/cohort.csv"))
    parser.add_argument("--lesions", type=Path,
                        default=Path("data/synthetic/manifests/lesions.csv"))
    parser.add_argument("--patients", type=int, default=24)
    parser.add_argument("--max-studies", type=int, default=4) 
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_logging("INFO")
    write_cohort(args.root, n_patients=args.patients,
                 max_studies=args.max_studies, seed=args.seed,
                 lesion_manifest=args.lesions)
    studies = scan_directory(args.root)
    path = write_manifest(studies, args.manifest)
    log.info("manifest: %s (%d studies, %d patients)",
             path, len(studies), len({s.patient_id for s in studies}))


if __name__ == "__main__":
    main()