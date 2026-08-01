"""Render a radiology report for every study in a cohort.

    python scripts/build_report_corpus.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from panorama.clinical.corpus import build_corpus, write_corpus
from panorama.core.logging import configure_logging, get_logger
from panorama.data.manifest import read_manifest
from panorama.data.synthetic import read_lesions

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/synthetic/manifests/cohort.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic/raw"))
    parser.add_argument("--lesions", type=Path,
                        default=Path("data/synthetic/manifests/lesions.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/synthetic/manifests/reports.jsonl"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    studies = read_manifest(args.manifest, args.data_root)
    lesions = read_lesions(args.lesions)
    corpus = build_corpus(studies, lesions, seed=args.seed)
    write_corpus(corpus, args.out)


if __name__ == "__main__":
    main()