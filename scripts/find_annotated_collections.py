"""Find TCIA collections with expert segmentations and longitudinal follow-up.

Aim 2's generation work is on synthetic imaging because QIN-BREAST has no lesion
annotations (ADR-0010). SEG and RTSTRUCT objects carry expert contours from which
real diameters can be measured.

Note `getModalityValues` ignores a Modality filter -- it returns the archive's
modality NAMES. Filtered by Collection, it returns that collection's modalities,
which is the question we actually want to ask.

    python scripts/find_annotated_collections.py
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import requests

from panorama.core.logging import configure_logging, get_logger

log = get_logger(__name__)
BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
ANNOTATION = {"SEG", "RTSTRUCT"}
IMAGING = {"CT", "PT", "MR"}


def get(endpoint: str, attempts: int = 4, **params):
    """TCIA drops connections under sustained querying -- retry with backoff."""
    for attempt in range(attempts):
        try:
            response = requests.get(f"{BASE}/{endpoint}", params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, OSError):
            if attempt == attempts - 1:
                raise
            wait = min(30.0, 3.0 * 2 ** attempt) * (0.5 + random.random() * 0.5)
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-patients", type=int, default=5)
    parser.add_argument("--cache", type=Path,
                        default=Path("data/tcia/collection_survey.json"),
                        help="resume point -- the survey takes many API calls")
    args = parser.parse_args()

    configure_logging("INFO")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    surveyed: dict[str, list[str]] = (
        json.loads(args.cache.read_text()) if args.cache.is_file() else {})

    names = sorted(c["Collection"] for c in get("getCollectionValues"))
    log.info("%d collections in the archive (%d already surveyed)",
             len(names), len(surveyed))

    for i, name in enumerate(names, start=1):
        if name in surveyed:
            continue
        try:
            surveyed[name] = sorted(
                m["Modality"] for m in get("getModalityValues", Collection=name))
        except Exception as exc:
            log.warning("%s: %s -- skipped", name, type(exc).__name__)
            surveyed[name] = []
        if i % 20 == 0:
            args.cache.write_text(json.dumps(surveyed))
            log.info("surveyed %d/%d", i, len(names))
    args.cache.write_text(json.dumps(surveyed))

    annotated = {n: m for n, m in surveyed.items()
                 if set(m) & ANNOTATION and set(m) & IMAGING}
    print(f"\n{len(annotated)} collections have BOTH annotations and imaging\n")
    print(f"{'collection':46} {'annotation':14} {'imaging':14} {'multi-tp':>10}")
    print("-" * 88)

    ranked = []
    for name, modalities in sorted(annotated.items()):
        annot = "+".join(sorted(set(modalities) & ANNOTATION))
        images = "+".join(sorted(set(modalities) & IMAGING))
        try:
            patients = [p["PatientId"] for p in get("getPatient", Collection=name)]
        except Exception:
            continue

        multi = probed = 0
        for pid in sorted(patients)[:args.probe_patients]:
            try:
                series = get("getSeries", Collection=name, PatientID=pid)
            except Exception:
                continue
            probed += 1
            dates = {s["StudyDate"][:10] for s in series
                     if s.get("Modality") in IMAGING}
            multi += len(dates) >= 2

        fraction = multi / probed if probed else 0.0
        print(f"{name[:46]:46} {annot:14} {images:14} {multi:>4}/{probed:<5}")
        ranked.append((fraction, len(patients), name, annot, images))

    print("\n=== best candidates for Aim 2 route (c) ===")
    for fraction, n_pat, name, annot, images in sorted(ranked, reverse=True)[:10]:
        if fraction > 0:
            print(f"  {name[:46]:46} {annot:12} {images:12} "
                  f"{n_pat:>4} patients, {fraction:.0%} longitudinal")


if __name__ == "__main__":
    main()