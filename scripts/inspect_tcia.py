"""Inspect a TCIA collection's series metadata before downloading anything.

The NBIA REST API exposes modality, series description, image count and file
size per series, so the selection rules can be designed against real metadata
rather than assumptions -- and the download cost is known in advance.

    python scripts/inspect_tcia.py --collection QIN-BREAST --patients 3
"""
from __future__ import annotations

import random
import time

import argparse
import collections

import requests

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get(endpoint: str, attempts: int = 4, **params) -> list[dict]:
    """TCIA drops connections under sustained querying -- retry with backoff."""
    for attempt in range(attempts):
        try:
            response = requests.get(f"{BASE}/{endpoint}", params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, OSError):
            if attempt == attempts - 1:
                raise
            wait = min(30.0, 3.0 * 2 ** attempt) * (0.5 + random.random() * 0.5)
            time.sleep(wait)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="QIN-BREAST")
    parser.add_argument("--patients", type=int, default=3)
    args = parser.parse_args()

    patients = [p["PatientId"] for p in get("getPatient", Collection=args.collection)]
    print(f"{len(patients)} patients in {args.collection}\n")

    total_mb = 0.0
    modality_counts: collections.Counter = collections.Counter()
    description_counts: collections.Counter = collections.Counter()

    for pid in sorted(patients)[:args.patients]:
        series = get("getSeries", Collection=args.collection, PatientID=pid)
        by_study = collections.defaultdict(list)
        for s in series:
            by_study[s["StudyDate"][:10]].append(s)

        print(f"{pid}  --  {len(by_study)} timepoints, {len(series)} series")
        for date in sorted(by_study):
            print(f"  {date}")
            for s in sorted(by_study[date], key=lambda x: x.get("SeriesNumber", 0)):
                mb = s.get("FileSize", 0) / 1e6
                total_mb += mb
                modality = s.get("Modality", "?")
                desc = str(s.get("SeriesDescription", ""))[:36]
                modality_counts[modality] += 1
                description_counts[(modality, desc)] += 1
                print(f"    {modality:3} {desc:38} "
                      f"n={s.get('ImageCount', 0):>4}  {mb:>7.1f} MB")
        print()

    print(f"~{total_mb:.0f} MB for these {args.patients} patients")
    print(f"\nmodalities: {dict(modality_counts)}")
    print("\nseries descriptions by modality:")
    for (modality, desc), n in sorted(description_counts.items()):
        print(f"  {modality:3} {desc:38} x{n}")


if __name__ == "__main__":
    main()