"""Download a TCIA collection's DICOM series via the public NBIA REST API.

No account or desktop installer required -- the public endpoints are
unauthenticated. The download is RESUMABLE: a series is skipped only when a
completion marker proves it was fully extracted, so an interrupted run picks up
where it stopped rather than restarting or silently keeping a truncated series.

    python scripts/download_tcia.py --collection QIN-BREAST --patients 10
"""
from __future__ import annotations

import argparse
import collections
import io
import random
import time
import zipfile
from pathlib import Path

import requests

from panorama.core.logging import configure_logging, get_logger

log = get_logger(__name__)

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
MARKER = ".complete"          # written LAST, so its presence proves wholeness
MAX_ATTEMPTS = 5


def request_json(endpoint: str, **params) -> list[dict]:
    return _with_retry(lambda: requests.get(f"{BASE}/{endpoint}",
                                            params=params, timeout=60)).json()


def _with_retry(call, what: str = "request"):
    """Exponential backoff with jitter. TCIA drops connections under load."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = call()
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            wait = min(60.0, 2.0 * (2 ** attempt)) * (0.5 + random.random() * 0.5)
            log.warning("%s failed (%s); retrying in %.1fs [%d/%d]",
                        what, type(exc).__name__, wait, attempt + 1, MAX_ATTEMPTS)
            time.sleep(wait)


def download_series(series_uid: str, out_dir: Path) -> bool:
    """Fetch one series as a ZIP and extract it. Returns False if skipped."""
    if (out_dir / MARKER).exists():
        return False
    if out_dir.exists():
        log.info("re-downloading %s (no completion marker -- prior run was cut short)",
                 out_dir.name)

    response = _with_retry(
        lambda: requests.get(f"{BASE}/getImage",
                             params={"SeriesInstanceUID": series_uid},
                             timeout=600, stream=False),
        what=f"getImage {series_uid[-12:]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(out_dir)

    n_files = len(list(out_dir.glob("*.dcm")))
    (out_dir / MARKER).write_text(f"{series_uid}\n{n_files} files\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="QIN-BREAST")
    parser.add_argument("--out", type=Path, default=Path("data/tcia"))
    parser.add_argument("--patients", type=int, default=10,
                        help="how many eligible patients to download")
    parser.add_argument("--modalities", nargs="+", default=["CT", "PT"],
                        help="DICOM modality codes (PT = PET)")
    parser.add_argument("--min-timepoints", type=int, default=2,
                        help="skip patients with fewer distinct study dates")
    args = parser.parse_args()

    configure_logging("INFO")
    root = args.out / args.collection.lower()
    wanted = set(args.modalities)

    patients = sorted(p["PatientId"] for p in
                      request_json("getPatient", Collection=args.collection))
    log.info("%s: %d patients", args.collection, len(patients))

    # Select eligible patients from METADATA first -- no image bytes yet.
    plan: list[tuple[str, list[dict]]] = []
    for pid in patients:
        series = [s for s in request_json("getSeries", Collection=args.collection,
                                          PatientID=pid)
                  if s.get("Modality") in wanted]
        dates = {s["StudyDate"][:10] for s in series}
        if len(dates) >= args.min_timepoints:
            plan.append((pid, series))
        if len(plan) >= args.patients:
            break

    total_mb = sum(s.get("FileSize", 0) for _, ss in plan for s in ss) / 1e6
    n_series = sum(len(ss) for _, ss in plan)
    log.info("plan: %d patients, %d series, %.1f GB", len(plan), n_series, total_mb / 1000)

    downloaded = skipped = 0
    for i, (pid, series) in enumerate(plan, start=1):

        for s in sorted(series, key=lambda x: (x["StudyDate"], x.get("SeriesNumber", 0))):
            date = s["StudyDate"][:10]
            # Layout: <patient>/<date>/<MODALITY>_<SeriesNumber>/
            # A study can contain SEVERAL series of the same modality -- HCC-TACE-Seg
            # has both "PRE LIVER" and "Recon 2: LIVER 3 PHASE" as CT. Keying the
            # directory on modality alone makes them collide, and the completion
            # marker then hides the loss.
            number = s.get("SeriesNumber", 0)
            out_dir = root / pid / date / f"{s['Modality']}_{number}"
            if download_series(s["SeriesInstanceUID"], out_dir):
                downloaded += 1
                log.info("[%d/%d] %s %s %s (%d images)", i, len(plan), pid, date,
                         s["Modality"], s.get("ImageCount", 0))
            else:
                skipped += 1

    log.info("done: %d series downloaded, %d already present", downloaded, skipped)
    log.info("output: %s", root)


if __name__ == "__main__":
    main()