"""Pair every study's imaging with a rendered radiology report.

The corpus is the supervision signal for Aim 2: for each study we hold the
STRUCTURED facts (for evaluation) and the rendered PROSE (for training).
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from panorama.clinical.recist import Lesion, TimepointAssessment, assess_course
from panorama.clinical.report import StructuredReport, build_report, render
from panorama.core.logging import get_logger
from panorama.data.schema import Study
from panorama.data.splits import build_timelines

log = get_logger(__name__)


def build_corpus(studies: Sequence[Study],
                 lesions_by_study: dict[str, list[Lesion]],
                 seed: int = 1337) -> dict[str, tuple[StructuredReport, str]]:
    """study_id -> (structured facts, rendered prose)."""
    corpus: dict[str, tuple[StructuredReport, str]] = {}
    skipped = 0

    for timeline in build_timelines(studies):
        known = [s for s in timeline.studies if s.study_id in lesions_by_study]
        if not known:
            skipped += len(timeline.studies)
            continue

        course = assess_course([TimepointAssessment(s.study_id,
                                                    lesions_by_study[s.study_id])
                                for s in known])
        baseline = course[0].sld_mm
        for i, (study, tp) in enumerate(zip(known, course)):
            report = build_report(
                patient_id=timeline.patient_id, tp=tp,
                acquired_on=study.acquired_on,
                modalities=list(study.present),
                baseline_sld=baseline,
                nadir_sld=min(t.sld_mm for t in course[:i + 1]),
                prior=course[i - 1] if i else None)
            # Per-study seed: reproducible, yet phrasing varies across studies.
            rng = random.Random(f"{seed}:{study.study_id}")
            corpus[study.study_id] = (report, render(report, rng))

    log.info("built %d reports (%d studies had no lesion data)", len(corpus), skipped)
    return corpus


def write_corpus(corpus: dict[str, tuple[StructuredReport, str]],
                 path: Path | str) -> Path:
    """JSONL: one record per study. Text keeps its newlines untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for study_id in sorted(corpus):
            report, text = corpus[study_id]
            fh.write(json.dumps({
                "study_id": study_id,
                "patient_id": report.patient_id,
                "acquired_on": report.acquired_on.isoformat(),
                "modalities": [m.value for m in report.modalities],
                "response": report.response.value,
                "sld_mm": report.sld_mm,
                "baseline_sld_mm": report.baseline_sld_mm,
                "nadir_sld_mm": report.nadir_sld_mm,
                "prior_study_id": report.prior_study_id,
                "n_lesions": len(report.lesions),
                "report": text,
            }, ensure_ascii=False) + "\n")
    log.info("corpus written: %s (%d reports)", path, len(corpus))
    return path


def read_corpus(path: Path | str) -> dict[str, dict]:
    """study_id -> record, for the dataset to consume."""
    out: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                out[record["study_id"]] = record
    return out