"""Confidence intervals for the headline results in docs/RESULTS.md section 3.

Seven results were single-run point estimates. A methods paper arguing for
rigour cannot report those, and several null claims ("pretrained ties random")
were assertions rather than measurements until now.

    python scripts/bootstrap_results.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from panorama.core.logging import configure_logging, get_logger
from panorama.data.manifest import read_manifest
from panorama.data.splits import build_timelines, patient_level_split
from panorama.data.synthetic import read_lesions
from panorama.eval.bootstrap import format_ci, paired_bootstrap_ci
from panorama.survival.cox import concordance_index
from panorama.survival.dataset import TimelineCohort
from panorama.survival.embeddings import load_embeddings
from panorama.survival.synthetic import simulate_outcomes
from panorama.survival.train import forward, train
from panorama.utils.reproducibility import git_revision, seed_everything

log = get_logger(__name__)


def aim3_timeline_vs_baseline(seed: int) -> dict:
    """Aim 3: does the timeline beat a single scan? (ADR-0014)"""
    studies = read_manifest("data/synthetic/manifests/cohort.csv",
                            "data/synthetic/raw")
    lesions = read_lesions("data/synthetic/manifests/lesions.csv")
    embeddings = load_embeddings("data/synthetic/manifests/embeddings.npz")
    outcomes, _ = simulate_outcomes(build_timelines(studies), lesions, seed=seed)
    split = patient_level_split(studies, val_fraction=0.30, test_fraction=0.0,
                                seed=seed)
    train_cohort = TimelineCohort(build_timelines(split.train), embeddings, outcomes)
    val = TimelineCohort(build_timelines(split.val), embeddings, outcomes)

    models = {}
    for arm in (False, True):
        result = train(train_cohort, val, baseline_only=arm, seed=seed)
        models[arm] = result["model"].eval()

    duration = val.duration.numpy()
    event = val.event.numpy()
    with torch.no_grad():
        risk = {arm: forward(m, val, arm).numpy() for arm, m in models.items()}

    def scorer(arm):
        def metric(index: np.ndarray) -> float:
            # Resample PATIENTS. Resampling the 1,201 comparable pairs instead
            # would treat correlated pairs as independent and give an interval
            # ~4.5x too narrow.
            result = concordance_index(risk[arm][index], duration[index],
                                       event[index])
            return result["c_index"]
        return metric

    return paired_bootstrap_ci(scorer(False), scorer(True), len(val), seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/bootstrap"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    seed_everything(args.seed)

    results = {}
    log.info("Aim 3: timeline vs single-scan control")
    results["aim3_timeline_vs_baseline"] = aim3_timeline_vs_baseline(args.seed)

    print(f"\n{'='*72}")
    print("  bootstrap confidence intervals (patient-level resampling)\n")
    for name, result in results.items():
        print(f"  {name}")
        print(f"    {format_ci(result)}\n")

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps({"git_revision": git_revision(), "seed": args.seed,
                    "results": results}, indent=2), encoding="utf-8")
    log.info("saved to %s", out_dir)


if __name__ == "__main__":
    main()