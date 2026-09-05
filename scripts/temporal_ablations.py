"""Do the timeline results depend on TIME, or just on seeing several scans?

Two ablations, both attacking the ADR-0014 claim from opposite directions.

  SHUFFLED CHRONOLOGY -- same studies, permuted order (and their elapsed days
  permuted with them). If disease evolution is being modelled, this must hurt.
  The benchmark can detect it: an order-blind proxy for growth (max/min burden)
  correlates -0.133 with the true growth ratio, so ordering cannot be recovered
  from the set alone.

  ORDINAL POSITIONS -- real elapsed days replaced by visit indices 0,1,2,3.
  If continuous time encoding matters, this must hurt too. Real follow-up is
  irregular: 30 days and 400 days between identical scans imply completely
  different growth rates.

    python scripts/temporal_ablations.py
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
from panorama.survival.cox import concordance_index
from panorama.survival.dataset import TimelineCohort
from panorama.survival.embeddings import load_embeddings
from panorama.survival.synthetic import simulate_outcomes
from panorama.utils.reproducibility import git_revision, seed_everything

log = get_logger(__name__)


def permute_within_timeline(cohort: TimelineCohort,
                            seed: int) -> TimelineCohort:
    """Shuffle each patient's studies, carrying their elapsed days along.

    Days move WITH their embeddings, so the model still sees a valid
    (embedding, day) pairing -- only the sequence order changes. Shuffling
    embeddings against fixed days would instead break the pairing, which tests
    something different and less interesting.
    """
    import copy

    shuffled = copy.deepcopy(cohort)
    generator = torch.Generator().manual_seed(seed)
    for i in range(len(cohort)):
        n = int(cohort.mask[i].sum())
        order = torch.randperm(n, generator=generator)
        shuffled.embeddings[i, :n] = cohort.embeddings[i, order]
        shuffled.days[i, :n] = cohort.days[i, order]
    return shuffled


def ordinal_days(cohort: TimelineCohort) -> TimelineCohort:
    """Replace elapsed days with visit indices 0, 1, 2, ..."""
    import copy

    ordinal = copy.deepcopy(cohort)
    for i in range(len(cohort)):
        n = int(cohort.mask[i].sum())
        ordinal.days[i, :n] = torch.arange(n, dtype=torch.float32)
    return ordinal

from panorama.survival.train import train

def evaluate(train_cohort, val_cohort, label: str, seed: int,
             steps: int, repeats: int) -> dict:
    """Train `repeats` times with different seeds; report mean and spread.

    A single split at 60 val patients has a C-index standard error near 0.05,
    so one number cannot distinguish a 0.05 effect from noise.
    """
    

    scores = []
    for r in range(repeats):
        result = train(train_cohort, val_cohort, baseline_only=False,
                       steps=steps, lr=1e-3, weight_decay=0.01, warmup=50,
                       patience=15, seed=seed + r)
        scores.append(result["best"]["c_index"])
    scores = np.array(scores)
    log.info("%-24s C-index %.4f +/- %.4f  (%s)", label, scores.mean(),
             scores.std(), " ".join(f"{s:.3f}" for s in scores))
    return {"label": label, "mean": float(scores.mean()),
            "std": float(scores.std()), "scores": scores.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path,
                        default=Path("data/synthetic/manifests/embeddings.npz"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/synthetic/manifests/cohort.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic/raw"))
    parser.add_argument("--lesions", type=Path,
                        default=Path("data/synthetic/manifests/lesions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--repeats", type=int, default=5,
                        help="seeds per arm -- one run cannot resolve 0.05")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    seed_everything(args.seed)

    studies = read_manifest(args.manifest, args.data_root)
    lesions = read_lesions(args.lesions)
    embeddings = load_embeddings(args.embeddings)
    outcomes, truth = simulate_outcomes(build_timelines(studies), lesions,
                                        seed=args.seed)
    split = patient_level_split(studies, val_fraction=0.30, test_fraction=0.0,
                                seed=args.seed)
    train_cohort = TimelineCohort(build_timelines(split.train), embeddings, outcomes)
    val_cohort = TimelineCohort(build_timelines(split.val), embeddings, outcomes)

    log.info("running %d seeds per arm", args.repeats)
    arms = [
        evaluate(train_cohort, val_cohort, "correct chronology",
                 args.seed, args.steps, args.repeats),
        # Shuffle TRAIN and VAL identically in spirit: the model must learn
        # from disordered timelines and is tested on disordered timelines.
        evaluate(permute_within_timeline(train_cohort, args.seed),
                 permute_within_timeline(val_cohort, args.seed + 1),
                 "shuffled chronology", args.seed, args.steps, args.repeats),
        evaluate(ordinal_days(train_cohort), ordinal_days(val_cohort),
                 "ordinal positions", args.seed, args.steps, args.repeats),
    ]

    lookup = dict(zip(truth["patient_ids"], truth["true_risk"]))
    oracle = concordance_index([lookup[p] for p in val_cohort.patient_ids],
                               val_cohort.duration.numpy(),
                               val_cohort.event.numpy())

    print(f"\n{'='*66}")
    print(f"  {'arm':26} {'C-index':>10} {'sd':>8}")
    for arm in arms:
        print(f"  {arm['label']:26} {arm['mean']:>10.4f} {arm['std']:>8.4f}")
    print(f"  {'oracle (true risk)':26} {oracle['c_index']:>10.4f}")
    print(f"\n  chronology effect: "
          f"{arms[0]['mean'] - arms[1]['mean']:+.4f}")
    print(f"  elapsed-time effect: "
          f"{arms[0]['mean'] - arms[2]['mean']:+.4f}")
    print(f"\n  {val_cohort.event.sum()} events over {len(val_cohort)} val patients")

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({
        "git_revision": git_revision(), "seed": args.seed,
        "repeats": args.repeats, "arms": arms,
        "oracle_c_index": oracle["c_index"],
    }, indent=2), encoding="utf-8")
    log.info("saved to %s", out_dir)


if __name__ == "__main__":
    main()