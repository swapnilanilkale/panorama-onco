"""Aim 3 end-to-end: timeline encoder trained with the Cox partial likelihood.

No Lightning. The Cox loss is full-batch by necessity (mini-batching shrinks the
risk sets and biases the estimate), the cohort is 1.4 MB of cached embeddings,
and training takes seconds -- so the framework contributes a dummy dataloader,
an artificial epoch concept, and state-management bugs, with nothing in return.

    python scripts/train_survival.py
    python scripts/train_survival.py --baseline-only    # control arm
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from panorama.core.logging import configure_logging, get_logger
from panorama.data.manifest import read_manifest
from panorama.data.splits import build_timelines, patient_level_split
from panorama.data.synthetic import read_lesions
from panorama.survival.cox import concordance_index, cox_partial_likelihood_loss
from panorama.survival.dataset import TimelineCohort
from panorama.survival.embeddings import load_embeddings
from panorama.survival.synthetic import simulate_outcomes
from panorama.survival.timeline import TimelineEncoder
from panorama.utils.reproducibility import git_revision, seed_everything

log = get_logger(__name__)


def forward(model: TimelineEncoder, cohort: TimelineCohort,
            baseline_only: bool) -> torch.Tensor:
    if baseline_only:
        # Control arm: the first study only, elapsed time zeroed. Same
        # architecture and parameter count, no temporal information.
        return model(cohort.embeddings[:, :1],
                     torch.zeros_like(cohort.days[:, :1]),
                     torch.ones_like(cohort.mask[:, :1]))
    return model(cohort.embeddings, cohort.days, cohort.mask)


def train(train_cohort: TimelineCohort, val_cohort: TimelineCohort,
          baseline_only: bool, steps: int, lr: float, weight_decay: float,
          warmup: int, patience: int, seed: int) -> dict:
    seed_everything(seed)
    model = TimelineEncoder(embed_dim=train_cohort.embed_dim, hidden_dim=64,
                            depth=2, num_heads=4, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    history, best = [], {"c_index": -1.0, "step": -1, "state": None}
    since_best = 0

    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        loss = cox_partial_likelihood_loss(
            forward(model, train_cohort, baseline_only),
            train_cohort.duration, train_cohort.event)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                train_c = concordance_index(
                    forward(model, train_cohort, baseline_only).numpy(),
                    train_cohort.duration.numpy(), train_cohort.event.numpy())
                val_result = concordance_index(
                    forward(model, val_cohort, baseline_only).numpy(),
                    val_cohort.duration.numpy(), val_cohort.event.numpy())
            history.append({"step": step, "loss": float(loss.detach()),
                            "train_c": train_c["c_index"],
                            "val_c": val_result["c_index"]})

            # Select on VALIDATION concordance. Training C-index reaches 0.99
            # while validation peaks near 0.64 -- 140 patients and ~77 events
            # cannot support 108K parameters without heavy overfitting.
            if val_result["c_index"] > best["c_index"]:
                best = {"c_index": val_result["c_index"], "step": step,
                        "state": {k: v.clone() for k, v in model.state_dict().items()},
                        "comparable_pairs": val_result["comparable_pairs"]}
                since_best = 0
            else:
                since_best += 1
                if since_best >= patience:
                    log.info("early stop at step %d (no improvement for %d checks)",
                             step, patience)
                    break

    model.load_state_dict(best["state"])
    return {"model": model, "best": best, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path,
                        default=Path("data/synthetic/manifests/embeddings.npz"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/synthetic/manifests/cohort.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic/raw"))
    parser.add_argument("--lesions", type=Path,
                        default=Path("data/synthetic/manifests/lesions.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/survival"))
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.30,
                        help="30%% gives ~60 patients; below that the C-index "
                             "cannot resolve a 0.05 difference (ADR-0013)")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    seed_everything(args.seed)

    studies = read_manifest(args.manifest, args.data_root)
    lesions = read_lesions(args.lesions)
    embeddings = load_embeddings(args.embeddings)
    outcomes, truth = simulate_outcomes(build_timelines(studies), lesions,
                                        seed=args.seed)
    split = patient_level_split(studies, val_fraction=args.val_fraction,
                                test_fraction=0.0, seed=args.seed)
    train_cohort = TimelineCohort(build_timelines(split.train), embeddings, outcomes)
    val_cohort = TimelineCohort(build_timelines(split.val), embeddings, outcomes)

    result = train(train_cohort, val_cohort, args.baseline_only, args.steps,
                   args.lr, args.weight_decay, args.warmup, args.patience,
                   args.seed)

    # The oracle on THIS split: the ceiling any model could reach given the
    # simulated hazard. No real cohort permits this comparison.
    lookup = dict(zip(truth["patient_ids"], truth["true_risk"]))
    oracle = concordance_index(
        [lookup[p] for p in val_cohort.patient_ids],
        val_cohort.duration.numpy(), val_cohort.event.numpy())

    arm = "baseline only (control)" if args.baseline_only else "full timeline"
    best = result["best"]
    print(f"\n{'='*60}")
    print(f"  arm              {arm}")
    print(f"  val patients     {len(val_cohort)}  ({int(val_cohort.event.sum())} events)")
    print(f"  best val C-index {best['c_index']:.4f} at step {best['step']} "
          f"({best['comparable_pairs']:,} pairs)")
    print(f"  oracle on split  {oracle['c_index']:.4f}")
    print(f"\n  ADR-0013: achievable ceiling 0.659, burden-only 0.612")

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps({
        "arm": arm, "git_revision": git_revision(), "seed": args.seed,
        "val_patients": len(val_cohort),
        "val_events": int(val_cohort.event.sum()),
        "best_val_c_index": best["c_index"], "best_step": best["step"],
        "oracle_c_index": oracle["c_index"],
        "history": result["history"],
    }, indent=2), encoding="utf-8")
    torch.save(result["model"].state_dict(), out_dir / "model.pt")
    log.info("saved to %s", out_dir)


if __name__ == "__main__":
    main()