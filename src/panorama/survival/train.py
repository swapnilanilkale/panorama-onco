"""Full-batch Cox training for the timeline encoder.

Lives in the package rather than a script so both `train_survival.py` and the
ablation study import the same implementation -- an ablation that trains
differently from the main experiment measures the wrong thing.
"""
from __future__ import annotations

import math

import torch

from panorama.core.logging import get_logger
from panorama.survival.cox import concordance_index, cox_partial_likelihood_loss
from panorama.survival.dataset import TimelineCohort
from panorama.survival.timeline import TimelineEncoder
from panorama.utils.reproducibility import seed_everything

log = get_logger(__name__)


def forward(model: TimelineEncoder, cohort: TimelineCohort,
            baseline_only: bool) -> torch.Tensor:
    if baseline_only:
        # Control arm: first study only, elapsed time zeroed. Same architecture
        # and parameter count, no temporal information.
        return model(cohort.embeddings[:, :1],
                     torch.zeros_like(cohort.days[:, :1]),
                     torch.ones_like(cohort.mask[:, :1]))
    return model(cohort.embeddings, cohort.days, cohort.mask)


def train(train_cohort: TimelineCohort, val_cohort: TimelineCohort,
          baseline_only: bool = False, steps: int = 800, lr: float = 1e-3,
          weight_decay: float = 0.01, warmup: int = 50, patience: int = 15,
          seed: int = 1337) -> dict:
    """Train and return the best-by-validation model, with its history."""
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
    history, best, since_best = [], {"c_index": -1.0, "step": -1, "state": None}, 0

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

            # Select on VALIDATION concordance: train C-index reaches 0.99
            # against 0.71 validation, so the final model is not the best one.
            if val_result["c_index"] > best["c_index"]:
                best = {"c_index": val_result["c_index"], "step": step,
                        "comparable_pairs": val_result["comparable_pairs"],
                        "state": {k: v.clone()
                                  for k, v in model.state_dict().items()}}
                since_best = 0
            else:
                since_best += 1
                if since_best >= patience:
                    break

    model.load_state_dict(best["state"])
    return {"model": model, "best": best, "history": history}