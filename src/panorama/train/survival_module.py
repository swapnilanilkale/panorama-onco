"""Aim 3: predict survival risk from a patient's imaging timeline."""
from __future__ import annotations

import math

import lightning as L
import torch

from panorama.core.logging import get_logger
from panorama.survival.cox import concordance_index, cox_partial_likelihood_loss
from panorama.survival.dataset import TimelineCohort
from panorama.survival.timeline import TimelineEncoder

log = get_logger(__name__)


class SurvivalModule(L.LightningModule):
    """Full-batch Cox training on cached study embeddings."""

    def __init__(self, train_cohort: TimelineCohort, val_cohort: TimelineCohort,
                 hidden_dim: int = 64, depth: int = 2, num_heads: int = 4,
                 time_dim: int = 32, dropout: float = 0.1,
                 baseline_only: bool = False,
                 lr: float = 1.0e-3, weight_decay: float = 0.01,
                 max_steps: int = 500, warmup_steps: int = 50) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["train_cohort", "val_cohort"])
        self.train_cohort = train_cohort
        self.val_cohort = val_cohort
        self.encoder = TimelineEncoder(
            embed_dim=train_cohort.embed_dim, hidden_dim=hidden_dim,
            depth=depth, num_heads=num_heads, time_dim=time_dim, dropout=dropout)

    def risk(self, cohort: TimelineCohort) -> torch.Tensor:
        if self.hparams.baseline_only:
            # Control arm: only the FIRST study, no temporal information.
            # A timeline model that fails to beat this has learned nothing
            # a single scan could not supply.
            first = cohort.embeddings[:, :1]
            days = torch.zeros_like(cohort.days[:, :1])
            mask = torch.ones_like(cohort.mask[:, :1])
            return self.encoder(first, days, mask)
        return self.encoder(cohort.embeddings, cohort.days, cohort.mask)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        cohort = self.train_cohort
        loss = cox_partial_likelihood_loss(self.risk(cohort),
                                           cohort.duration, cohort.event)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(cohort))
        with torch.no_grad():
            result = concordance_index(self.risk(cohort).cpu().numpy(),
                                       cohort.duration.cpu().numpy(),
                                       cohort.event.cpu().numpy())
        self.log("train/c_index", result["c_index"], prog_bar=True,
                 batch_size=len(cohort))
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        cohort = self.val_cohort
        with torch.no_grad():
            risk = self.risk(cohort)
        loss = cox_partial_likelihood_loss(risk, cohort.duration, cohort.event)
        result = concordance_index(risk.cpu().numpy(),
                                   cohort.duration.cpu().numpy(),
                                   cohort.event.cpu().numpy())
        self.log("val/loss", loss, prog_bar=True, batch_size=len(cohort))
        self.log("val/c_index", result["c_index"], prog_bar=True,
                 batch_size=len(cohort))
        # A C-index is uninterpretable without knowing how many pairs it rests
        # on: 0.75 over 143 pairs means far less than the same over 8,000.
        self.log("val/comparable_pairs", float(result["comparable_pairs"]),
                 batch_size=len(cohort))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        warmup, total = self.hparams.warmup_steps, self.hparams.max_steps

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, factor), "interval": "step"}}