"""Aim 2: predict a structured report from current + prior scans.

The vision encoder is fine-tuned end to end by default. ADR-0009 established
that MAE pretraining on this scale learned nothing generalisable (validation
variance explained 0.005), so there is no pretrained representation worth
freezing -- and report supervision is far stronger signal than masked
reconstruction.

`freeze_vision=True` gives the control arm: a RANDOM frozen encoder with a
trainable head. If that scores the same, the encoder is not contributing and the
head is reading something trivial.
"""
from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch

from panorama.clinical.recist import classify
from panorama.core.constants import RECISTResponse
from panorama.core.logging import get_logger
from panorama.train.mae_module import NO_DECAY_NAMES, MAEPretrainModule
from panorama.vision.encoder import MultiStreamViT
from panorama.vlm.report_head import StructuredReportHead, report_loss

log = get_logger(__name__)

RESPONSES = list(RECISTResponse)


class ReportModule(L.LightningModule):
    """Study pair -> structured report fields, with RECIST derived at eval."""

    def __init__(self,
                 n_organs: int,
                 pretrained_checkpoint: str | None = None,
                 freeze_vision: bool = False,
                 volume_shape: tuple[int, int, int] = (32, 32, 32),
                 patch_size: int = 8,
                 embed_dim: int = 128,
                 depth: int = 4,
                 num_heads: int = 8,
                 fusion_every: int = 2,
                 max_lesions: int = 4,
                 head_hidden: int = 256,
                 dropout: float = 0.1,
                 base_lr: float = 1.0e-3,
                 effective_batch_size: int = 256,
                 vision_lr_multiplier: float = 1.0,
                 weight_decay: float = 0.05,
                 warmup_steps: int = 200,
                 max_steps: int = 5000) -> None:
        super().__init__()
        self.save_hyperparameters()

        if pretrained_checkpoint:
            mae = MAEPretrainModule.load_from_checkpoint(
                pretrained_checkpoint, map_location="cpu")
            self.vision = mae.model.encoder
            vision_dim = mae.hparams.embed_dim
            log.info("loaded pretrained encoder (embed_dim=%d)", vision_dim)
        else:
            self.vision = MultiStreamViT(
                volume_shape=tuple(volume_shape), patch_size=patch_size,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                fusion_every=fusion_every)
            vision_dim = embed_dim

        if freeze_vision:
            self.vision.requires_grad_(False)
            log.warning("vision encoder FROZEN -- this is the control arm")

        self.head = StructuredReportHead(
            embed_dim=vision_dim, n_organs=n_organs, max_lesions=max_lesions,
            hidden=head_hidden, dropout=dropout)

        self._val_rows: list[dict] = []

    # ------------------------------------------------------------------ steps

    def _encode(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.hparams.freeze_vision:
            with torch.no_grad():
                _, pooled = self.vision(image, mask)
            return pooled.detach()
        _, pooled = self.vision(image, mask)
        return pooled

    def forward(self, batch: dict) -> dict:
        current = self._encode(batch["image"], batch["modality_mask"])
        prior = self._encode(batch["prior_image"], batch["prior_modality_mask"])
        return self.head(current, prior, batch["prior_present"])

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = report_loss(self(batch), batch)
        n = batch["image"].shape[0]
        self.log("train/loss", losses["loss"], prog_bar=True, batch_size=n)
        self.log("train/diameter_mae_mm", losses["diameter_mae_mm"],
                 prog_bar=True, batch_size=n)
        for key in ("count", "organ", "new_lesion"):
            self.log(f"train/{key}", losses[key], batch_size=n)
        for key in ("count", "organ", "new_lesion", "change_mae_mm",
                            "consistency_mm"):
            self.log(f"train/{key}", losses[key], batch_size=n)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        prediction = self(batch)
        losses = report_loss(prediction, batch)
        n = batch["image"].shape[0]
        self.log("val/change_mae_mm", losses["change_mae_mm"],
                 prog_bar=True, batch_size=n)
        self.log("val/consistency_mm", losses["consistency_mm"], batch_size=n)
        self.log("val/loss", losses["loss"], prog_bar=True, batch_size=n)
        self.log("val/diameter_mae_mm", losses["diameter_mae_mm"],
                 prog_bar=True, batch_size=n)

        # Keep what is needed to DERIVE RECIST once, over the whole split.
        counts = prediction["lesion_count_logits"].argmax(dim=-1)
        for i in range(n):
            k = int(counts[i])
            self._val_rows.append({
                "pred_sld": float(prediction["diameters_mm"][i, :k].sum()),
                "true_sld": float(batch["sld_mm"][i]),
                "baseline": float(batch["baseline_sld_mm"][i]),
                "nadir": float(batch["nadir_sld_mm"][i]),
                "true_response": int(batch["response"][i]),
                "is_baseline": float(batch["prior_present"][i]) == 0.0,
            })

    def on_validation_epoch_end(self) -> None:
        """RECIST accuracy DERIVED from predicted diameters.

        This is the metric Aim 2 is actually about. Diameter MAE says how good
        the measurements are; derived accuracy says whether the resulting
        clinical statement is right, and the two answer different questions.
        """
        rows = self._val_rows
        self._val_rows = []
        if not rows:
            return

        correct = np.zeros(len(RESPONSES))
        total = np.zeros(len(RESPONSES))
        n_right = 0
        for row in rows:
            if row["is_baseline"]:
                predicted = RECISTResponse.SD      # baseline is SD by definition
            else:
                predicted, _ = classify(row["pred_sld"], row["baseline"],
                                        row["nadir"])
            truth = RESPONSES[row["true_response"]]
            total[row["true_response"]] += 1
            if predicted is truth:
                correct[row["true_response"]] += 1
                n_right += 1

        self.log("val/recist_accuracy", n_right / len(rows), prog_bar=True)
        # Overall accuracy hides the class that changes management: a model
        # that never predicts PD still scores ~55% on this label mix.
        recalls = [correct[i] / total[i] for i in range(len(RESPONSES))
                   if total[i] > 0]
        self.log("val/recist_balanced_accuracy", float(np.mean(recalls)))
        for i, response in enumerate(RESPONSES):
            if total[i] > 0:
                self.log(f"val/recall_{response.name}", correct[i] / total[i],
                         prog_bar=(response is RECISTResponse.PD))
        self.log("val/n_examples", float(len(rows)))

    def on_train_epoch_start(self) -> None:
        loader = getattr(self.trainer, "train_dataloader", None)
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    # -------------------------------------------------------------- optimizer

    def configure_optimizers(self):
        lr = self.hparams.base_lr * self.hparams.effective_batch_size / 256
        groups: dict[tuple, list] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim <= 1 or any(k in name for k in NO_DECAY_NAMES)
            group_lr = (lr * self.hparams.vision_lr_multiplier
                        if name.startswith("vision.") else lr)
            groups.setdefault((group_lr, no_decay), []).append(param)

        optimizer = torch.optim.AdamW(
            [{"params": params, "lr": group_lr,
              "weight_decay": 0.0 if no_decay else self.hparams.weight_decay}
             for (group_lr, no_decay), params in groups.items()],
            betas=(0.9, 0.95))

        warmup, total = self.hparams.warmup_steps, self.hparams.max_steps

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, factor), "interval": "step"}}