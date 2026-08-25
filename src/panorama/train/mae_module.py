from __future__ import annotations

import numpy as np
import math

import lightning as L
import torch

from panorama.vision.encoder import MultiStreamViT
from panorama.vision.mae import MultiModalMAE

# These carry signal that weight decay would erode toward zero.
NO_DECAY_NAMES = ("pos_embed", "modality_embed", "missing_token", "mask_token")

def effective_rank(features: np.ndarray, threshold: float = 0.95) -> int:
    """Number of principal directions holding `threshold` of the variance.

    A representation that collapses to a handful of dimensions cannot support
    downstream tasks however low its reconstruction loss. Note the measure is
    capped at min(n_samples, n_dims), so it must be computed over an accumulated
    set of embeddings -- at batch_size=2 a per-batch value could never exceed 2.
    """
    if features.shape[0] < 2:
        return 0
    centred = features - features.mean(axis=0, keepdims=True)
    centred = centred / (centred.std(axis=0, keepdims=True) + 1e-8)
    variance = np.linalg.svd(centred, compute_uv=False) ** 2
    total = variance.sum()
    if total <= 0:
        return 0
    return int((np.cumsum(variance) / total < threshold).sum()) + 1

class MAEPretrainModule(L.LightningModule):
    """Self-supervised pretraining of the Aim 1 encoder."""

    def __init__(self,
                 volume_shape: tuple[int, int, int] = (96, 96, 96),
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 fusion_every: int = 4,
                 share_stream_weights: bool = True,
                 mask_ratio: float = 0.75,
                 decoder_dim: int = 256,
                 decoder_depth: int = 2,
                 decoder_heads: int = 8,
                 norm_pix_loss: bool = True,
                 base_lr: float = 1.5e-4,
                 effective_batch_size: int = 256,
                 weight_decay: float = 0.05,
                 warmup_steps: int = 1000,
                 max_steps: int = 100_000,
                 min_lr_ratio: float = 0.0) -> None:
        super().__init__()
        self.save_hyperparameters()      # checkpoints then carry their own config
                # Pooled embeddings accumulated over a validation epoch, for the
        # effective-rank diagnostic.
        self._val_pooled: list[np.ndarray] = []

        encoder = MultiStreamViT(
            volume_shape=volume_shape, patch_size=patch_size, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, fusion_every=fusion_every,
            share_stream_weights=share_stream_weights)
        self.model = MultiModalMAE(
            encoder, patch_size=patch_size, mask_ratio=mask_ratio,
            decoder_dim=decoder_dim, decoder_depth=decoder_depth,
            decoder_heads=decoder_heads, norm_pix_loss=norm_pix_loss)

    # ------------------------------------------------------------------ steps

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        out = self.model(batch["image"], batch["modality_mask"])
        loss = out["loss"]
        n = batch["image"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"),
                 on_step=(stage == "train"), on_epoch=True, batch_size=n)
        self.log(f"{stage}/masked_tokens",
                 out["token_mask"].float().sum() / n,
                 on_step=False, on_epoch=True, batch_size=n)
        if self.hparams.norm_pix_loss:
            self.log(f"{stage}/variance_explained", 1.0 - loss,
                     prog_bar=(stage == "train"), on_step=False,
                     on_epoch=True, batch_size=n)
        if stage == "val":
            self._val_pooled.append(out["pooled"].detach().cpu().numpy())
        return loss


    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_train_epoch_start(self) -> None:
        # Lets the Dataset vary its patch crops across epochs (see M1.8).
        ds = getattr(self.trainer, "train_dataloader", None)
        ds = getattr(ds, "dataset", None)
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        """Effective rank of the pooled representation.

        Reconstruction loss can improve while the feature space narrows, so the
        MAE objective alone does not reveal collapse. Logging rank per epoch
        turns "pretraining did not help" into "pretraining collapsed at step N",
        which is a far more actionable statement.
        """
        if not self._val_pooled:
            return
        pooled = np.concatenate(self._val_pooled)
        self._val_pooled.clear()
        rank = effective_rank(pooled)
        self.log("val/effective_rank", float(rank))
        # The ceiling is min(n_samples, embed_dim); log it so a low rank is not
        # misread when the val set is small.
        self.log("val/rank_ceiling", float(min(pooled.shape)))
        self.log("val/rank_fraction", rank / max(1, min(pooled.shape)))

    # ------------------------------------------------------------- optimizer

    def _param_groups(self) -> list[dict]:
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or any(k in name for k in NO_DECAY_NAMES):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        # Linear scaling rule: keep per-sample update size constant.
        lr = self.hparams.base_lr * self.hparams.effective_batch_size / 256

        optimizer = torch.optim.AdamW(self._param_groups(), lr=lr,
                                      betas=(0.9, 0.95))

        warmup = self.hparams.warmup_steps
        total = self.hparams.max_steps
        min_ratio = self.hparams.min_lr_ratio

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}