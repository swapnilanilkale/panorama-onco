from __future__ import annotations

import math

import lightning as L
import torch

from panorama.vision.encoder import MultiStreamViT
from panorama.vision.mae import MultiModalMAE

# These carry signal that weight decay would erode toward zero.
NO_DECAY_NAMES = ("pos_embed", "modality_embed", "missing_token", "mask_token")


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