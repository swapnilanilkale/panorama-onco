from __future__ import annotations

import math
from math import comb

import lightning as L
import torch

from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.train.mae_module import NO_DECAY_NAMES, MAEPretrainModule
from panorama.vision.encoder import MultiStreamViT
from panorama.vlm.contrastive import ContrastiveAlignment, retrieval_metrics
from panorama.vlm.text_encoder import ReportTextEncoder

log = get_logger(__name__)


class AlignmentModule(L.LightningModule):
    """Contrastive image-report alignment (Aim 2, stage 1)."""

    def __init__(self,
                 vocab_size: int,
                 pretrained_checkpoint: str | None = None,
                 freeze_vision: bool = True,
                 vision_lr_multiplier: float = 0.1,
                 # vision -- REQUIRED when no checkpoint is supplied, so the
                 # scratch control matches the pretrained encoder exactly
                 volume_shape: tuple[int, int, int] | None = None,
                 patch_size: int | None = None,
                 embed_dim: int | None = None,
                 depth: int | None = None,
                 num_heads: int | None = None,
                 fusion_every: int | None = None,
                 # text
                 text_dim: int = 256,
                 text_depth: int = 4,
                 text_heads: int = 8,
                 max_text_length: int = 192,
                 pad_id: int = 0,
                 # alignment
                 projection_dim: int = 256,
                 dropout: float = 0.1,
                 # optimisation
                 base_lr: float = 1.0e-4,
                 effective_batch_size: int = 256,
                 weight_decay: float = 0.05,
                 warmup_steps: int = 200,
                 max_steps: int = 10_000) -> None:
        super().__init__()
        self.save_hyperparameters()

        if pretrained_checkpoint:
            mae = MAEPretrainModule.load_from_checkpoint(
                pretrained_checkpoint, map_location="cpu")
            self.vision = mae.model.encoder          # discard the MAE decoder
            vision_dim = mae.hparams.embed_dim
            log.info("loaded pretrained vision encoder from %s (embed_dim=%d)",
                     pretrained_checkpoint, vision_dim)
        else:
            # Scratch control for the pretraining ablation. The architecture MUST
            # be given explicitly and match the pretrained encoder exactly, or
            # the comparison measures capacity rather than pretraining.
            missing = [k for k in ("volume_shape", "patch_size", "embed_dim",
                                   "depth", "num_heads", "fusion_every")
                       if self.hparams.get(k) is None]
            if missing:
                raise ConfigError(
                    f"no pretrained_checkpoint, so the vision architecture must be "
                    f"specified explicitly; missing: {missing}")
            self.vision = MultiStreamViT(
                volume_shape=tuple(volume_shape), patch_size=patch_size,
                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                fusion_every=fusion_every)
            vision_dim = embed_dim
            log.warning("vision encoder starts from SCRATCH (%d parameters) -- "
                        "this is the ablation control, not a normal run",
                        sum(p.numel() for p in self.vision.parameters()))

        if freeze_vision:
            self.vision.requires_grad_(False)
            log.info("vision encoder FROZEN (%d parameters)",
                     sum(p.numel() for p in self.vision.parameters()))

        self.text = ReportTextEncoder(
            vocab_size=vocab_size, embed_dim=text_dim, depth=text_depth,
            num_heads=text_heads, max_length=max_text_length,
            dropout=dropout, pad_id=pad_id)
        self.align = ContrastiveAlignment(
            image_dim=vision_dim, text_dim=text_dim,
            embed_dim=projection_dim, dropout=dropout)

        # Accumulated across a whole eval split, so recall is not batch-limited.
        self._val_image: list[torch.Tensor] = []
        self._val_text: list[torch.Tensor] = []

    # ------------------------------------------------------------------ steps

    def embed(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hparams.freeze_vision:
            with torch.no_grad():
                _, pooled = self.vision(batch["image"], batch["modality_mask"])
            pooled = pooled.detach()
        else:
            _, pooled = self.vision(batch["image"], batch["modality_mask"])
        text = self.text(batch["token_ids"], batch["attention_mask"])
        return pooled, text

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        image, text = self.embed(batch)
        out = self.align(image, text, pair_ids=batch.get("pair_id"))
        n = image.shape[0]
        self.log("train/loss", out["loss"], prog_bar=True, batch_size=n)
        self.log("train/temperature", self.align.temperature, batch_size=n)
        with torch.no_grad():
            m = retrieval_metrics(out["logits"])
            self.log("train/i2t_r1", m["i2t_r1"], prog_bar=True, batch_size=n)
        return out["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        image, text = self.embed(batch)
        img_p, txt_p = self.align.encode(image, text)
        # Store projected embeddings; score once over the FULL split, because
        # recall computed within a batch is easier the smaller the batch.
        self._val_image.append(img_p.detach().cpu())
        self._val_text.append(txt_p.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_image:
            return
        img = torch.cat(self._val_image)
        txt = torch.cat(self._val_text)
        self._val_image.clear()
        self._val_text.clear()

        scale = self.align.logit_scale.exp().clamp(max=100.0).detach().cpu()
        logits = scale * img @ txt.t()
        targets = torch.arange(logits.shape[0])

        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, targets)
                      + torch.nn.functional.cross_entropy(logits.t(), targets))
        self.log("val/loss", loss, prog_bar=True)
        for key, value in retrieval_metrics(logits).items():
            self.log(f"val/{key}", value, prog_bar=key in ("i2t_r1", "chance_r1"))

        # A recall number is uninterpretable without knowing whether the split
        # is large enough to distinguish it from chance.
        n = logits.shape[0]
        correct = int((logits.argmax(dim=1) == targets).sum())
        p_chance = 1.0 / n
        p_value = sum(comb(n, k) * p_chance ** k * (1 - p_chance) ** (n - k)
                      for k in range(correct, n + 1))
        self.log("val/n_candidates", float(n))
        self.log("val/i2t_correct", float(correct))
        self.log("val/i2t_p_value", float(p_value), prog_bar=True)

    def on_train_epoch_start(self) -> None:
        loader = getattr(self.trainer, "train_dataloader", None)
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    # -------------------------------------------------------------- optimizer

    def _param_groups(self) -> list[dict]:
        lr = self.hparams.base_lr * self.hparams.effective_batch_size / 256
        vision_lr = lr * self.hparams.vision_lr_multiplier
        groups: dict[tuple, list] = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim <= 1 or any(k in name for k in NO_DECAY_NAMES)
            # Pretrained weights move slowly; new heads move at full speed.
            group_lr = vision_lr if name.startswith("vision.") else lr
            groups.setdefault((group_lr, no_decay), []).append(param)

        return [{"params": params, "lr": group_lr,
                 "weight_decay": 0.0 if no_decay else self.hparams.weight_decay}
                for (group_lr, no_decay), params in groups.items()]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._param_groups(), betas=(0.9, 0.98))
        warmup, total = self.hparams.warmup_steps, self.hparams.max_steps

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}