"""Entry point for contrastive image-report alignment.

    python -m panorama.train.align configs/align_smoke.yaml \
        pretrained_checkpoint=outputs/smoke/<run>/checkpoints/last.ckpt
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from panorama.core.exceptions import ConfigError
from panorama.core.logging import configure_logging, get_logger
from panorama.train.align_module import AlignmentModule
from panorama.utils.reproducibility import git_revision, seed_everything
from panorama.vlm.datamodule import AlignmentDataModule

log = get_logger(__name__)
DEFAULT_CONFIG = Path("configs/align_smoke.yaml")


def load_config(path, overrides=None) -> DictConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def validate_config(cfg: DictConfig) -> None:
    """Catch the coupling errors that silently ruin a run."""
    if cfg.trainer.max_steps != cfg.model.max_steps:
        raise ConfigError(
            f"trainer.max_steps ({cfg.trainer.max_steps}) != model.max_steps "
            f"({cfg.model.max_steps}); the LR schedule would be wrong.")
    if cfg.data.max_text_length != cfg.model.max_text_length:
        raise ConfigError(
            f"data.max_text_length ({cfg.data.max_text_length}) != "
            f"model.max_text_length ({cfg.model.max_text_length}); positional "
            f"embeddings would be sized for the wrong sequence.")
    interval = cfg.trainer.get("val_check_interval")
    if interval is not None and cfg.trainer.get("check_val_every_n_epoch", 1) is not None:
        log.warning(
            "val_check_interval=%s is counted in within-epoch batches unless "
            "trainer.check_val_every_n_epoch is null; set it to null to count "
            "global steps instead", interval)
    if not cfg.get("pretrained_checkpoint"):
        log.warning("no pretrained_checkpoint: the vision encoder starts from "
                    "scratch, which defeats the point of Aim 1 pretraining")


def run(cfg: DictConfig) -> Path:
    out_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    log.info("run directory: %s", out_dir)

    seed_everything(cfg.seed)
    revision = git_revision()
    snapshot = OmegaConf.create({"git_revision": revision,
                                 **OmegaConf.to_container(cfg)})
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(snapshot), encoding="utf-8")

    datamodule = AlignmentDataModule(
        **OmegaConf.to_container(cfg.data, resolve=True), seed=cfg.seed)
    datamodule.prepare_data()
    datamodule.setup("fit")           # need the vocabulary before building the model

    module = AlignmentModule(
        vocab_size=len(datamodule.tokenizer),
        pad_id=datamodule.tokenizer.pad_id,
        pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
        effective_batch_size=cfg.data.batch_size * cfg.trainer.accumulate_grad_batches,
        **OmegaConf.to_container(cfg.model, resolve=True))

    callbacks = [
        # Select on RECALL, not loss: loss is batch-size dependent and hard to read.
        ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="step{step:06d}",
                        monitor="val/i2t_r1", mode="max", save_top_k=3,
                        save_last=True, auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = L.Trainer(default_root_dir=out_dir, callbacks=callbacks,
                        **OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer.fit(module, datamodule=datamodule)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    configure_logging("INFO")
    args = list(sys.argv[1:] if argv is None else argv)
    config_path = DEFAULT_CONFIG
    if args and "=" not in args[0]:
        config_path, args = Path(args[0]), args[1:]
    cfg = load_config(config_path, args)
    validate_config(cfg)
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()