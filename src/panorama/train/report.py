"""Entry point for Aim 2 structured report generation.

    python -m panorama.train.report configs/report_synth.yaml
    python -m panorama.train.report configs/report_synth.yaml model.freeze_vision=true
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
from panorama.train.report_module import ReportModule
from panorama.utils.reproducibility import git_revision, seed_everything
from panorama.vlm.report_datamodule import ReportDataModule

log = get_logger(__name__)
DEFAULT_CONFIG = Path("configs/report_synth.yaml")


def load_config(path, overrides=None) -> DictConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def validate_config(cfg: DictConfig) -> None:
    if cfg.trainer.max_steps != cfg.model.max_steps:
        raise ConfigError(
            f"trainer.max_steps ({cfg.trainer.max_steps}) != model.max_steps "
            f"({cfg.model.max_steps}); the LR schedule would be wrong.")
    if tuple(cfg.data.crop_size) != tuple(cfg.model.volume_shape):
        raise ConfigError(
            f"data.crop_size != model.volume_shape: the encoder would compute "
            f"the wrong token count.")
    if cfg.data.max_lesions != cfg.model.max_lesions:
        raise ConfigError(
            f"data.max_lesions ({cfg.data.max_lesions}) != model.max_lesions "
            f"({cfg.model.max_lesions}); target and prediction shapes must match.")
    if cfg.trainer.get("val_check_interval") is not None and \
            cfg.trainer.get("check_val_every_n_epoch", 1) is not None:
        raise ConfigError(
            "trainer.val_check_interval is counted in within-epoch batches "
            "unless trainer.check_val_every_n_epoch is null.")


def run(cfg: DictConfig) -> Path:
    out_dir = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    log.info("run directory: %s", out_dir)

    seed_everything(cfg.seed)
    snapshot = OmegaConf.create({"git_revision": git_revision(),
                                 **OmegaConf.to_container(cfg)})
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(snapshot), encoding="utf-8")

    datamodule = ReportDataModule(
        **OmegaConf.to_container(cfg.data, resolve=True), seed=cfg.seed)
    datamodule.prepare_data()
    datamodule.setup("fit")          # n_organs is needed to build the head

    module = ReportModule(
        n_organs=datamodule.n_organs,
        pretrained_checkpoint=cfg.get("pretrained_checkpoint"),
        effective_batch_size=cfg.data.batch_size * cfg.trainer.accumulate_grad_batches,
        **OmegaConf.to_container(cfg.model, resolve=True))

    callbacks = [
        # Select on the CLINICAL metric, not the loss.
        ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="step{step:06d}",
                        monitor="val/recist_balanced_accuracy", mode="max",
                        save_top_k=3, save_last=True, auto_insert_metric_name=False),
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