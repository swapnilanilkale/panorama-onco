from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from panorama.core.exceptions import ConfigError
from panorama.core.logging import configure_logging, get_logger
from panorama.data.datamodule import PanoramaDataModule
from panorama.train.mae_module import MAEPretrainModule
from panorama.utils.reproducibility import git_revision, seed_everything

log = get_logger(__name__)
DEFAULT_CONFIG = Path("configs/pretrain.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG,
                overrides: list[str] | None = None) -> DictConfig:
    """YAML base + `key.subkey=value` CLI overrides (deep-merged)."""
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
            f"({cfg.model.max_steps}). The LR schedule and the training length "
            f"must agree or the cosine anneal will be wrong.")
    if cfg.model.warmup_steps >= cfg.model.max_steps:
        raise ConfigError(
            f"warmup_steps ({cfg.model.warmup_steps}) >= max_steps "
            f"({cfg.model.max_steps}): the LR would never decay.")
    if tuple(cfg.data.patch_size) != tuple(cfg.model.volume_shape):
        raise ConfigError(
            f"data.patch_size {list(cfg.data.patch_size)} != model.volume_shape "
            f"{list(cfg.model.volume_shape)}: the encoder would compute the wrong "
            f"token count.")
    for name in ("base_lr", "weight_decay"):
        value = cfg.model[name]
        if not isinstance(value, (int, float)):
            raise ConfigError(
                f"model.{name} is {value!r} ({type(value).__name__}), not a number. "
                f"In YAML write 1.5e-4 (with a decimal point), never 3e-4.")


def run(cfg: DictConfig) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    revision = git_revision()
    log.info("git revision: %s", revision)

    # The resolved config IS the experiment record -- save it before training.
    snapshot = OmegaConf.create({"git_revision": revision, **OmegaConf.to_container(cfg)})
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(snapshot), encoding="utf-8")

    datamodule = PanoramaDataModule(
        **OmegaConf.to_container(cfg.data, resolve=True),
        seed=cfg.seed, split_file=out_dir / "splits.csv")
    module = MAEPretrainModule(
        **OmegaConf.to_container(cfg.model, resolve=True),
        effective_batch_size=cfg.data.batch_size * cfg.trainer.accumulate_grad_batches)

    callbacks = [
        ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="step{step}-{val/loss:.4f}",
                        monitor="val/loss", mode="min", save_top_k=3,
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
    if args and not "=" in args[0]:
        config_path, args = Path(args[0]), args[1:]
    cfg = load_config(config_path, args)
    validate_config(cfg)
    log.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":       # required on Windows for num_workers > 0
    main()