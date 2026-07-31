from __future__ import annotations

import csv
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.dataset import MultiModalPatchDataset
from panorama.data.manifest import read_manifest
from panorama.data.resample import DEFAULT_SPACING_MM
from panorama.data.splits import CohortSplit, patient_level_split
from panorama.data.patches import DEFAULT_PATCH

log = get_logger(__name__)


class PanoramaDataModule(L.LightningDataModule):
    """manifest -> patient-level splits -> patch datasets -> dataloaders."""

    def __init__(self,
                 manifest_path: Path | str,
                 data_root: Path | str,
                 crop_size: tuple[int, int, int] = DEFAULT_PATCH,
                 target_spacing: tuple[float, float, float] = DEFAULT_SPACING_MM,
                 batch_size: int = 2,
                 num_workers: int = 0,
                 patches_per_study: int = 4,
                 fg_threshold: float | None = 0.1,
                 val_fraction: float = 0.15,
                 test_fraction: float = 0.15,
                 seed: int = 1337,
                 split_file: Path | str | None = None) -> None:
        super().__init__()
        # Store paths as STRINGS: keeps checkpoints loadable with the safe
        # weights_only=True default, and portable between Windows and Linux.
        manifest_path = str(manifest_path)
        data_root = str(data_root)
        split_file = str(split_file) if split_file is not None else None
        self.save_hyperparameters()
        self.split: CohortSplit | None = None
        self._datasets: dict[str, MultiModalPatchDataset] = {}


    # --------------------------------------------------------------- hooks

    def prepare_data(self) -> None:
        """Rank 0 only, once. Verify inputs exist -- never build state here."""
        manifest = Path(self.hparams.manifest_path)
        if not manifest.is_file():
            raise ConfigError(
                f"manifest not found: {manifest}. Build one with "
                f"`panorama.data.manifest.scan_directory` + `write_manifest`.")
        if not Path(self.hparams.data_root).is_dir():
            raise ConfigError(f"data_root not found: {self.hparams.data_root}")

    def setup(self, stage: str | None = None) -> None:
        """Runs in EVERY process. Deterministic, so all ranks agree."""
        if self._datasets:
            return

        studies = read_manifest(self.hparams.manifest_path, self.hparams.data_root)
        log.info("manifest: %d studies, %d patients",
                 len(studies), len({s.patient_id for s in studies}))

        self.split = patient_level_split(
            studies,
            val_fraction=self.hparams.val_fraction,
            test_fraction=self.hparams.test_fraction,
            seed=self.hparams.seed)
        log.info("cohort split:\n%s", self.split.summary())

        common = dict(crop_size=self.hparams.crop_size,               
                      target_spacing=self.hparams.target_spacing,
                      seed=self.hparams.seed)

        
        
        self._datasets = {
            # Train: many random crops per study, biased toward foreground.
            "train": MultiModalPatchDataset(
                self.split.train, patches_per_study=self.hparams.patches_per_study,
                fg_threshold=self.hparams.fg_threshold, **common),
            # Val/test: ONE crop per study, so the metric is not resampled noise.
            "val": MultiModalPatchDataset(
                self.split.val, patches_per_study=1,
                fg_threshold=self.hparams.fg_threshold, **common),
            "test": MultiModalPatchDataset(
                self.split.test, patches_per_study=1,
                fg_threshold=self.hparams.fg_threshold, **common),
        }

        if self.hparams.split_file:
            self.save_split(self.hparams.split_file)

    # ---------------------------------------------------------- dataloaders

    def _loader(self, name: str, shuffle: bool) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(
            self._datasets[name],
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=True,
            drop_last=shuffle,                  # only for train
            persistent_workers=nw > 0,          # worker startup is expensive
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    # ------------------------------------------------------- auditability

    def save_split(self, path: Path | str) -> Path:
        """Freeze which patient went to which split -- a committable artifact."""
        if self.split is None:
            raise ConfigError("call setup() before save_split()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = self.split.patient_ids()
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["split", "patient_id"])
            for name in ("train", "val", "test"):
                for pid in sorted(ids[name]):
                    writer.writerow([name, pid])
        log.info("split written to %s", path)
        return path