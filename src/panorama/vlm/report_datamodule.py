"""DataModule for Aim 2 report generation."""
from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.manifest import read_manifest
from panorama.data.patches import DEFAULT_PATCH
from panorama.data.resample import DEFAULT_SPACING_MM
from panorama.data.splits import CohortSplit, patient_level_split
from panorama.data.synthetic import REGIONS, read_lesions
from panorama.vlm.report_dataset import StudyPairDataset

log = get_logger(__name__)


class ReportDataModule(L.LightningDataModule):
    """manifest + lesion ground truth -> study-pair loaders."""

    def __init__(self,
                 manifest_path: Path | str,
                 data_root: Path | str,
                 lesions_path: Path | str,
                 crop_size: tuple[int, int, int] = DEFAULT_PATCH,
                 target_spacing: tuple[float, float, float] = DEFAULT_SPACING_MM,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 patches_per_study: int = 4,
                 fg_threshold: float | None = 0.3,
                 max_lesions: int = 4,
                 val_fraction: float = 0.15,
                 test_fraction: float = 0.15,
                 seed: int = 1337) -> None:
        super().__init__()
        manifest_path = str(manifest_path)
        data_root = str(data_root)
        lesions_path = str(lesions_path)
        self.save_hyperparameters()
        self.split: CohortSplit | None = None
        self.organs: list[str] = [name for _, _, name in REGIONS]
        self._datasets: dict[str, StudyPairDataset] = {}

    @property
    def n_organs(self) -> int:
        return len(self.organs)

    def prepare_data(self) -> None:
        for name in ("manifest_path", "lesions_path"):
            if not Path(self.hparams[name]).is_file():
                raise ConfigError(f"{name} not found: {self.hparams[name]}")

    def setup(self, stage: str | None = None) -> None:
        if self._datasets:
            return

        studies = read_manifest(self.hparams.manifest_path, self.hparams.data_root)
        lesions = read_lesions(self.hparams.lesions_path)
        self.split = patient_level_split(
            studies, self.hparams.val_fraction, self.hparams.test_fraction,
            self.hparams.seed)
        log.info("cohort split:\n%s", self.split.summary())

        common = dict(lesions_by_study=lesions, organs=self.organs,
                      max_lesions=self.hparams.max_lesions,
                      crop_size=self.hparams.crop_size,
                      target_spacing=self.hparams.target_spacing,
                      fg_threshold=self.hparams.fg_threshold,
                      seed=self.hparams.seed)

        self._datasets = {
            "train": StudyPairDataset(
                self.split.train,
                patches_per_study=self.hparams.patches_per_study, **common),
            # Two crops per study at eval: the dataset needs a distinct crop for
            # the current and prior roles, so one would collide.
            "val": StudyPairDataset(self.split.val, patches_per_study=2, **common),
            "test": StudyPairDataset(self.split.test, patches_per_study=2, **common),
        }

    def _loader(self, name: str, shuffle: bool) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(self._datasets[name],
                          batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=nw,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=shuffle, persistent_workers=nw > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)