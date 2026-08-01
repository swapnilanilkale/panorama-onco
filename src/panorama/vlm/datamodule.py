"""DataModule for contrastive image-report alignment."""
from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from panorama.clinical.corpus import read_corpus
from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.manifest import read_manifest
from panorama.data.patches import DEFAULT_PATCH
from panorama.data.resample import DEFAULT_SPACING_MM
from panorama.data.splits import CohortSplit, patient_level_split
from panorama.vlm.dataset import ImageReportDataset
from panorama.vlm.tokenizer import ReportTokenizer

log = get_logger(__name__)


class AlignmentDataModule(L.LightningDataModule):
    """manifest + report corpus -> paired train/val/test loaders."""

    def __init__(self,
                 manifest_path: Path | str,
                 data_root: Path | str,
                 corpus_path: Path | str,
                 tokenizer_path: Path | str | None = None,
                 crop_size: tuple[int, int, int] = DEFAULT_PATCH,
                 target_spacing: tuple[float, float, float] = DEFAULT_SPACING_MM,
                 max_text_length: int = 192,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 patches_per_study: int = 4,
                 fg_threshold: float | None = 0.1,
                 val_fraction: float = 0.15,
                 test_fraction: float = 0.15,
                 seed: int = 1337) -> None:
        super().__init__()
        manifest_path = str(manifest_path)
        data_root = str(data_root)
        corpus_path = str(corpus_path)
        tokenizer_path = str(tokenizer_path) if tokenizer_path else None
        self.save_hyperparameters()
        self.split: CohortSplit | None = None
        self.tokenizer: ReportTokenizer | None = None
        self._datasets: dict[str, ImageReportDataset] = {}

    def prepare_data(self) -> None:
        for name in ("manifest_path", "corpus_path"):
            if not Path(self.hparams[name]).is_file():
                raise ConfigError(f"{name} not found: {self.hparams[name]}")

    def setup(self, stage: str | None = None) -> None:
        if self._datasets:
            return

        studies = read_manifest(self.hparams.manifest_path, self.hparams.data_root)
        corpus = read_corpus(self.hparams.corpus_path)
        self.split = patient_level_split(
            studies, self.hparams.val_fraction, self.hparams.test_fraction,
            self.hparams.seed)
        log.info("cohort split:\n%s", self.split.summary())

        if self.hparams.tokenizer_path and Path(self.hparams.tokenizer_path).is_file():
            self.tokenizer = ReportTokenizer.load(self.hparams.tokenizer_path)
        else:
            # Build the vocabulary from TRAIN reports only: test-set words must
            # not influence the model's inputs.
            train_ids = {s.study_id for s in self.split.train}
            self.tokenizer = ReportTokenizer.build(
                [r["report"] for sid, r in corpus.items() if sid in train_ids],
                max_length=self.hparams.max_text_length)
            if self.hparams.tokenizer_path:
                self.tokenizer.save(self.hparams.tokenizer_path)
        log.info("tokenizer vocabulary: %d tokens", len(self.tokenizer))

        common = dict(crop_size=self.hparams.crop_size,
                      target_spacing=self.hparams.target_spacing,
                      fg_threshold=self.hparams.fg_threshold,
                      seed=self.hparams.seed)

        self._datasets = {
            "train": ImageReportDataset(
                self.split.train, corpus, self.tokenizer,
                patches_per_study=self.hparams.patches_per_study, **common),
            # ONE crop per study at eval: retrieval is a per-STUDY question,
            # and duplicate crops would inflate recall.
            "val": ImageReportDataset(self.split.val, corpus, self.tokenizer,
                                      patches_per_study=1, **common),
            "test": ImageReportDataset(self.split.test, corpus, self.tokenizer,
                                       patches_per_study=1, **common),
        }

    def _loader(self, name: str, shuffle: bool) -> DataLoader:
        nw = self.hparams.num_workers
        return DataLoader(self._datasets[name],
                          batch_size=self.hparams.batch_size,
                          shuffle=shuffle, num_workers=nw,
                          pin_memory=torch.cuda.is_available(),
                          drop_last=shuffle,
                          persistent_workers=nw > 0)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)