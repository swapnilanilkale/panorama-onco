"""Pair each imaging patch with its study's radiology report."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset

from panorama.core.constants import RECISTResponse
from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.dataset import MultiModalPatchDataset
from panorama.data.schema import Study
from panorama.vlm.tokenizer import ReportTokenizer

log = get_logger(__name__)

RESPONSE_INDEX = {r: i for i, r in enumerate(RECISTResponse)}


class ImageReportDataset(Dataset):
    """Wraps MultiModalPatchDataset, attaching the study's tokenized report."""

    def __init__(self, studies: Sequence[Study], corpus: dict[str, dict],
                 tokenizer: ReportTokenizer, **patch_kwargs) -> None:
        paired = [s for s in studies if s.study_id in corpus]
        if not paired:
            raise ConfigError("no study in this split has a report in the corpus")
        if len(paired) < len(studies):
            log.warning("%d of %d studies have no report and were dropped",
                        len(studies) - len(paired), len(studies))

        self.patches = MultiModalPatchDataset(paired, **patch_kwargs)
        self.corpus = corpus
        self.tokenizer = tokenizer
        # Stable integer id per study: lets the collated batch mark which
        # samples are crops of the SAME study (see ContrastiveAlignment).
        self.study_index = {s.study_id: i for i, s in enumerate(paired)}

    def __len__(self) -> int:
        return len(self.patches)

    def set_epoch(self, epoch: int) -> None:
        self.patches.set_epoch(epoch)

    def __getitem__(self, idx: int) -> dict:
        sample = self.patches[idx]
        study_id = sample["study_id"]
        record = self.corpus[study_id]

        ids, mask = self.tokenizer.encode(record["report"])
        sample["token_ids"] = torch.tensor(ids, dtype=torch.long)
        sample["attention_mask"] = torch.tensor(mask, dtype=torch.long)
        sample["pair_id"] = torch.tensor(self.study_index[study_id], dtype=torch.long)
        sample["response"] = torch.tensor(
            RESPONSE_INDEX[RECISTResponse(record["response"])], dtype=torch.long)
        return sample