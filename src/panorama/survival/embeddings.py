"""Cache study-level embeddings so the timeline encoder trains on tensors.

The whole synthetic cohort becomes ~1.4 MB of embeddings against ~94 GB of
preprocessed volumes, so an epoch costs a matmul rather than disk reads and
resampling -- seconds instead of minutes on CPU.

It also isolates the question. Aim 3 asks whether temporal modelling recovers
growth; running the vision encoder inside the loop would confound that with
whether the vision encoder learns anything, which ADR-0009 already answered
separately.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from panorama.core.logging import get_logger
from panorama.data.dataset import MultiModalPatchDataset
from panorama.data.schema import Study

log = get_logger(__name__)


@torch.no_grad()
def cache_study_embeddings(encoder, studies: Sequence[Study],
                           crops_per_study: int = 4,
                           pooling: str = "mean",
                           device: str = "cpu",
                           **patch_kwargs) -> dict[str, np.ndarray]:
    """study_id -> pooled embedding, averaged over several crops.

    A 64mm crop sees roughly one lesion, so one crop cannot represent a study.
    Mean pooling over crops is used here because lesion count barely varies in
    this cohort (2-3); with wider variation, SUM pooling would better match how
    tumour burden is built -- it is a sum over lesions, not an average.
    """
    if pooling not in ("mean", "sum"):
        raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")

    dataset = MultiModalPatchDataset(list(studies),
                                     patches_per_study=crops_per_study,
                                     **patch_kwargs)
    encoder = encoder.to(device).eval()

    collected: dict[str, list[np.ndarray]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["modality_mask"].unsqueeze(0).to(device)
        _, pooled = encoder(image, mask)
        collected.setdefault(sample["study_id"], []).append(
            pooled.squeeze(0).cpu().numpy())

    reduce = np.mean if pooling == "mean" else np.sum
    embeddings = {sid: reduce(vectors, axis=0).astype(np.float32)
                  for sid, vectors in collected.items()}
    log.info("cached %d study embeddings (%d dims, %s over %d crops)",
             len(embeddings), len(next(iter(embeddings.values()))),
             pooling, crops_per_study)
    return embeddings


def save_embeddings(embeddings: dict[str, np.ndarray],
                    path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **embeddings)
    return path


def load_embeddings(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}