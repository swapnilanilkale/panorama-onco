"""Patient timelines with survival outcomes, from cached embeddings.

Full-batch by design. The Cox partial likelihood compares each event against
everyone at risk IN THE SAME BATCH, so a mini-batch gives small risk sets and a
biased estimate -- at batch 8 the mean risk set is ~4 against a true ~100. With
the whole cohort as 1.4 MB of embeddings there is no reason to approximate.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from panorama.core.exceptions import ConfigError
from panorama.core.logging import get_logger
from panorama.data.schema import PatientTimeline
from panorama.survival.data import SurvivalOutcome
from panorama.survival.timeline import collate_timelines

log = get_logger(__name__)


class TimelineCohort:
    """A whole split as padded tensors -- embeddings, elapsed days, outcomes."""

    def __init__(self, timelines: Sequence[PatientTimeline],
                 embeddings: dict[str, np.ndarray],
                 outcomes: dict[str, SurvivalOutcome],
                 max_timepoints: int | None = None) -> None:
        rows = []
        for timeline in timelines:
            outcome = outcomes.get(timeline.patient_id)
            if outcome is None:
                continue
            studies = [s for s in timeline.studies if s.study_id in embeddings]
            if len(studies) < 2:
                continue                    # no interval, so no temporal signal
            if max_timepoints:
                studies = studies[:max_timepoints]
            rows.append((timeline.patient_id, studies, outcome))

        if not rows:
            raise ConfigError("no patient has both embeddings and an outcome")

        self.patient_ids = [pid for pid, _, _ in rows]
        embed_list = [torch.tensor(np.stack([embeddings[s.study_id] for s in studies]))
                      for _, studies, _ in rows]
        day_list = [torch.tensor([float((s.acquired_on - studies[0].acquired_on).days)
                                  for s in studies])
                    for _, studies, _ in rows]

        self.embeddings, self.days, self.mask = collate_timelines(embed_list, day_list)
        self.duration = torch.tensor([o.duration_days for _, _, o in rows],
                                     dtype=torch.float32)
        self.event = torch.tensor([o.event for _, _, o in rows])

        log.info("cohort: %d patients, %d events (%.0f%%), timelines %d-%d studies",
                 len(rows), int(self.event.sum()), 100 * float(self.event.float().mean()),
                 int(self.mask.sum(1).min()), int(self.mask.sum(1).max()))

    def __len__(self) -> int:
        return len(self.patient_ids)

    def to(self, device: str) -> "TimelineCohort":
        for name in ("embeddings", "days", "mask", "duration", "event"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[-1]

    def baseline_only(self) -> torch.Tensor:
        """First study's embedding -- the no-temporal-information control."""
        return self.embeddings[:, 0]