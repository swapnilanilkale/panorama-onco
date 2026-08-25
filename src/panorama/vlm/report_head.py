"""Predict a structured report from paired current/prior visual features.

The central design decision: the model predicts MEASUREMENTS, and the RECIST
category is COMPUTED from them by the rule in `panorama.clinical.recist` -- it
is never predicted directly.

A free-text decoder emits the number 34 as a token the language model chose;
nothing ties it to the image, so a fluent model states plausible wrong
measurements confidently, and can even assert "stable disease" while its own
quoted numbers imply progression. Predicting fields makes that internal
contradiction structurally impossible, and bounds the error to measurement
error propagating through a known, tested rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MAX_LESIONS = 4          # RECIST 1.1 allows up to 5 target lesions; 4 covers
                         # our cohort and keeps the head small.


class StructuredReportHead(nn.Module):
    """current + prior study embeddings -> the fields a report is made of.

    Input : current [B, E], prior [B, E], prior_present [B]
    Output: dict of per-field predictions
    """

    def __init__(self, embed_dim: int, n_organs: int,
                 max_lesions: int = MAX_LESIONS, hidden: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.max_lesions = max_lesions
        self.n_organs = n_organs

        # A learnable stand-in for "no prior study exists" -- the baseline case.
        # Same device as the missing-modality token in M2.2, and for the same
        # reason: absent must be distinguishable from all-zero.
        self.no_prior = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.no_prior, std=0.02)

        # The trunk sees current, prior, and their DIFFERENCE. The difference is
        # what RECIST is about, so giving it explicitly rather than making the
        # model derive it is a useful inductive bias on a small cohort.
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.lesion_count = nn.Linear(hidden, max_lesions + 1)   # 0..max
        self.diameters = nn.Linear(hidden, max_lesions)          # mm, per slot
        # Per-lesion change from the prior study, signed (no softplus -- lesions
        # shrink as well as grow). This target is NOT computable from the current
        # scan alone, so a model that ignores the prior cannot fit it. That is
        # the point: predicting absolute diameters gave the prior no work to do.
        self.diameter_change = nn.Linear(hidden, max_lesions)
        self.organs = nn.Linear(hidden, max_lesions * n_organs)
        self.new_lesion = nn.Linear(hidden, 1)                   # logit

    def forward(self, current: torch.Tensor,
                prior: torch.Tensor | None = None,
                prior_present: torch.Tensor | None = None) -> dict:
        b = current.shape[0]
        if prior is None:
            prior = self.no_prior.expand(b, -1)
        elif prior_present is not None:
            gate = prior_present.view(-1, 1)
            prior = gate * prior + (1.0 - gate) * self.no_prior

        features = self.trunk(torch.cat([current, prior, current - prior], dim=-1))

        return {
            "lesion_count_logits": self.lesion_count(features),
            # softplus keeps diameters positive without saturating: a lesion
            # cannot have negative size, and ReLU would kill gradients at 0.
            "diameters_mm": nn.functional.softplus(self.diameters(features)),
            "diameter_change_mm": self.diameter_change(features),
            "organ_logits": self.organs(features).view(b, self.max_lesions,
                                                       self.n_organs),
            "new_lesion_logit": self.new_lesion(features).squeeze(-1),
        }


def report_loss(prediction: dict, target: dict,
                change_weight: float = 1.0,
                consistency_weight: float = 0.5) -> dict:
    """Per-field losses, scored on PRESENT lesions only.

    Two diameter terms:
      * absolute -- what the current scan shows
      * change   -- current minus prior, which requires BOTH inputs

    Plus a consistency penalty tying them together: (d_cur - delta) should equal
    the prior's diameter. Without it the two heads can disagree, and a model
    whose own outputs are mutually contradictory is exactly what structured
    generation exists to prevent.
    """
    count_loss = nn.functional.cross_entropy(
        prediction["lesion_count_logits"], target["lesion_count"])

    b, max_lesions = prediction["diameters_mm"].shape
    slots = torch.arange(max_lesions, device=prediction["diameters_mm"].device)
    present = (slots.unsqueeze(0) < target["lesion_count"].unsqueeze(1)).float()
    n_present = present.sum().clamp(min=1.0)

    diameter_loss = ((prediction["diameters_mm"] - target["diameters_mm"]).abs()
                     * present).sum() / n_present

    # Change is only defined where a prior exists.
    has_prior = target["prior_present"].view(-1, 1)
    change_mask = present * has_prior
    n_change = change_mask.sum().clamp(min=1.0)
    change_loss = ((prediction["diameter_change_mm"]
                    - target["diameter_change_mm"]).abs()
                   * change_mask).sum() / n_change

    implied_prior = prediction["diameters_mm"] - prediction["diameter_change_mm"]
    consistency = ((implied_prior - target["prior_diameters_mm"]).abs()
                   * change_mask).sum() / n_change

    organ_loss = (nn.functional.cross_entropy(
        prediction["organ_logits"].reshape(b * max_lesions, -1),
        target["organs"].reshape(-1), reduction="none")
        .reshape(b, max_lesions) * present).sum() / n_present

    new_lesion_loss = nn.functional.binary_cross_entropy_with_logits(
        prediction["new_lesion_logit"], target["new_lesion"])

    # Millimetre terms are divided by 10 to sit on the same scale as the
    # cross-entropy terms. This weighting is a hyperparameter, not a truth.
    total = (count_loss
             + diameter_loss / 10.0
             + change_weight * change_loss / 10.0
             + consistency_weight * consistency / 10.0
             + organ_loss
             + new_lesion_loss)
    return {"loss": total, "count": count_loss,
            "diameter_mae_mm": diameter_loss,
            "change_mae_mm": change_loss,
            "consistency_mm": consistency,
            "organ": organ_loss, "new_lesion": new_lesion_loss}
