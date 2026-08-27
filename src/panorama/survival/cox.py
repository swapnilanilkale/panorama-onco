"""Cox proportional hazards loss and concordance, in PyTorch.

The partial likelihood compares each patient who had an event against everyone
still at risk at that moment:

    log L = sum over events of [ r_i - logsumexp(r_j for j at risk at t_i) ]

Censored patients never appear in a numerator but DO appear in the denominator
of every event preceding their censoring -- their information is used without
assuming an outcome for them.
"""
from __future__ import annotations

import numpy as np
import torch

from panorama.core.logging import get_logger

log = get_logger(__name__)


def cox_partial_likelihood_loss(risk: torch.Tensor,
                                duration: torch.Tensor,
                                event: torch.Tensor) -> torch.Tensor:
    """Negative log partial likelihood, averaged over observed events.

    Sorting by DESCENDING duration makes each risk set a prefix of the sorted
    array, so the denominators are a cumulative log-sum-exp -- one pass rather
    than a loop over event times.

    Ties at the same duration are handled by Breslow's approximation, which
    falls out of the prefix structure: patients failing at the same recorded
    time share a risk set. Monthly follow-up produces many ties, so this is not
    a corner case.
    """
    event = event.bool()
    if not event.any():
        # No events means no partial likelihood. Returning zero keeps training
        # alive on an unlucky batch, but a whole epoch of this means the cohort
        # cannot support the model.
        return risk.sum() * 0.0

    order = torch.argsort(duration, descending=True)
    risk_sorted = risk[order]
    event_sorted = event[order]

    # cumulative logsumexp over the prefix = the risk set at each time.
    # logcumsumexp is numerically stable; a naive cumsum of exp overflows once
    # risk scores exceed ~700, turning the loss silently into nan.
    log_risk_set = torch.logcumsumexp(risk_sorted, dim=0)

    contributions = (risk_sorted - log_risk_set)[event_sorted]
    return -contributions.mean()


@torch.no_grad()
def concordance_index(risk: np.ndarray | torch.Tensor,
                      duration: np.ndarray | torch.Tensor,
                      event: np.ndarray | torch.Tensor) -> dict:
    """Harrell's C-index: the fraction of COMPARABLE pairs ranked correctly.

    A pair is comparable only when we know which patient failed first. Two
    censored patients are never comparable; a censored patient is comparable
    only to those who had an event before the censoring time. Roughly a third
    of all pairs are typically incomparable, which is why accuracy is undefined
    here and concordance is the standard metric.

    0.5 is random ranking; 1.0 is perfect. Higher risk should mean SHORTER
    survival, so a model predicting risk is concordant when risk_i > risk_j
    whenever patient i fails first.
    """
    risk = np.asarray(risk, dtype=float).ravel()
    duration = np.asarray(duration, dtype=float).ravel()
    event = np.asarray(event, dtype=bool).ravel()

    concordant = tied = comparable = 0
    n = len(risk)
    for i in range(n):
        if not event[i]:
            continue                      # only an observed failure anchors a pair
        # Everyone who outlived patient i is comparable, whatever their status.
        others = duration > duration[i]
        if not others.any():
            continue
        comparable += int(others.sum())
        concordant += int((risk[others] < risk[i]).sum())
        tied += int((risk[others] == risk[i]).sum())

    if comparable == 0:
        return {"c_index": float("nan"), "comparable_pairs": 0,
                "concordant": 0, "tied": 0}

    # Ties in risk count as half-concordant, per Harrell's definition.
    c_index = (concordant + 0.5 * tied) / comparable
    return {"c_index": float(c_index), "comparable_pairs": comparable,
            "concordant": concordant, "tied": tied}


def baseline_hazard(risk: np.ndarray, duration: np.ndarray,
                    event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Breslow estimator of the cumulative baseline hazard.

    Cox gives relative risk only -- it says patient A is twice as likely to fail
    as patient B, not when either will fail. Turning risk scores into an actual
    survival CURVE, which is what Aim 3 asks for, needs the baseline hazard
    estimated separately after fitting.

    Returns (event_times, cumulative_baseline_hazard).
    """
    order = np.argsort(duration)
    duration, event, risk = duration[order], event[order], risk[order]
    exp_risk = np.exp(risk - risk.max())      # shift for stability

    times, increments = [], []
    for i in range(len(duration)):
        if not event[i]:
            continue
        at_risk = duration >= duration[i]
        denominator = exp_risk[at_risk].sum()
        if denominator > 0:
            times.append(duration[i])
            increments.append(1.0 / denominator)

    return np.asarray(times), np.cumsum(increments) * np.exp(-risk.max())


def survival_curve(risk_score: float, times: np.ndarray,
                   cumulative_hazard: np.ndarray) -> np.ndarray:
    """S(t | x) = exp(-H0(t) * exp(risk)) -- the proportional hazards form."""
    return np.exp(-cumulative_hazard * np.exp(risk_score))