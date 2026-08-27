"""Cox partial likelihood and concordance guards."""
import numpy as np
import pytest
import torch

from panorama.survival.cox import (
    baseline_hazard, concordance_index, cox_partial_likelihood_loss, survival_curve,
)
from panorama.survival.data import simulate_cohort


@pytest.fixture(scope="module")
def cohort():
    features, outcomes = simulate_cohort(n=1500, n_features=8,
                                         censoring_rate=0.45, seed=7)
    return (torch.tensor(features, dtype=torch.float32),
            torch.tensor([o.duration_days for o in outcomes], dtype=torch.float32),
            torch.tensor([o.event for o in outcomes]))


def fit(X, duration, event, steps=200):
    beta = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([beta], lr=0.5, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = cox_partial_likelihood_loss(X @ beta, duration, event)
        loss.backward()
        return loss

    opt.step(closure)
    return beta.detach()


def test_recovers_known_coefficients(cohort):
    """The decisive correctness test: simulated data has a KNOWN true hazard.

    A real cohort cannot test this, because nobody knows its true coefficients.
    """
    beta = fit(*cohort)
    true = torch.tensor([0.8, -0.5, 0.3])
    assert torch.allclose(beta[:3], true, atol=0.15)
    assert beta[3:].abs().max() < 0.15          # distractors stay near zero


def test_censored_patients_are_not_treated_as_events(cohort):
    """Flipping every censored flag to 'event' must change the fit.

    If it does not, censoring is being ignored -- the single most consequential
    error in survival modelling.
    """
    X, duration, event = cohort
    correct = fit(X, duration, event)
    all_events = fit(X, duration, torch.ones_like(event))
    assert not torch.allclose(correct, all_events, atol=0.05)


def test_loss_is_finite_at_extreme_risk():
    """Naive log(cumsum(exp(r))) overflows past r ~ 709 and yields nan."""
    duration = torch.arange(100, dtype=torch.float32)
    event = torch.ones(100, dtype=torch.bool)
    for value in (0.0, 100.0, 800.0):
        loss = cox_partial_likelihood_loss(torch.full((100,), value),
                                           duration, event)
        assert torch.isfinite(loss), value


def test_loss_survives_a_batch_with_no_events():
    """An unlucky batch of all-censored patients must not crash training."""
    loss = cox_partial_likelihood_loss(
        torch.randn(16), torch.rand(16) * 100, torch.zeros(16, dtype=torch.bool))
    assert torch.isfinite(loss)


def test_higher_risk_means_shorter_survival(cohort):
    """Sign convention: the loss must reward risk that ANTICIPATES failure."""
    X, duration, event = cohort
    beta = fit(X, duration, event)
    risk = (X @ beta).numpy()
    result = concordance_index(risk, duration.numpy(), event.numpy())
    assert result["c_index"] > 0.65
    # The inverted ranking must be correspondingly bad.
    inverted = concordance_index(-risk, duration.numpy(), event.numpy())
    assert inverted["c_index"] == pytest.approx(1 - result["c_index"], abs=0.02)


def test_random_risk_scores_half(cohort):
    _, duration, event = cohort
    rng = np.random.default_rng(0)
    result = concordance_index(rng.normal(size=len(duration)),
                               duration.numpy(), event.numpy())
    assert 0.45 < result["c_index"] < 0.55


def test_only_comparable_pairs_are_counted():
    """Two censored patients can never be ordered, so they are not a pair."""
    # Three patients: one event at t=1, two censored at t=2 and t=3.
    risk = np.array([0.0, 1.0, 2.0])
    duration = np.array([1.0, 2.0, 3.0])
    event = np.array([True, False, False])
    result = concordance_index(risk, duration, event)
    assert result["comparable_pairs"] == 2      # the event vs each censored
    # The two censored patients contribute no pair between themselves.


def test_no_comparable_pairs_returns_nan():
    result = concordance_index(np.zeros(4), np.arange(4.0),
                               np.zeros(4, dtype=bool))
    assert np.isnan(result["c_index"])


def test_survival_curves_separate_by_risk(cohort):
    X, duration, event = cohort
    beta = fit(X, duration, event)
    risk = (X @ beta).numpy()
    times, cumulative = baseline_hazard(risk, duration.numpy(), event.numpy())

    low = survival_curve(float(np.percentile(risk, 10)), times, cumulative)
    high = survival_curve(float(np.percentile(risk, 90)), times, cumulative)
    assert (low >= high).all()                  # higher risk, lower survival
    assert low[-1] > high[-1]
    # Survival is a probability and must not increase over time.
    assert (np.diff(low) <= 1e-9).all()