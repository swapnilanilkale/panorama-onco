"""Bootstrap confidence intervals for evaluation metrics.

Two rules that are easy to get wrong.

RESAMPLE THE INDEPENDENT UNIT. The C-index is computed over pairs, but pairs
from the same patient are not independent. Resampling 1,201 pairs instead of 60
patients gives an interval 4.5x too narrow.

BOOTSTRAP THE DIFFERENCE, NOT TWO INTERVALS. Two models scored on the same
samples have correlated errors. Comparing marginal intervals and concluding "they
overlap, so no difference" ignores that correlation; resample the samples ONCE
per replicate and recompute both metrics on the same resample.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_ci(metric: Callable[[np.ndarray], float],
                 n_samples: int,
                 n_boot: int = 4000,
                 alpha: float = 0.05,
                 seed: int = 1337) -> dict:
    """Percentile CI for a metric computed on a resample of sample INDICES.

    `metric` receives an index array and returns a scalar, so the caller
    controls what the independent unit is.
    """
    rng = np.random.default_rng(seed)
    point = float(metric(np.arange(n_samples)))
    replicates = []
    for _ in range(n_boot):
        index = rng.integers(0, n_samples, n_samples)
        value = metric(index)
        if value is not None and np.isfinite(value):
            replicates.append(float(value))

    replicates = np.asarray(replicates)
    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "sd": float(replicates.std()), "n_boot": len(replicates),
            "n_samples": n_samples}


def paired_bootstrap_ci(metric_a: Callable[[np.ndarray], float],
                        metric_b: Callable[[np.ndarray], float],
                        n_samples: int,
                        n_boot: int = 4000,
                        alpha: float = 0.05,
                        seed: int = 1337) -> dict:
    """CI for (metric_a - metric_b), resampling the SAME indices for both.

    A difference interval containing zero is the measurement that licenses a
    null claim. Reporting two point estimates and asserting they are "the same"
    is not.
    """
    rng = np.random.default_rng(seed)
    full = np.arange(n_samples)
    point = float(metric_a(full)) - float(metric_b(full))

    differences = []
    for _ in range(n_boot):
        index = rng.integers(0, n_samples, n_samples)
        a, b = metric_a(index), metric_b(index)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            differences.append(float(a) - float(b))

    differences = np.asarray(differences)
    lo, hi = np.percentile(differences, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p: how often the resampled difference crosses zero.
    p = 2 * min(float((differences <= 0).mean()), float((differences >= 0).mean()))
    return {"difference": point, "lo": float(lo), "hi": float(hi),
            "sd": float(differences.std()), "p": min(1.0, p),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": len(differences)}


def format_ci(result: dict, digits: int = 4) -> str:
    if "difference" in result:
        verdict = "excludes 0" if result["excludes_zero"] else "INCLUDES 0"
        return (f"{result['difference']:+.{digits}f} "
                f"[{result['lo']:+.{digits}f}, {result['hi']:+.{digits}f}] "
                f"p={result['p']:.4f}  {verdict}")
    return (f"{result['point']:.{digits}f} "
            f"[{result['lo']:.{digits}f}, {result['hi']:.{digits}f}]")