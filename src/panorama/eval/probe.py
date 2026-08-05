"""Linear probing of frozen representations.

The standard protocol for evaluating self-supervised encoders (SimCLR, MAE,
DINO): freeze the encoder, fit a LINEAR classifier on its features, and report
balanced accuracy. A linear head can only succeed if the feature space has
already organised the classes into linearly separable regions -- which is what
"a good representation" means.
"""
from __future__ import annotations

import numpy as np
import torch

from panorama.core.constants import RECISTResponse
from panorama.core.logging import get_logger

log = get_logger(__name__)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall.

    Plain accuracy is misleading on imbalanced clinical labels: always
    predicting the majority class scores well while never detecting the
    category that matters. Balanced accuracy scores that strategy at chance.
    """
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum():
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def standardize(train: np.ndarray, *others: np.ndarray):
    """Fit mean/std on TRAIN only, apply everywhere."""
    mean, std = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-8
    return ((train - mean) / std, *[(o - mean) / std for o in others])


def fit_ridge_probe(features: np.ndarray, labels: np.ndarray, n_classes: int,
                    alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression onto one-hot targets.

    Deterministic and instant -- no optimiser, no learning rate, no seeds. An
    evaluation must not become a second experiment whose own tuning confounds
    the comparison.
    """
    x = np.concatenate([features, np.ones((len(features), 1))], axis=1)  # bias
    y = np.eye(n_classes)[labels]
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    x = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    return (x @ weights).argmax(axis=1)


def fit_ridge_regression(features: np.ndarray, targets: np.ndarray,
                         alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge for a continuous target (e.g. tumour burden in mm)."""
    x = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ targets)


def regression_report(train_features: np.ndarray, train_targets: np.ndarray,
                      test_features: np.ndarray, test_targets: np.ndarray,
                      alpha: float = 1.0) -> dict:
    """R^2 against the mean-predictor baseline.

    Unlike the RECIST category, these targets are FULLY determined by a single
    scan -- so a failure here is a failure of the representation, not of the
    task specification.
    """
    tr, te = standardize(train_features, test_features)
    weights = fit_ridge_regression(tr, train_targets, alpha)
    pred = np.concatenate([te, np.ones((len(te), 1))], axis=1) @ weights

    ss_res = float(((test_targets - pred) ** 2).sum())
    ss_tot = float(((test_targets - train_targets.mean()) ** 2).sum())
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
        "mae": float(np.abs(test_targets - pred).mean()),
        "target_std": float(test_targets.std()),
        "n_test": len(test_targets),
    }

@torch.no_grad()
def extract_features(encoder, loader, device: str = "cpu"):
    """Per-crop embeddings with every probe target.

    Returns (features, recist_labels, sld_mm, n_lesions,
             local_lesion_mm, local_in_view, study_ids).

    The last two are CROP-LOCAL: the diameter of the lesion nearest this crop's
    centre, and whether it falls inside the crop. Unlike the study-level sld_mm,
    those are determined by what the encoder actually sees.
    """
    encoder = encoder.to(device).eval()
    feats, labels, sld, n_les = [], [], [], []
    local, in_view, study_ids = [], [], []
    for batch in loader:
        _, pooled = encoder(batch["image"].to(device),
                            batch["modality_mask"].to(device))
        feats.append(pooled.cpu().numpy())
        labels.append(batch["response"].numpy())
        sld.append(batch["sld_mm"].numpy())
        n_les.append(batch["n_lesions"].numpy())
        local.append(batch["local_lesion_mm"].numpy())
        in_view.append(batch["local_in_view"].numpy())
        study_ids.extend(batch["study_id"])
    return (np.concatenate(feats), np.concatenate(labels),
            np.concatenate(sld), np.concatenate(n_les),
            np.concatenate(local), np.concatenate(in_view), study_ids)

@torch.no_grad()
def extract_study_features(encoder, loader, pooling: str = "sum",
                           device: str = "cpu"):
    """Pool multiple crops per study into one study-level feature.

    A single 64mm crop sees roughly one lesion, so it cannot determine a
    study-level property like total tumour burden. Summing the per-crop
    embeddings matches how the SLD is built (a sum over lesions); mean pooling
    discards that scale and only works when lesion count is near-constant.
    """
    if pooling not in ("sum", "mean"):
        raise ValueError(f"pooling must be 'sum' or 'mean', got {pooling!r}")

    encoder = encoder.to(device).eval()
    by_study: dict[str, list[np.ndarray]] = {}
    meta: dict[str, tuple] = {}

    for batch in loader:
        _, pooled = encoder(batch["image"].to(device),
                            batch["modality_mask"].to(device))
        pooled = pooled.cpu().numpy()
        for i, study_id in enumerate(batch["study_id"]):
            by_study.setdefault(study_id, []).append(pooled[i])
            meta[study_id] = (int(batch["response"][i]),
                              float(batch["sld_mm"][i]),
                              float(batch["n_lesions"][i]))

    ids = sorted(by_study)
    reduce = np.sum if pooling == "sum" else np.mean
    feats = np.stack([reduce(by_study[s], axis=0) for s in ids])
    labels = np.array([meta[s][0] for s in ids])
    sld = np.array([meta[s][1] for s in ids], dtype=np.float32)
    n_les = np.array([meta[s][2] for s in ids], dtype=np.float32)
    n_crops = np.array([len(by_study[s]) for s in ids])
    return feats, labels, sld, n_les, ids, n_crops


def probe_report(train_features: np.ndarray, train_labels: np.ndarray,
                 test_features: np.ndarray, test_labels: np.ndarray,
                 alpha: float = 1.0) -> dict:
    """Fit and evaluate, with the baselines that make the number readable."""
    n_classes = len(RECISTResponse)
    tr, te = standardize(train_features, test_features)
    weights = fit_ridge_probe(tr, train_labels, n_classes, alpha)
    pred = predict(weights, te)

    present = sorted(set(train_labels.tolist()) | set(test_labels.tolist()))
    majority = int(np.bincount(train_labels, minlength=n_classes).argmax())
    majority_pred = np.full_like(test_labels, majority)

    return {
        "balanced_accuracy": balanced_accuracy(test_labels, pred, n_classes),
        "accuracy": float((pred == test_labels).mean()),
        "majority_balanced_accuracy": balanced_accuracy(
            test_labels, majority_pred, n_classes),
        "majority_accuracy": float((majority_pred == test_labels).mean()),
        "chance_balanced_accuracy": 1.0 / len(present),
        "n_classes_present": len(present),
        "n_test": len(test_labels),
        "confusion": confusion(test_labels, pred, n_classes),
    }