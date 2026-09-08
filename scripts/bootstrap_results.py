"""Confidence intervals for the headline results in docs/RESULTS.md section 3.

Seven results were single-run point estimates. A methods paper arguing for
rigour cannot report those, and several null claims ("pretrained ties random")
were assertions rather than measurements until now.

    python scripts/bootstrap_results.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from panorama.core.logging import configure_logging, get_logger
from panorama.data.manifest import read_manifest
from panorama.data.splits import build_timelines, patient_level_split
from panorama.data.synthetic import read_lesions
from panorama.eval.bootstrap import format_ci, paired_bootstrap_ci
from panorama.survival.cox import concordance_index
from panorama.survival.dataset import TimelineCohort
from panorama.survival.embeddings import load_embeddings
from panorama.survival.synthetic import simulate_outcomes
from panorama.survival.train import forward, train
from panorama.utils.reproducibility import git_revision, seed_everything

log = get_logger(__name__)


def aim3_timeline_vs_baseline(seed: int) -> dict:
    """Aim 3: does the timeline beat a single scan? (ADR-0014)"""
    studies = read_manifest("data/synthetic/manifests/cohort.csv",
                            "data/synthetic/raw")
    lesions = read_lesions("data/synthetic/manifests/lesions.csv")
    embeddings = load_embeddings("data/synthetic/manifests/embeddings.npz")
    outcomes, _ = simulate_outcomes(build_timelines(studies), lesions, seed=seed)
    split = patient_level_split(studies, val_fraction=0.30, test_fraction=0.0,
                                seed=seed)
    train_cohort = TimelineCohort(build_timelines(split.train), embeddings, outcomes)
    val = TimelineCohort(build_timelines(split.val), embeddings, outcomes)

    models = {}
    for arm in (False, True):
        result = train(train_cohort, val, baseline_only=arm, seed=seed)
        models[arm] = result["model"].eval()

    duration = val.duration.numpy()
    event = val.event.numpy()
    with torch.no_grad():
        risk = {arm: forward(m, val, arm).numpy() for arm, m in models.items()}

    def scorer(arm):
        def metric(index: np.ndarray) -> float:
            # Resample PATIENTS. Resampling the 1,201 comparable pairs instead
            # would treat correlated pairs as independent and give an interval
            # ~4.5x too narrow.
            result = concordance_index(risk[arm][index], duration[index],
                                       event[index])
            return result["c_index"]
        return metric

    return paired_bootstrap_ci(scorer(False), scorer(True), len(val), seed=seed)


def aim1_probe_comparison(checkpoint: Path, seed: int, n_boot: int = 2000) -> dict:
    """Aim 1: does MAE pretraining beat random init on a crop-local probe?

    This underpins ADR-0007 and ADR-0009's null findings, which until now were
    point estimates without intervals -- an assertion rather than a measurement.
    """
    import torch
    from panorama.data.datamodule import PanoramaDataModule
    from panorama.eval.probe import extract_features, fit_ridge_regression, standardize
    from panorama.train.mae_module import MAEPretrainModule
    from panorama.vision.encoder import MultiStreamViT

    mae = MAEPretrainModule.load_from_checkpoint(checkpoint, map_location="cpu")
    hp = mae.hparams
    encoders = {
        "pretrained": mae.model.encoder,
        # Architecturally identical, random weights. The comparison isolates
        # pretraining, not capacity.
        "scratch": MultiStreamViT(
            volume_shape=tuple(hp.volume_shape), patch_size=hp.patch_size,
            embed_dim=hp.embed_dim, depth=hp.depth, num_heads=hp.num_heads,
            fusion_every=hp.fusion_every,
            share_stream_weights=hp.share_stream_weights),
    }

    dm = PanoramaDataModule(
        manifest_path="data/tcia/manifests/qin-breast.csv",
        data_root="data/tcia/qin-breast-nifti",
        precomputed_root="data/tcia/qin-breast-preprocessed",
        crop_size=(32, 32, 32), target_spacing=(2.0, 2.0, 2.0),
        batch_size=8, num_workers=0, patches_per_study=4,
        fg_threshold=0.5, val_fraction=0.2, test_fraction=0.1, seed=seed)
    dm.prepare_data()
    dm.setup("fit")

    # Extract once per encoder; the bootstrap resamples indices, not features.
    data = {}
    for name, encoder in encoders.items():
        encoder.requires_grad_(False)
        train_rows = _peak_pet_rows(encoder, dm.train_dataloader())
        val_rows = _peak_pet_rows(encoder, dm.val_dataloader())
        data[name] = (train_rows, val_rows)

    # Patient identity per row, so resampling is at the PATIENT level: crops
    # from one patient are correlated, and resampling crops would give an
    # interval ~2.9x too narrow.
    train_patients = np.array(data["pretrained"][0][2])
    val_patients = np.array(data["pretrained"][1][2])
    unique_train = sorted(set(train_patients))
    unique_val = sorted(set(val_patients))

    def scorer(name):
        (tr_f, tr_y, _), (te_f, te_y, _) = data[name]

        def metric(index: np.ndarray) -> float:
            # `index` selects PATIENTS; expand to their rows.
            chosen_train = np.concatenate(
                [np.flatnonzero(train_patients == unique_train[i % len(unique_train)])
                 for i in index]) if len(index) else np.arange(len(tr_y))
            chosen_val = np.concatenate(
                [np.flatnonzero(val_patients == unique_val[i % len(unique_val)])
                 for i in index]) if len(index) else np.arange(len(te_y))

            # Refit the probe on the resampled training rows: the probe is
            # itself fitted, so holding it fixed understates uncertainty by
            # about a third.
            tr_s, te_s = standardize(tr_f[chosen_train], te_f[chosen_val])
            weights = fit_ridge_regression(tr_s, tr_y[chosen_train])
            pred = np.c_[te_s, np.ones(len(te_s))] @ weights
            y = te_y[chosen_val]
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - tr_y[chosen_train].mean()) ** 2).sum())
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return metric

    return paired_bootstrap_ci(scorer("pretrained"), scorer("scratch"),
                               n_samples=min(len(unique_train), len(unique_val)),
                               n_boot=n_boot, seed=seed)


@torch.no_grad()
def _peak_pet_rows(encoder, loader):
    """Pooled CT-only embeddings, peak PET target, and patient ids per row.

    Predicting peak PET from CT alone is Aim 1's claim in probe form: do
    structural features carry metabolic information?
    """
    encoder.eval()
    feats, targets, patients = [], [], []
    for batch in loader:
        image, mask = batch["image"], batch["modality_mask"]
        ct_image, ct_mask = image.clone(), mask.clone()
        ct_image[:, 2] = 0.0        # zero PET pixels AND clear its presence bit,
        ct_mask[:, 2] = 0.0         # so the missing-modality token is substituted
        _, pooled = encoder(ct_image, ct_mask)
        feats.append(pooled.numpy())
        targets.append(image[:, 2].amax(dim=(1, 2, 3)).numpy())
        patients.extend(batch["patient_id"])
    return np.concatenate(feats), np.concatenate(targets), patients



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/bootstrap"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    seed_everything(args.seed)

    results = {}
    log.info("Aim 3: timeline vs single-scan control")
    results["aim3_timeline_vs_baseline"] = aim3_timeline_vs_baseline(args.seed)

    print(f"\n{'='*72}")
    print("  bootstrap confidence intervals (patient-level resampling)\n")
    for name, result in results.items():
        print(f"  {name}")
        print(f"    {format_ci(result)}\n")

    checkpoint = max(Path("outputs/qin").glob("*/checkpoints/last.ckpt"),
                     key=lambda p: p.stat().st_mtime, default=None)
    if checkpoint:
        log.info("Aim 1: pretrained vs scratch, peak-PET probe (%s)",
                 checkpoint.parents[1].name)
        results["aim1_peak_pet_pretrained_vs_scratch"] = aim1_probe_comparison(
            checkpoint, args.seed)
    else:
        log.warning("no pretraining checkpoint found -- skipping Aim 1")


    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps({"git_revision": git_revision(), "seed": args.seed,
                    "results": results}, indent=2), encoding="utf-8")
    log.info("saved to %s", out_dir)



if __name__ == "__main__":
    main()