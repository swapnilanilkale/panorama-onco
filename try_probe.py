from pathlib import Path

from panorama.core.constants import RECISTResponse
from panorama.eval.probe import (
    extract_features, extract_study_features, probe_report, regression_report,
)
from panorama.train.align_module import AlignmentModule
from panorama.vlm.datamodule import AlignmentDataModule

# patches_per_study=4 in TRAIN gives multi-crop pooling something to pool.
dm = AlignmentDataModule(
    manifest_path="data/synthetic/manifests/cohort.csv",
    data_root="data/synthetic/raw",
    corpus_path="data/synthetic/manifests/reports.jsonl",
    tokenizer_path="data/synthetic/manifests/tokenizer.json",
    lesions_path="data/synthetic/manifests/lesions.csv",
    crop_size=(32, 32, 32), target_spacing=(2.0, 2.0, 2.0),
    batch_size=16, num_workers=0, patches_per_study=4,
    fg_threshold=0.3, val_fraction=0.2, test_fraction=0.1, seed=1337)
dm.prepare_data(); dm.setup("fit")
# The datamodule hard-codes 1 crop/study for val (right for retrieval, wrong
# for pooled probing). Match the train setting so both sides pool identically.
dm._datasets["val"].patches.patches_per_study = 4

runs = [r for r in sorted(Path("outputs/align").glob("*/"),
                          key=lambda p: p.stat().st_mtime)
        if (r / "checkpoints" / "last.ckpt").is_file()][-2:]
if len(runs) < 2:
    raise SystemExit(f"need 2 completed runs, found {len(runs)}")

for run in runs:
    cfg = (run / "config.yaml").read_text(encoding="utf-8")
    label = "PRETRAINED" if "checkpoints" in cfg else "SCRATCH"
    module = AlignmentModule.load_from_checkpoint(
        run / "checkpoints" / "last.ckpt", map_location="cpu")
    print(f"\n{'='*70}\n{label}")

    # --- FIX 2: a target the crop itself determines -------------------------
    tr = extract_features(module.vision, dm.train_dataloader())
    te = extract_features(module.vision, dm.val_dataloader())
    tr_f, tr_y, tr_sld, tr_n, tr_loc, tr_view, _ = tr
    te_f, te_y, te_sld, te_n, te_loc, te_view, _ = te

    print("\n  -- CROP-LOCAL: nearest-lesion diameter (in view only) --")
    m_tr, m_te = tr_view > 0.5, te_view > 0.5
    print(f"    crops containing a lesion: train {m_tr.mean():.1%},"
          f" val {m_te.mean():.1%}")
    if m_tr.sum() > 20 and m_te.sum() > 5:
        rr = regression_report(tr_f[m_tr], tr_loc[m_tr], te_f[m_te], te_loc[m_te])
        print(f"    lesion diameter (mm) R^2 {rr['r2']:>7.4f}   MAE {rr['mae']:>6.2f}"
              f"   (target std {rr['target_std']:.2f}, n={rr['n_test']})")
    else:
        print("    too few in-view crops to fit")

    # --- FIX 1: study-level targets from pooled crops -----------------------
    print("\n  -- STUDY-LEVEL: sum-pooled over crops --")
    s_tr = extract_study_features(module.vision, dm.train_dataloader(), pooling="mean")
    s_te = extract_study_features(module.vision, dm.val_dataloader(), pooling="mean")
    sf_tr, sy_tr, ssld_tr, sn_tr, _, sc_tr = s_tr
    sf_te, sy_te, ssld_te, sn_te, _, sc_te = s_te
    print(f"    crops per study: train {sc_tr.mean():.1f}, val {sc_te.mean():.1f}")
    for name, a, b in (("tumour burden (mm)", ssld_tr, ssld_te),
                       ("lesion count", sn_tr, sn_te)):
        rr = regression_report(sf_tr, a, sf_te, b)
        print(f"    {name:20} R^2 {rr['r2']:>7.4f}   MAE {rr['mae']:>6.2f}"
              f"   (target std {rr['target_std']:.2f})")

    # --- baseline: single-crop, for comparison ------------------------------
    print("\n  -- SINGLE-CROP baseline (previous result) --")
    for name, a, b in (("tumour burden (mm)", tr_sld, te_sld),
                       ("lesion count", tr_n, te_n)):
        rr = regression_report(tr_f, a, te_f, b)
        print(f"    {name:20} R^2 {rr['r2']:>7.4f}")