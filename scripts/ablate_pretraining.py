"""Does MAE pretraining produce better frozen features than random init?

The direct test of Aim 1's representation claim. Both encoders are
architecturally identical -- only the weights differ -- and both are frozen.
A linear probe on a target the crop determines is the measurement.

    python scripts/ablate_pretraining.py --checkpoint outputs/qin/<run>/checkpoints/last.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from panorama.core.logging import configure_logging, get_logger
from panorama.data.datamodule import PanoramaDataModule
from panorama.eval.probe import regression_report, standardize
from panorama.train.mae_module import MAEPretrainModule
from panorama.utils.reproducibility import seed_everything
from panorama.vision.encoder import MultiStreamViT

log = get_logger(__name__)


@torch.no_grad()
def pooled_features(encoder, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frozen embeddings plus two probe targets.

    `ct_max_suv`  -- the peak SUV in this crop, predicted from CT ALONE. This is
                     Aim 1's claim in probe form: do structural features carry
                     metabolic information? A global statistic like the input
                     mean is recoverable by any linear map and therefore useless
                     as a discriminator; a spatial maximum is not.
    `pet_max_suv` -- the same target with PET visible, as an upper bound.
    """
    encoder = encoder.eval()
    ct_feats, both_feats, targets = [], [], []
    for batch in loader:
        image, mask = batch["image"], batch["modality_mask"]

        # CT only: zero the PET channel AND clear its presence bit, so the
        # encoder substitutes the missing-modality token rather than reading
        # a channel of zeros.
        ct_image = image.clone()
        ct_image[:, 2] = 0.0
        ct_mask = mask.clone()
        ct_mask[:, 2] = 0.0

        _, ct_pooled = encoder(ct_image, ct_mask)
        _, both_pooled = encoder(image, mask)
        ct_feats.append(ct_pooled.numpy())
        both_feats.append(both_pooled.numpy())
        # Peak normalised PET in the crop (0-1 after the SUV clip at 10).
        targets.append(image[:, 2].amax(dim=(1, 2, 3)).numpy())
    return (np.concatenate(ct_feats), np.concatenate(both_feats),
            np.concatenate(targets))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/tcia/manifests/qin-breast.csv"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/tcia/qin-breast-nifti"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    seed_everything(args.seed)

    dm = PanoramaDataModule(
        manifest_path=args.manifest, data_root=args.data_root,
        crop_size=(32, 32, 32), target_spacing=(2.0, 2.0, 2.0),
        batch_size=8, num_workers=0, patches_per_study=4,
        fg_threshold=0.5, val_fraction=0.2, test_fraction=0.1, seed=args.seed)
    dm.prepare_data()
    dm.setup("fit")
        # More val samples: 21 studies x 1 crop gives an unstable R^2.
    dm._datasets["val"].patches_per_study = 4

    mae = MAEPretrainModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    hp = mae.hparams
    encoders = {
        "PRETRAINED": mae.model.encoder,
        # Architecturally identical, random weights. Same seed for both so the
        # comparison isolates pretraining, not initialisation luck.
        "SCRATCH": MultiStreamViT(
            volume_shape=tuple(hp.volume_shape), patch_size=hp.patch_size,
            embed_dim=hp.embed_dim, depth=hp.depth, num_heads=hp.num_heads,
            fusion_every=hp.fusion_every,
            share_stream_weights=hp.share_stream_weights),
    }


    for name, encoder in encoders.items():
        encoder.requires_grad_(False)
        tr_ct, tr_both, tr_y = pooled_features(encoder, dm.train_dataloader())
        te_ct, te_both, te_y = pooled_features(encoder, dm.val_dataloader())

        print(f"\n{'='*62}\n{name}")
        print(f"  target: peak PET (normalised SUV), std {tr_y.std():.4f}, "
              f"n={len(tr_y)} train / {len(te_y)} val")
        for label, tr_f, te_f in (("from CT alone   ", tr_ct, te_ct),
                                  ("from CT+PET     ", tr_both, te_both)):
            r = regression_report(tr_f, tr_y, te_f, te_y)
            print(f"    {label} R^2 {r['r2']:>8.4f}   MAE {r['mae']:>7.4f}")
        std = standardize(tr_both)[0]
        sv = np.linalg.svd(std, compute_uv=False) ** 2
        rank = int((np.cumsum(sv) / sv.sum() < 0.95).sum()) + 1
        print(f"    effective rank {rank} of {tr_both.shape[1]}")


if __name__ == "__main__":
    main()