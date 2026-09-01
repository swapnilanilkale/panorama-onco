"""Cache study embeddings from a pretrained (or random) vision encoder.

    python scripts/cache_embeddings.py --checkpoint outputs/smoke/<run>/checkpoints/last.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from panorama.core.logging import configure_logging, get_logger
from panorama.data.manifest import read_manifest
from panorama.eval.probe import regression_report
from panorama.survival.embeddings import cache_study_embeddings, save_embeddings
from panorama.train.mae_module import MAEPretrainModule
from panorama.utils.reproducibility import seed_everything
from panorama.vision.encoder import MultiStreamViT

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="omit for a random encoder (the control arm)")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/synthetic/manifests/cohort.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic/raw"))
    parser.add_argument("--lesions", type=Path,
                        default=Path("data/synthetic/manifests/lesions.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/synthetic/manifests/embeddings.npz"))
    parser.add_argument("--crops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    configure_logging("INFO")
    seed_everything(args.seed)

    if args.checkpoint:
        module = MAEPretrainModule.load_from_checkpoint(args.checkpoint,
                                                        map_location="cpu")
        encoder, hp = module.model.encoder, module.hparams
        log.info("pretrained encoder, embed_dim %d", hp.embed_dim)
        volume_shape, patch_size = tuple(hp.volume_shape), hp.patch_size
        spacing = (2.0, 2.0, 2.0)
    else:
        volume_shape, patch_size = (32, 32, 32), 8
        spacing = (2.0, 2.0, 2.0)
        encoder = MultiStreamViT(volume_shape=volume_shape, patch_size=patch_size,
                                 embed_dim=128, depth=4, num_heads=8, fusion_every=2)
        log.warning("RANDOM encoder -- the control arm")

    studies = read_manifest(args.manifest, args.data_root)
    embeddings = cache_study_embeddings(
        encoder, studies, crops_per_study=args.crops,
        crop_size=volume_shape, target_spacing=spacing,
        fg_threshold=0.3, seed=args.seed)
    save_embeddings(embeddings, args.out)
    log.info("saved to %s", args.out)

    # PREREQUISITE CHECK. If the embeddings do not encode tumour burden, the
    # timeline encoder cannot recover growth from them, and a null result would
    # not distinguish "temporal modelling failed" from "there was nothing to
    # model". This must be answered BEFORE the experiment.
    from panorama.data.synthetic import read_lesions
    lesions = read_lesions(args.lesions)
    ids = [s for s in embeddings if s in lesions]
    features = np.stack([embeddings[s] for s in ids])
    burden = np.array([sum(l.longest_diameter_mm for l in lesions[s]) for s in ids])

    split = int(0.7 * len(ids))
    report = regression_report(features[:split], burden[:split],
                               features[split:], burden[split:])
    log.info("burden recovery from embeddings: R^2 %.4f, MAE %.1f mm "
             "(target sd %.1f mm)", report["r2"], report["mae"],
             report["target_std"])
    if report["r2"] < 0.2:
        log.warning("embeddings barely encode tumour burden -- a weak timeline "
                    "result would be uninterpretable")


if __name__ == "__main__":
    main()