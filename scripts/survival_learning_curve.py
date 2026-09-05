
"""Train/validation learning curves for the survival timeline encoder.

`train_survival.py` reports only the best checkpoint. This prints the full
trajectory, which is where the overfitting is visible: train C-index reaches
0.99 while validation peaks near 0.71 (ADR-0014). It also served as the
reference implementation that identified a state-management bug in the
LightningModule, and is kept as an independent check on the training path.

    python scripts/survival_learning_curve.py
"""
import numpy as np
import torch

from panorama.data.manifest import read_manifest
from panorama.data.splits import build_timelines, patient_level_split
from panorama.data.synthetic import read_lesions
from panorama.survival.cox import concordance_index, cox_partial_likelihood_loss
from panorama.survival.dataset import TimelineCohort
from panorama.survival.embeddings import load_embeddings
from panorama.survival.synthetic import simulate_outcomes
from panorama.survival.timeline import TimelineEncoder
from panorama.utils.reproducibility import seed_everything

seed_everything(1337)

studies = read_manifest("data/synthetic/manifests/cohort.csv", "data/synthetic/raw")
lesions = read_lesions("data/synthetic/manifests/lesions.csv")
emb = load_embeddings("data/synthetic/manifests/embeddings.npz")
outcomes, truth = simulate_outcomes(build_timelines(studies), lesions, seed=1337)
split = patient_level_split(studies, val_fraction=0.30, test_fraction=0.0, seed=1337)
train = TimelineCohort(build_timelines(split.train), emb, outcomes)
val = TimelineCohort(build_timelines(split.val), emb, outcomes)


def run(baseline_only: bool, steps: int = 600, lr: float = 1e-3):
    seed_everything(1337)
    model = TimelineEncoder(embed_dim=train.embed_dim, hidden_dim=64,
                            depth=2, num_heads=4, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def forward(cohort):
        if baseline_only:
            return model(cohort.embeddings[:, :1],
                         torch.zeros_like(cohort.days[:, :1]),
                         torch.ones_like(cohort.mask[:, :1]))
        return model(cohort.embeddings, cohort.days, cohort.mask)

    history = []
    for step in range(steps):
        model.train()
        opt.zero_grad()
        loss = cox_partial_likelihood_loss(forward(train), train.duration, train.event)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                tr = concordance_index(forward(train).numpy(),
                                       train.duration.numpy(), train.event.numpy())
                va = concordance_index(forward(val).numpy(),
                                       val.duration.numpy(), val.event.numpy())
            history.append((step, float(loss.detach()), tr["c_index"], va["c_index"]))
    return history


for label, flag in (("FULL TIMELINE", False), ("BASELINE ONLY", True)):
    print(f"\n{'='*56}\n{label}")
    print(f"  {'step':>5} {'loss':>8} {'train C':>9} {'val C':>8}")
    for step, loss, tr, va in run(flag):
        print(f"  {step:>5} {loss:>8.4f} {tr:>9.4f} {va:>8.4f}")

print(f"\n  oracle on this val split: 0.8168")
print(f"  ADR-0013 achievable ceiling: 0.659   burden-only: 0.612")
print("\n=== is the Lightning module computing the same thing? ===")
from panorama.train.survival_module import SurvivalModule
m = SurvivalModule(train, val, baseline_only=False)
with torch.no_grad():
    direct = m.risk(val)
print(f"  module.risk(val) sd {float(direct.std()):.4f}")
c = concordance_index(direct.numpy(), val.duration.numpy(), val.event.numpy())
print(f"  C-index at init: {c['c_index']:.4f}  pairs {c['comparable_pairs']}")
print(f"  (the training run reported 0.4667 CONSTANT -- if this differs from")
print(f"   0.4667, validation_step was not using the trained weights)")