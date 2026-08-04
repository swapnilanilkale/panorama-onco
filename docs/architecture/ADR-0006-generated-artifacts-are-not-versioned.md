# ADR-0006: Generated data and run artifacts are not versioned

- **Status:** Accepted
- **Date:** 2026-08-03

## Context
Synthetic cohorts and training outputs were committed, growing the repository
to 2.38 GiB and making `git push` fail. NIfTI volumes and checkpoints are
already compressed, so git's delta compression achieves nothing and every
regeneration adds a full copy permanently.

## Decision
`data/` and `outputs/` are in `.gitignore`. What is versioned instead is the
*recipe*: `scripts/build_synthetic_cohort.py --seed N` regenerates an identical
cohort, and each run directory's `config.yaml` (with its git revision) records
what produced a given result.

For real TCIA data the same rule applies with added force: the manifest CSV is
versioned, the volumes are not -- which is also the correct posture for patient
data governance.

## Consequences
- Repository stays small and cloneable.
- Reproducing a result means rerunning a script, not downloading a dataset.
- History was rewritten with `git filter-repo` on 2026-08-03 to remove
  previously committed artifacts.