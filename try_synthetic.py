import shutil
from pathlib import Path

import numpy as np

from panorama.core.constants import Modality
from panorama.data.manifest import scan_directory
from panorama.data.synthetic import GEOMETRY, synth_volume, write_cohort

print("=== single volumes ===")
rng = np.random.default_rng(0)
lesions = [(30., 40., 50.), (70., 60., 30.)]
radii = [8., 6.]
for mod in Modality.imaging_streams():
    vol, aff = synth_volume(mod, lesions, radii, np.random.default_rng(0))
    spacing = tuple(round(float(np.linalg.norm(aff[:3, i])), 1) for i in range(3))
    extent = tuple(round(s * n) for s, n in zip(spacing, vol.shape))
    print(f"  {mod.value:3} shape={str(vol.shape):16} spacing={str(spacing):18} "
          f"extent={str(extent):16} range=[{vol.min():8.1f},{vol.max():7.1f}]")

print("\n=== the planted correspondence ===")
ct, _ = synth_volume(Modality.CT, lesions, radii, np.random.default_rng(0))
pet, _ = synth_volume(Modality.PET, lesions, radii, np.random.default_rng(0))
for name, pt in [("lesion 1", lesions[0]), ("lesion 2", lesions[1]),
                 ("background", (15., 15., 80.))]:
    ci = tuple(int(round(v / 1.0)) for v in pt)
    pi = tuple(int(round(v / 4.0)) for v in pt)
    print(f"  {name:11} CT={ct[ci]:8.2f}  PET={pet[pi]:6.2f}")

print("\n=== a whole cohort on disk ===")
root = Path("_tmp_synth")
write_cohort(root, n_patients=12, max_studies=3, seed=0)
studies = scan_directory(root)
print(f"  {len(studies)} studies, {len({s.patient_id for s in studies})} patients")

from collections import Counter
patterns = Counter(tuple(m.value for m in s.present) for s in studies)
print("  modality patterns:")
for pattern, n in sorted(patterns.items()):
    print(f"    {str(pattern):28} x{n}")

print("\n=== lesions grow across a patient's timeline ===")
from panorama.data.splits import build_timelines
tl = max(build_timelines(studies), key=len)
print(f"  {tl.patient_id}: {len(tl)} timepoints at days "
      f"{[tl.days_since_baseline(s) for s in tl.studies]}")

shutil.rmtree(root)