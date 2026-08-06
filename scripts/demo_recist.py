from panorama.clinical.recist import (
    Lesion, TimepointAssessment, assess_course, classify, first_progression,
)
from panorama.core.constants import RECISTResponse

print("=== a responder who later progresses ===")
course = [
    TimepointAssessment("S0", [Lesion("L1", 60.0, "lung"), Lesion("L2", 40.0, "liver")]),
    TimepointAssessment("S1", [Lesion("L1", 38.0, "lung"), Lesion("L2", 24.0, "liver")]),
    TimepointAssessment("S2", [Lesion("L1", 33.0, "lung"), Lesion("L2", 22.0, "liver")]),
    TimepointAssessment("S3", [Lesion("L1", 42.0, "lung"), Lesion("L2", 26.0, "liver")]),
    TimepointAssessment("S4", [Lesion("L1", 56.0, "lung"), Lesion("L2", 34.0, "liver")]),
]
assess_course(course)
print(f"{'visit':6} {'SLD':>7} {'response':>22}  rationale")
for tp in course:
    print(f"{tp.study_id:6} {tp.sld_mm:>6.0f}mm {tp.response.value:>22}  {tp.rationale}")

idx = first_progression(course)
print(f"\nfirst progression: {course[idx].study_id} (index {idx}) -- this ends PFS")
print(f"SLD there is {course[idx].sld_mm:.0f}mm vs a {course[0].sld_mm:.0f}mm baseline "
      f"({(course[idx].sld_mm - course[0].sld_mm)/course[0].sld_mm:+.0%})")
print("-> below baseline, yet correctly PD, because it is measured from the nadir")

print("\n=== complete response ===")
cr = [TimepointAssessment("S0", [Lesion("L1", 30.0)]),
      TimepointAssessment("S1", [Lesion("L1", 0.0)])]
assess_course(cr)
print(f"  {cr[1].response.value}: {cr[1].rationale}")

print("\n=== a new lesion is PD regardless of the sum ===")
resp, why = classify(sld_mm=10.0, baseline_mm=100.0, nadir_mm=10.0, new_lesion=True)
print(f"  sum shrank 90% but a new lesion appeared -> {resp.value}: {why}")

print("\n=== the 5mm floor ===")
for nadir, sld in ((10.0, 12.5), (10.0, 16.0), (100.0, 125.0)):
    resp, why = classify(sld, baseline_mm=nadir, nadir_mm=nadir)
    print(f"  {nadir:>5.0f} -> {sld:>5.1f}mm ({(sld-nadir)/nadir:+.0%}, "
          f"{sld-nadir:+.1f}mm) -> {resp.value}")

print("\n=== stable disease sits between the thresholds ===")
for sld in (75.0, 100.0, 115.0):
    resp, why = classify(sld, baseline_mm=100.0, nadir_mm=100.0)
    print(f"  {sld:>5.0f}mm vs 100mm baseline -> {resp.value:>20}")