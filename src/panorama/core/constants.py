from __future__ import annotations
from enum import Enum


class Modality(str, Enum):
    CT = "CT"
    MRI = "MRI"
    PET = "PET"
    SEG = "SEG"  # a label/mask volume, not an image

    @classmethod
    def imaging_streams(cls) -> tuple["Modality", ...]:
        return (cls.CT, cls.MRI, cls.PET)


class RECISTResponse(str, Enum):
    CR = "complete_response"     # tumor gone
    PR = "partial_response"      # shrank >= 30%
    SD = "stable_disease"        # in between
    PD = "progressive_disease"   # grew >= 20% or a new lesion
    NE = "not_evaluable"
# RECIST 1.1 numeric thresholds, evaluated on the sum of longest diameters.
RECIST_PR_SHRINK_FRACTION = 0.30   # >= 30% decrease from BASELINE -> Partial Response
RECIST_PD_GROWTH_FRACTION = 0.20   # >= 20% increase from NADIR    -> Progressive Disease
RECIST_PD_ABSOLUTE_MM = 5.0        # ...and must ALSO be >= 5 mm absolute


class IntensityNorm:
    CT_HU_WINDOW = (-1000.0, 400.0)     # CT: calibrated Hounsfield Units
    MRI_CLIP_PERCENTILES = (0.5, 99.5)  # MRI: no fixed units -> z-score
    PET_SUV_CLIP = (0.0, 25.0)          # PET -> SUV (metabolic activity)
    PET_SUV_SUSPICIOUS = 2.5            # common reference for hypermetabolism

