import numpy as np
import pytest

from panorama.core.constants import Modality
from panorama.data.pipeline import preprocess, preprocess_study
from panorama.data.volume import MedicalVolume

TARGET = (1.5, 1.5, 1.5)


def make(spacing, shape, modality, fill=None, dtype=np.float32):
    arr = (np.full(shape, fill, dtype) if fill is not None
           else np.random.default_rng(0).random(shape).astype(dtype))
    return MedicalVolume(arr, np.diag([*spacing, 1.0]), modality)


def test_resample_preserves_physical_extent():
    """Shapes change; the size of the scanned body region must not."""
    v = make((0.7, 0.7, 3.0), (64, 64, 20), Modality.CT)
    before = v.extent_mm
    after = preprocess(v, TARGET).extent_mm
    assert np.allclose(before, after, atol=TARGET[0])


def test_affine_is_updated_not_just_the_array():
    """Regression guard: a stale affine makes every mm measurement wrong."""
    v = make((0.7, 0.7, 3.0), (64, 64, 20), Modality.CT)
    out = preprocess(v, TARGET)
    assert np.allclose(out.spacing_mm, TARGET, atol=1e-6)


def test_ct_normalization_is_scanner_invariant():
    """Identical tissue must map to an identical value regardless of context."""
    chest = make((1.5, 1.5, 1.5), (16, 16, 16), Modality.CT, fill=-1000, dtype=np.int16)
    abdo = make((1.5, 1.5, 1.5), (16, 16, 16), Modality.CT, fill=40, dtype=np.int16)
    chest.array[6:10, 6:10, 6:10] = 60
    abdo.array[2:14, 2:14, 2:14] = 60
    a = preprocess(chest, TARGET).array[8, 8, 8]
    b = preprocess(abdo, TARGET).array[8, 8, 8]
    assert a == pytest.approx(b, abs=1e-5)


def test_segmentation_labels_survive_intact():
    """No phantom labels may be invented at organ boundaries."""
    lab = np.zeros((20, 20, 20), np.uint8)
    lab[:10] = 1
    lab[10:] = 3
    seg = MedicalVolume(lab, np.diag([2., 2., 2., 1.]), Modality.SEG)
    out = preprocess(seg, (1.0, 1.0, 1.0))
    assert set(np.unique(out.array).tolist()) <= {0, 1, 3}


def test_outputs_are_float32():
    """float64 would silently double memory for every batch."""
    v = make((2.0, 2.0, 2.0), (16, 16, 16), Modality.MRI)
    assert preprocess(v, TARGET).array.dtype == np.float32


def test_study_with_missing_modality_is_handled():
    """A CT-only follow-up must preprocess without error."""
    study = {Modality.CT: make((0.7, 0.7, 3.0), (32, 32, 12), Modality.CT)}
    out = preprocess_study(study, TARGET)
    assert set(out) == {Modality.CT}
    assert np.allclose(out[Modality.CT].spacing_mm, TARGET, atol=1e-6)