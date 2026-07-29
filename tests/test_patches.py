import numpy as np

from panorama.core.constants import Modality
from panorama.data.patches import (
    crop_around_world_point,
    crop_with_pad,
    sample_center_voxel,
    sample_study_patch,
    voxel_to_world,
    world_to_voxel,
)
from panorama.data.volume import MedicalVolume


def lesion_volume(shape, spacing, centre_mm, radius_mm, modality):
    """A volume with a spherical 'lesion' at a known PHYSICAL location."""
    affine = np.diag([*spacing, 1.0])
    grids = np.meshgrid(*[np.arange(s) * sp for s, sp in zip(shape, spacing)], indexing="ij")
    dist = sum((g - c) ** 2 for g, c in zip(grids, centre_mm)) ** 0.5
    return MedicalVolume((dist <= radius_mm).astype(np.float32), affine, modality)


def test_world_voxel_roundtrip():
    """Coordinate conversion must be exactly invertible."""
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    affine[:3, 3] = [-100.0, -50.0, 20.0]          # non-zero origin
    voxel = np.array([12.0, 47.0, 3.0])
    back = world_to_voxel(affine, voxel_to_world(affine, voxel))
    assert np.allclose(back, voxel, atol=1e-9)


def test_patch_shape_is_always_exact():
    """The DataLoader stacks these -- shape may never vary, even at edges."""
    arr = np.ones((10, 10, 10), np.float32)
    for origin in [(0, 0, 0), (-5, -5, -5), (8, 8, 8), (99, 99, 99)]:
        assert crop_with_pad(arr, origin, (6, 6, 6)).shape == (6, 6, 6)


def test_streams_on_different_grids_capture_same_anatomy():
    """ADR-0003: alignment is by millimetres, never by voxel index."""
    centre = (60.0, 60.0, 60.0)
    ct = lesion_volume((120, 120, 120), (1.0, 1.0, 1.0), centre, 8.0, Modality.CT)
    pet = lesion_volume((30, 30, 30), (4.0, 4.0, 4.0), centre, 8.0, Modality.PET)

    ct_patch = crop_around_world_point(ct, np.array(centre), (32, 32, 32))
    pet_patch = crop_around_world_point(pet, np.array(centre), (32, 32, 32))

    # Both must contain lesion, and its physical volume must roughly agree.
    ct_ml = ct_patch.sum() * 1.0 ** 3 / 1000
    pet_ml = pet_patch.sum() * 4.0 ** 3 / 1000
    assert ct_patch.sum() > 0 and pet_patch.sum() > 0
    assert abs(ct_ml - pet_ml) / ct_ml < 0.35        # coarse grid -> some error


def test_naive_index_cropping_would_fail():
    """Proves the world-anchoring is doing real work, not decoration."""
    centre = (60.0, 60.0, 60.0)
    ct = lesion_volume((120, 120, 120), (1.0, 1.0, 1.0), centre, 8.0, Modality.CT)
    pet = lesion_volume((30, 30, 30), (4.0, 4.0, 4.0), centre, 8.0, Modality.PET)

    # Same VOXEL index in both -> lands in totally different anatomy.
    naive_ct = crop_with_pad(ct.array, (44, 44, 44), (32, 32, 32))
    naive_pet = crop_with_pad(pet.array, (44, 44, 44), (32, 32, 32))
    assert naive_ct.sum() > 0        # CT happens to hit it
    assert naive_pet.sum() == 0      # PET index 44 is outside a 30-voxel array


def test_foreground_bias_finds_tiny_lesions():
    """Uniform sampling misses sub-1% targets; biased sampling must not."""
    vol = np.zeros((80, 80, 80), np.float32)
    vol[38:43, 38:43, 38:43] = 1.0
    rng = np.random.default_rng(0)
    hits = sum(vol[tuple(sample_center_voxel(vol, rng, fg_threshold=0.5).astype(int))] > 0.5
               for _ in range(500))
    assert hits / 500 > 0.6


def test_sampling_is_reproducible():
    """Same seed -> same patch. Required for debuggable training runs."""
    ct = lesion_volume((60, 60, 60), (1.5, 1.5, 1.5), (45.0, 45.0, 45.0), 9.0, Modality.CT)
    vols = {Modality.CT: ct}
    a, ca = sample_study_patch(vols, np.random.default_rng(42), (16, 16, 16), 0.5)
    b, cb = sample_study_patch(vols, np.random.default_rng(42), (16, 16, 16), 0.5)
    assert np.array_equal(a[Modality.CT], b[Modality.CT])
    assert np.allclose(ca, cb)


def test_missing_modality_study_still_samples():
    """A CT-only follow-up is the common case, not an error case."""
    ct = lesion_volume((60, 60, 60), (1.5, 1.5, 1.5), (45.0, 45.0, 45.0), 9.0, Modality.CT)
    patches, _ = sample_study_patch({Modality.CT: ct}, np.random.default_rng(0), (16, 16, 16))
    assert set(patches) == {Modality.CT}
    assert patches[Modality.CT].shape == (16, 16, 16)