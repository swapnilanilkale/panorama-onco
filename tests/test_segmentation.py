"""DICOM SEG reader guards.

These tests WRITE their own SEG and CT files, so they need no downloaded data.
Each encodes a way a segmentation silently lands on the wrong anatomy.
"""
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SegmentationStorage, generate_uid

from panorama.core.constants import Modality
from panorama.core.exceptions import DataIngestionError
from panorama.data.dicom import read_series, split_acquisitions
from panorama.data.segmentation import (
    longest_axial_diameter_mm, mask_volume_ml, read_segmentation,
)
from panorama.data.volume import MedicalVolume




ROWS = COLS = 16


def write_seg(path, frames, labels, rows=ROWS, cols=COLS):
    """frames: list of (segment_number, z, 2D uint8 mask)."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = SegmentationStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = SegmentationStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "SEG"
    ds.Rows, ds.Columns = rows, cols
    ds.NumberOfFrames = len(frames)
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
    ds.PixelRepresentation = 0
    ds.SegmentationType = "BINARY"

    ds.SegmentSequence = Sequence()
    for number, label in sorted(labels.items()):
        seg = Dataset()
        seg.SegmentNumber = number
        seg.SegmentLabel = label
        seg.SegmentAlgorithmType = "MANUAL"
        ds.SegmentSequence.append(seg)

    ds.PerFrameFunctionalGroupsSequence = Sequence()
    for number, z, _ in frames:
        group = Dataset()
        ident = Dataset()
        ident.ReferencedSegmentNumber = number
        group.SegmentIdentificationSequence = Sequence([ident])
        plane = Dataset()
        plane.ImagePositionPatient = [-250.0, -250.0, float(z)]
        group.PlanePositionSequence = Sequence([plane])
        ds.PerFrameFunctionalGroupsSequence.append(group)

    ds.PixelData = np.stack([m for _, _, m in frames]).astype(np.uint8).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


def reference_volume(n_slices=6, spacing=(0.7, 0.7, 2.5)):
    affine = np.diag([*spacing, 1.0])
    affine[:3, 3] = [-250.0, -250.0, 0.0]
    return MedicalVolume(np.zeros((ROWS, COLS, n_slices), np.float32),
                         affine, Modality.CT)


def block(r0, r1, c0, c1):
    mask = np.zeros((ROWS, COLS), np.uint8)
    mask[r0:r1, c0:c1] = 1
    return mask


# --------------------------------------------------------- segment separation

def test_segments_are_separated_by_frame_metadata_not_order(tmp_path):
    """Frames are NOT guaranteed to be grouped segment-by-segment.

    A reader that slices pixel_array[0:n] as "segment 1" works on tidy files and
    silently mixes structures on others.
    """
    labels = {1: "Liver", 2: "Mass"}
    # Deliberately interleaved: liver, mass, liver, mass, ...
    frames = []
    for z in (0.0, 2.5, 5.0):
        frames.append((1, z, block(2, 12, 2, 12)))     # liver: 100 voxels/slice
        frames.append((2, z, block(4, 8, 4, 8)))       # mass:   16 voxels/slice
    path = write_seg(tmp_path / "seg.dcm", frames, labels)

    masks = read_segmentation(path, reference_volume())
    assert int(masks["Liver"].sum()) == 3 * 100
    assert int(masks["Mass"].sum()) == 3 * 16


def test_masks_do_not_bleed_between_segments(tmp_path):
    """A mass INSIDE a liver must not appear in the liver mask, or vice versa."""
    labels = {1: "Liver", 2: "Mass"}
    frames = [(1, 0.0, block(2, 12, 2, 12)), (2, 0.0, block(4, 8, 4, 8))]
    masks = read_segmentation(write_seg(tmp_path / "s.dcm", frames, labels),
                              reference_volume())
    assert not np.array_equal(masks["Liver"], masks["Mass"])
    assert int(masks["Mass"].sum()) < int(masks["Liver"].sum())


# ----------------------------------------------------------- position matching

def test_frames_are_placed_by_position_not_index(tmp_path):
    """A SEG covers only the slices containing its structures.

    Placing frame i at slice i puts contours in the wrong anatomy whenever the
    SEG starts partway down the volume.
    """
    labels = {1: "Mass"}
    # CT spans z = 0, 2.5, 5.0, 7.5, 10.0, 12.5; the SEG covers 5.0-10.0 only.
    frames = [(1, z, block(4, 8, 4, 8)) for z in (5.0, 7.5, 10.0)]
    masks = read_segmentation(write_seg(tmp_path / "s.dcm", frames, labels),
                              reference_volume(n_slices=6))
    occupied = [k for k in range(6) if masks["Mass"][:, :, k].any()]
    assert occupied == [2, 3, 4]           # NOT [0, 1, 2]


def test_frames_outside_the_reference_are_dropped(tmp_path):
    """A frame with no matching slice must be skipped, never snapped."""
    labels = {1: "Mass"}
    frames = [(1, 2.5, block(4, 8, 4, 8)),
              (1, 999.0, block(0, 16, 0, 16))]      # nowhere near the volume
    masks = read_segmentation(write_seg(tmp_path / "s.dcm", frames, labels),
                              reference_volume())
    assert int(masks["Mass"].sum()) == 16           # only the valid frame


def test_missing_frame_metadata_is_rejected(tmp_path):
    labels = {1: "Mass"}
    path = write_seg(tmp_path / "s.dcm",
                     [(1, 0.0, block(4, 8, 4, 8))], labels)
    ds = pydicom.dcmread(str(path))
    del ds.PerFrameFunctionalGroupsSequence
    ds.save_as(str(path), enforce_file_format=True)
    with pytest.raises(DataIngestionError, match="PerFrameFunctionalGroups"):
        read_segmentation(path, reference_volume())


# ------------------------------------------------------- acquisition splitting

def test_multi_acquisition_series_is_rejected(tmp_path, write_slice):
    """Two contrast phases in one series give duplicate positions.

    Reading them as one volume interleaves anatomically incoherent slices.
    """
    for i, z in enumerate([0.0, 0.0, 2.5, 2.5, 5.0, 5.0]):
        path = write_slice(tmp_path / f"IM-{i:04d}.dcm", (-250.0, -250.0, z), 1024)
        ds = pydicom.dcmread(str(path))
        ds.AcquisitionNumber = str(1 + i % 2)
        ds.save_as(str(path), enforce_file_format=True)

    files = sorted(tmp_path.glob("*.dcm"))
    with pytest.raises(DataIngestionError):
        read_series(files, Modality.CT)

    groups = split_acquisitions(files)
    assert set(groups) == {"1", "2"}
    assert all(len(v) == 3 for v in groups.values())
    # Each phase alone reads cleanly.
    volume = read_series(groups["1"], Modality.CT)
    assert volume.shape[2] == 3
    assert volume.spacing_mm[2] == pytest.approx(2.5)


# ------------------------------------------------------------- measurement

def test_longest_diameter_is_axial_not_the_3d_diagonal():
    """RECIST measures on a single slice, as a radiologist does.

    The 3D bounding diagonal over-measures whenever the through-plane extent
    exceeds the in-plane one -- common with thick slices.
    """
    volume = reference_volume(n_slices=10, spacing=(0.7, 0.7, 5.0))
    mask = np.zeros(volume.shape, bool)
    mask[2:12, 4:8, 3:8] = True            # 10 x 4 in plane, 5 slices deep
    diameter, _ = longest_axial_diameter_mm(mask, volume)

    in_plane = float(np.hypot(9 * 0.7, 3 * 0.7))
    three_d = float(np.sqrt((9 * 0.7) ** 2 + (3 * 0.7) ** 2 + (4 * 5.0) ** 2))
    assert diameter == pytest.approx(in_plane, abs=1e-6)
    assert diameter < three_d


def test_volume_uses_physical_spacing():
    volume = reference_volume(n_slices=10, spacing=(1.0, 1.0, 2.0))
    mask = np.zeros(volume.shape, bool)
    mask[0:10, 0:10, 0:5] = True           # 500 voxels x 2 mm^3 = 1000 mm^3
    assert mask_volume_ml(mask, volume) == pytest.approx(1.0)


def test_mask_shape_must_match_the_volume():
    volume = reference_volume()
    with pytest.raises(ValueError):
        longest_axial_diameter_mm(np.zeros((4, 4, 4), bool), volume)