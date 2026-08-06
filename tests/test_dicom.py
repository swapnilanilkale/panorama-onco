"""DICOM series reader guards.

These tests WRITE their own DICOM files, so they need no downloaded data. Each
one encodes a way that real DICOM silently corrupts a volume: wrong slice order,
a dropped slice, unrescaled pixel values, or an assumed slice axis.
"""
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from panorama.core.constants import Modality
from panorama.core.exceptions import DataIngestionError
from panorama.data.dicom import read_series


def write_slice(path, position, value, orientation=(1, 0, 0, 0, 1, 0),
                rows=16, cols=16, slope=1.0, intercept=-1024.0,
                pixel_spacing=(0.7, 0.7)):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = list(pixel_spacing)
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = [float(v) for v in position]
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((rows, cols), value, np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


def axial_series(directory, z_positions, filename_ids=None, **kwargs):
    """Write one slice per z. Slice k gets stored value 1024 + k*10."""
    ids = filename_ids or range(len(z_positions))
    for k, (z, name_id) in enumerate(zip(z_positions, ids)):
        write_slice(directory / f"IM-{name_id:04d}.dcm",
                    position=(-250.0, -250.0, z), value=1024 + k * 10, **kwargs)
    return sorted(directory.glob("*.dcm"))


# ----------------------------------------------------------------- ordering

def test_slices_are_sorted_by_position_not_filename(tmp_path):
    """Directory order is lexicographic: IM-0010 sorts before IM-0002.

    Reading in filename order shuffles anatomy. The result still looks like a
    volume, so nothing crashes -- it is simply wrong.
    """
    z = [k * 2.5 for k in range(8)]
    ids = [0, 10, 2, 20, 3, 1, 4, 5]        # deliberately not in slice order
    paths = axial_series(tmp_path, z, filename_ids=ids)

    # The fixture is only meaningful if the two orders actually differ.
    z_by_filename = [float(pydicom.dcmread(str(p)).ImagePositionPatient[2])
                     for p in paths]
    assert z_by_filename != sorted(z_by_filename)

    volume = read_series(paths, Modality.CT)
    centre = [float(volume.array[8, 8, k]) for k in range(volume.shape[2])]
    assert centre == sorted(centre)          # 0, 10, 20, ... strictly increasing


def test_slice_axis_is_the_cross_product_not_assumed_z(tmp_path):
    """For an oblique acquisition the slice axis is not a patient axis.

    DICOM stores only the in-plane direction cosines; the third axis must be
    derived. Assuming +z gives a wrong affine on any oblique series.
    """
    c = float(np.sqrt(0.5))
    oblique = (1, 0, 0, 0, c, c)             # coronal-oblique
    for k in range(6):
        write_slice(tmp_path / f"IM-{k:04d}.dcm",
                    position=(0.0, -k * 2.5 * c, k * 2.5 * c),
                    value=1024, orientation=oblique)
    volume = read_series(sorted(tmp_path.glob("*.dcm")), Modality.CT)
    slice_column = volume.affine[:3, 2]
    assert not np.allclose(slice_column / np.linalg.norm(slice_column), [0, 0, 1])


# ------------------------------------------------------------------ spacing

def test_spacing_is_derived_from_positions(tmp_path):
    paths = axial_series(tmp_path, [k * 2.5 for k in range(6)])
    volume = read_series(paths, Modality.CT)
    assert volume.spacing_mm == pytest.approx((0.7, 0.7, 2.5), abs=1e-6)


def test_missing_slice_is_rejected(tmp_path):
    """A dropped slice makes every physical measurement wrong, silently."""
    paths = axial_series(tmp_path, [0.0, 2.5, 5.0, 7.5, 12.5])   # 10.0 absent
    with pytest.raises(DataIngestionError, match="irregular slice spacing"):
        read_series(paths, Modality.CT)


def test_missing_slice_can_be_forced(tmp_path):
    paths = axial_series(tmp_path, [0.0, 2.5, 5.0, 7.5, 12.5])
    volume = read_series(paths, Modality.CT, allow_irregular=True)
    assert volume.shape[2] == 5


def test_small_jitter_is_tolerated(tmp_path):
    """Real scanners have sub-1% variation; the guard must not fire on it."""
    paths = axial_series(tmp_path, [0.0, 2.5, 5.02, 7.5, 10.0])
    volume = read_series(paths, Modality.CT)
    assert volume.spacing_mm[2] == pytest.approx(2.5, abs=0.05)


def test_single_slice_series_is_rejected(tmp_path):
    write_slice(tmp_path / "IM-0000.dcm", (0.0, 0.0, 0.0), 1024)
    with pytest.raises(DataIngestionError, match="only 1 slice"):
        read_series(sorted(tmp_path.glob("*.dcm")), Modality.CT)


# ----------------------------------------------------------------- rescale

def test_rescale_maps_stored_values_to_hounsfield_units(tmp_path):
    """Water is stored as 1024 and must read as 0 HU.

    Skipping RescaleSlope/Intercept breaks the fixed HU windowing that makes
    CT normalisation scanner-invariant.
    """
    for k, z in enumerate([0.0, 2.5, 5.0]):
        write_slice(tmp_path / f"IM-{k:04d}.dcm", (-250.0, -250.0, z), value=1024)
    volume = read_series(sorted(tmp_path.glob("*.dcm")), Modality.CT)
    assert float(volume.array[8, 8, 0]) == pytest.approx(0.0)


def test_non_unit_slope_is_applied(tmp_path):
    for k, z in enumerate([0.0, 2.5, 5.0]):
        write_slice(tmp_path / f"IM-{k:04d}.dcm", (-250.0, -250.0, z),
                    value=200, slope=2.0, intercept=-1024.0)
    volume = read_series(sorted(tmp_path.glob("*.dcm")), Modality.CT)
    assert float(volume.array[8, 8, 0]) == pytest.approx(-624.0)


# ---------------------------------------------------------------- geometry

def test_affine_matches_the_medical_volume_contract(tmp_path):
    """The affine must be the one MedicalVolume, resample and cropping expect."""
    paths = axial_series(tmp_path, [k * 2.5 for k in range(6)])
    volume = read_series(paths, Modality.CT)
    assert volume.extent_mm == pytest.approx((16 * 0.7, 16 * 0.7, 6 * 2.5), abs=1e-6)
    assert np.allclose(volume.affine[:3, 3], [-250.0, -250.0, 0.0])


def test_missing_position_tag_is_rejected(tmp_path):
    ds = pydicom.dcmread(str(write_slice(tmp_path / "a.dcm", (0., 0., 0.), 1024)))
    del ds.ImagePositionPatient
    ds.save_as(str(tmp_path / "a.dcm"), enforce_file_format=True)
    with pytest.raises(DataIngestionError, match="ImagePositionPatient"):
        read_series([tmp_path / "a.dcm"], Modality.CT)