"""Shared test fixtures.

pytest imports conftest.py automatically and injects its fixtures into every
test module, so helpers live here rather than being imported across test files.
"""
import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


@pytest.fixture
def write_slice():
    """Write one CT slice as DICOM. Returns the path."""
    def _write(path, position, value, orientation=(1, 0, 0, 0, 1, 0),
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
    return _write