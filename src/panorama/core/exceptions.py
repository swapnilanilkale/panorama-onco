from __future__ import annotations


class PanoramaError(Exception):
    """Base class for every error PANORAMA raises."""


class ConfigError(PanoramaError):
    """Invalid or inconsistent configuration."""


class DataIngestionError(PanoramaError):
    """Failed to read/parse a DICOM, NIfTI, or EHR source."""


class MissingModalityError(PanoramaError):
    """A required imaging stream (CT/MRI/PET) is absent for a study."""


class ModelBuildError(PanoramaError):
    """Failed to construct a component from config."""