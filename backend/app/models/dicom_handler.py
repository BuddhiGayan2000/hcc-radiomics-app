"""
DICOM parsing and series alignment for HCC radiomics.

Handles:
- Reading DICOM files with pydicom
- Windowing/leveling for display (liver CT: window=400, level=50)
- Ordering slices within a series and aligning pre/post-contrast series by
  their own DICOM position metadata (never by filename — see
  slice_sort_key/best_slice_position; real-world DICOM exports use every
  naming convention imaginable, e.g. "1-01.dcm", "1-012.dcm", "IMG1000012.dcm")
- Computing subtracted images (Post - Pre)
- Encoding images as base64 PNG for transmission to frontend
"""

import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pydicom
from PIL import Image


# DICOM Windowing constants (standard for liver CT)
LIVER_WINDOW_WIDTH = 400
LIVER_WINDOW_CENTER = 50


class DICOMParseError(Exception):
    """Raised when DICOM file cannot be parsed."""
    pass


def parse_dicom_file(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Parse a single DICOM file and extract pixel data.

    Args:
        file_path: Path to .dcm file

    Returns:
        (pixel_array, metadata_dict) where:
        - pixel_array: numpy array of pixel intensities (may be in HU units)
        - metadata_dict: Contains 'filename', 'shape', 'window_width', etc.

    Raises:
        DICOMParseError: If file cannot be read
    """
    try:
        dcm = pydicom.dcmread(file_path)
    except Exception as e:
        raise DICOMParseError(f"Cannot read DICOM file {file_path}: {e}")

    try:
        pixel_array = dcm.pixel_array.astype(np.float32)
    except Exception as e:
        raise DICOMParseError(f"Cannot extract pixel data from {file_path}: {e}")

    # Extract useful metadata
    filename = Path(file_path).name
    height, width = pixel_array.shape

    metadata = {
        "filename": filename,
        "width": width,
        "height": height,
        "shape": pixel_array.shape,
        "min_intensity": float(pixel_array.min()),
        "max_intensity": float(pixel_array.max()),
    }

    # Try to extract slice position and other DICOM metadata. These (not
    # filenames) are what ordering and cross-series alignment are based on.
    try:
        if hasattr(dcm, "SliceLocation"):
            metadata["slice_location"] = float(dcm.SliceLocation)
        if hasattr(dcm, "ImagePositionPatient") and len(dcm.ImagePositionPatient) == 3:
            metadata["image_position_z"] = float(dcm.ImagePositionPatient[2])
        if hasattr(dcm, "InstanceNumber"):
            metadata["instance_number"] = int(dcm.InstanceNumber)
        if hasattr(dcm, "PatientID"):
            metadata["patient_id"] = str(dcm.PatientID)
        if hasattr(dcm, "SeriesInstanceUID"):
            metadata["series_instance_uid"] = str(dcm.SeriesInstanceUID)
    except Exception:
        pass  # Not all DICOM files have this metadata

    return pixel_array, metadata


def slice_sort_key(metadata: Dict) -> Tuple[int, object]:
    """
    Best-effort ordering key for a slice within one uploaded series,
    independent of filename.

    Preference order: physical slice position (SliceLocation, then the z
    component of ImagePositionPatient) > InstanceNumber > filename. The
    leading int groups slices by which criterion was available, so a
    tuple's second element is only ever compared against another value of
    the same type (float position, int instance number, or str filename).
    """
    if "slice_location" in metadata:
        return (0, metadata["slice_location"])
    if "image_position_z" in metadata:
        return (0, metadata["image_position_z"])
    if "instance_number" in metadata:
        return (1, metadata["instance_number"])
    return (2, metadata["filename"])


def assign_slice_indices(metadata_list: List[Dict]) -> List[str]:
    """
    Assign a stable, zero-padded slice index ("01", "02", ...) to each file
    in an uploaded series, ordered by slice_sort_key rather than filename.

    Args:
        metadata_list: parsed metadata dicts in original upload order (each
            must include "filename"; see parse_dicom_file)

    Returns:
        Index strings parallel to metadata_list — metadata_list[i] is
        assigned indices[i].
    """
    n = len(metadata_list)
    width = max(2, len(str(n)))
    order = sorted(range(n), key=lambda i: slice_sort_key(metadata_list[i]))

    indices: List[Optional[str]] = [None] * n
    for rank, original_i in enumerate(order):
        indices[original_i] = str(rank + 1).zfill(width)
    return indices


def best_slice_position(metadata: Dict) -> Optional[float]:
    """
    Best-available physical position (mm) for a slice, used to line up the
    same anatomical location across two different series (e.g. matching a
    post-contrast slice to its pre-contrast counterpart). Prefers
    SliceLocation; falls back to the z component of ImagePositionPatient.
    Returns None if neither tag is present (some anonymized/stripped DICOM
    omit both).
    """
    if "slice_location" in metadata:
        return metadata["slice_location"]
    if "image_position_z" in metadata:
        return metadata["image_position_z"]
    return None


def apply_windowing(
    pixel_array: np.ndarray,
    window_width: float = LIVER_WINDOW_WIDTH,
    window_center: float = LIVER_WINDOW_CENTER,
) -> np.ndarray:
    """
    Apply CT windowing/leveling to DICOM pixel data.

    Windowing maps Hounsfield units to a display range.
    Standard for liver CT: window_width=400, window_center=50.

    Args:
        pixel_array: Input pixel array (may be in HU units)
        window_width: Width of window (default: 400 HU)
        window_center: Center of window (default: 50 HU)

    Returns:
        Windowed array clipped to [0, 255] for display
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2

    windowed = np.clip(pixel_array, min_val, max_val)
    windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return windowed


def subtract_images(
    post_array: np.ndarray,
    pre_array: np.ndarray,
) -> np.ndarray:
    """
    Compute subtracted image: Post - Pre (with clipping to [0, max]).

    Subtraction enhances regions that are brighter in post-contrast image
    (e.g., vascularized tumors that accumulate contrast).

    Args:
        post_array: Post-contrast pixel array
        pre_array: Pre-contrast pixel array

    Returns:
        Subtracted array (Post - Pre, clipped to >= 0)
    """
    if post_array.shape != pre_array.shape:
        raise ValueError(
            f"Arrays must have same shape: post {post_array.shape} vs pre {pre_array.shape}"
        )

    subtracted = post_array - pre_array
    subtracted = np.maximum(subtracted, 0)  # Clip to >= 0

    return subtracted


def normalize_for_display(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 255] range for display/PNG encoding.

    Args:
        array: Input array (any range)

    Returns:
        Normalized array as uint8 in range [0, 255]
    """
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    min_val = np.min(array)
    max_val = np.max(array)

    if min_val == max_val:
        # Constant array
        return np.full(array.shape, 128, dtype=np.uint8)

    normalized = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized


def array_to_png_base64(array: np.ndarray) -> str:
    """
    Convert numpy array to PNG and encode as base64 data URL.

    Args:
        array: 2D numpy array (uint8 or will be converted)

    Returns:
        Data URL string: "data:image/png;base64,..."
    """
    if array.dtype != np.uint8:
        array = normalize_for_display(array)

    # Create PIL image from array
    pil_img = Image.fromarray(array, mode="L")  # L = grayscale

    # Encode to PNG
    png_buffer = io.BytesIO()
    pil_img.save(png_buffer, format="PNG")
    png_bytes = png_buffer.getvalue()

    # Encode to base64
    b64_str = base64.b64encode(png_bytes).decode("utf-8")

    return f"data:image/png;base64,{b64_str}"
