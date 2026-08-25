"""
DICOM parsing and series alignment for HCC radiomics.

Handles:
- Reading DICOM files with pydicom
- Windowing/leveling for display (liver CT: window=400, level=50)
- Aligning pre/post-contrast series by filename
- Computing subtracted images (Post - Pre)
- Encoding images as base64 PNG for transmission to frontend
"""

import base64
import io
import re
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


class SeriesAlignmentError(Exception):
    """Raised when pre/post series cannot be aligned."""
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

    # Try to extract slice position and other DICOM metadata
    try:
        if hasattr(dcm, "SliceLocation"):
            metadata["slice_location"] = float(dcm.SliceLocation)
        if hasattr(dcm, "InstanceNumber"):
            metadata["instance_number"] = int(dcm.InstanceNumber)
        if hasattr(dcm, "PatientID"):
            metadata["patient_id"] = str(dcm.PatientID)
    except Exception:
        pass  # Not all DICOM files have this metadata

    return pixel_array, metadata


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


def extract_slice_index(filename: str) -> Optional[str]:
    """
    Extract slice index from DICOM filename.

    Expected format: "1-XX.dcm" where XX is the slice number (01, 02, ..., 20)
    Returns the XX part for matching.

    Args:
        filename: DICOM filename (e.g., "1-05.dcm")

    Returns:
        Slice index string (e.g., "05") or None if pattern doesn't match
    """
    match = re.match(r"1-(\d{2})\.dcm", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def align_series(
    post_files: List[str],
    pre_files: List[str],
) -> List[Tuple[str, str, str]]:
    """
    Align pre-contrast and post-contrast DICOM series by filename.

    Matches files with the same slice index (e.g., 1-05.dcm).

    Args:
        post_files: List of paths to post-contrast DICOM files
        pre_files: List of paths to pre-contrast DICOM files

    Returns:
        List of (post_path, pre_path, slice_index) tuples for matched pairs

    Raises:
        SeriesAlignmentError: If series cannot be aligned
    """
    # Build index maps
    post_map = {}
    for f in post_files:
        idx = extract_slice_index(Path(f).name)
        if idx:
            post_map[idx] = f

    pre_map = {}
    for f in pre_files:
        idx = extract_slice_index(Path(f).name)
        if idx:
            pre_map[idx] = f

    # Find common indices
    common_indices = set(post_map.keys()) & set(pre_map.keys())

    if not common_indices:
        raise SeriesAlignmentError(
            f"No matching slices found. Post-contrast indices: {sorted(post_map.keys())}, "
            f"Pre-contrast indices: {sorted(pre_map.keys())}"
        )

    # Return sorted by index
    pairs = [
        (post_map[idx], pre_map[idx], idx)
        for idx in sorted(common_indices)
    ]

    return pairs


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


def process_dicom_series(
    post_files: List[str],
    pre_files: List[str],
) -> List[Dict]:
    """
    Full pipeline: Load, align, and subtract a DICOM series.

    Args:
        post_files: List of post-contrast DICOM file paths
        pre_files: List of pre-contrast DICOM file paths

    Returns:
        List of dicts with keys:
        - 'index': Slice index (01, 02, ..., 20)
        - 'post_filename': Post-contrast filename
        - 'pre_filename': Pre-contrast filename
        - 'width': Image width
        - 'height': Image height
        - 'subtracted_image_b64': Base64-encoded PNG of subtracted image
        - 'raw_subtracted_array': Raw subtracted numpy array (for feature extraction)

    Raises:
        DICOMParseError: If any file cannot be parsed
        SeriesAlignmentError: If series cannot be aligned
    """
    # Align series
    pairs = align_series(post_files, pre_files)

    results = []
    for post_path, pre_path, slice_idx in pairs:
        # Parse both files
        post_array, post_meta = parse_dicom_file(post_path)
        pre_array, pre_meta = parse_dicom_file(pre_path)

        # Apply windowing for display
        post_windowed = apply_windowing(post_array)
        pre_windowed = apply_windowing(pre_array)

        # Compute subtraction (on windowed data for visual consistency)
        subtracted_windowed = subtract_images(post_windowed, pre_windowed)

        # Also compute on raw data for feature extraction
        subtracted_raw = subtract_images(post_array, pre_array)

        # Normalize and encode for transmission
        subtracted_display = normalize_for_display(subtracted_windowed)
        b64_png = array_to_png_base64(subtracted_display)

        result = {
            "index": slice_idx,
            "post_filename": Path(post_path).name,
            "pre_filename": Path(pre_path).name,
            "width": post_meta["width"],
            "height": post_meta["height"],
            "subtracted_image_b64": b64_png,
            "raw_subtracted_array": subtracted_raw,  # For feature extraction later
        }

        results.append(result)

    return results
