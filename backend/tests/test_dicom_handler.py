"""
Tests for DICOM parsing and series alignment (app/models/dicom_handler.py).

These tests verify:
- DICOM file parsing
- Windowing/leveling
- Series alignment by filename
- Image subtraction
- Base64 encoding
"""

import base64
import io
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from app.models.dicom_handler import (
    parse_dicom_file,
    apply_windowing,
    slice_sort_key,
    assign_slice_indices,
    best_slice_position,
    subtract_images,
    normalize_for_display,
    array_to_png_base64,
    DICOMParseError,
)


class TestSliceSortKey:
    """Test ordering-key selection, independent of filename convention."""

    def test_prefers_slice_location(self):
        """SliceLocation should win over instance_number/filename when present."""
        a = {"slice_location": 10.0, "instance_number": 5, "filename": "z.dcm"}
        b = {"slice_location": 5.0, "instance_number": 1, "filename": "a.dcm"}
        assert slice_sort_key(b) < slice_sort_key(a)

    def test_falls_back_to_image_position_z(self):
        """ImagePositionPatient z should be used when SliceLocation is absent."""
        a = {"image_position_z": 20.0, "filename": "a.dcm"}
        b = {"image_position_z": 5.0, "filename": "b.dcm"}
        assert slice_sort_key(b) < slice_sort_key(a)

    def test_falls_back_to_instance_number(self):
        """InstanceNumber should be used when no position metadata exists."""
        a = {"instance_number": 10, "filename": "a.dcm"}
        b = {"instance_number": 2, "filename": "b.dcm"}
        assert slice_sort_key(b) < slice_sort_key(a)

    def test_falls_back_to_filename_as_last_resort(self):
        """With no DICOM metadata at all, filename is the last-resort tiebreaker."""
        a = {"filename": "IMG1000012.dcm"}
        b = {"filename": "1-012.dcm"}
        # Whatever the filenames look like, sorting must be deterministic and
        # must not raise — the app no longer assumes any naming convention.
        assert slice_sort_key(b) < slice_sort_key(a)

    def test_position_always_outranks_instance_and_filename(self):
        """A file with position metadata sorts before one without, regardless of values."""
        has_position = {"slice_location": 999.0, "filename": "a.dcm"}
        no_position = {"instance_number": 1, "filename": "z.dcm"}
        assert slice_sort_key(has_position) < slice_sort_key(no_position)


class TestAssignSliceIndices:
    """Test assigning zero-padded slice indices from arbitrary filenames."""

    def test_orders_by_slice_location_regardless_of_filename(self):
        """Real-world filenames (no shared convention) still order correctly."""
        metadata_list = [
            {"slice_location": 30.0, "filename": "IMG1000012.dcm"},
            {"slice_location": 10.0, "filename": "scan_a.dcm"},
            {"slice_location": 20.0, "filename": "1-012.dcm"},
        ]
        indices = assign_slice_indices(metadata_list)

        # metadata_list[1] (location 10.0) should be first, etc.
        assert indices[1] == "01"
        assert indices[2] == "02"
        assert indices[0] == "03"

    def test_zero_pads_to_batch_size(self):
        """Index width should grow with slice count (e.g. 3 digits for >99 slices)."""
        metadata_list = [{"instance_number": i, "filename": f"f{i}.dcm"} for i in range(150)]
        indices = assign_slice_indices(metadata_list)

        assert len(indices[0]) == 3
        assert indices == sorted(indices)  # zero-padding keeps lexicographic order numeric

    def test_no_metadata_still_produces_deterministic_indices(self):
        """Files with no position/instance metadata at all still get an order, never an error."""
        metadata_list = [{"filename": "b.dcm"}, {"filename": "a.dcm"}]
        indices = assign_slice_indices(metadata_list)

        assert sorted(indices) == ["01", "02"]
        assert len(set(indices)) == 2  # each file gets a distinct index


class TestBestSlicePosition:
    """Test extraction of the best-available physical position."""

    def test_prefers_slice_location(self):
        assert best_slice_position({"slice_location": 12.5, "image_position_z": 99.0}) == 12.5

    def test_falls_back_to_image_position_z(self):
        assert best_slice_position({"image_position_z": 7.0}) == 7.0

    def test_returns_none_without_position_metadata(self):
        assert best_slice_position({"instance_number": 3, "filename": "a.dcm"}) is None


class TestWindowing:
    """Test DICOM windowing/leveling."""

    def test_windowing_standard_values(self):
        """Windowing should map HU to [0, 255]."""
        # Create array with known HU values
        array = np.array([[0, 50, 100], [25, 75, 150]], dtype=np.float32)

        windowed = apply_windowing(array, window_width=400, window_center=50)

        # window_center=50, window_width=400 means range [50-200, 50+200] = [-150, 250]
        # So HU=50 should map to ~128 (middle of 0-255)
        assert windowed.dtype == np.uint8
        assert windowed.min() >= 0
        assert windowed.max() <= 255

    def test_windowing_clipping(self):
        """Values below/above window should clip to 0/255."""
        array = np.array([[-200.0, 300.0]], dtype=np.float32)
        windowed = apply_windowing(array, window_width=400, window_center=50)

        # -200 is way below window → should be 0
        # 300 is way above window → should be 255
        assert windowed[0, 0] == 0
        assert windowed[0, 1] == 255

    def test_windowing_default_params(self):
        """Windowing should work with default liver CT params."""
        array = np.random.randn(100, 100).astype(np.float32) * 50 + 50
        windowed = apply_windowing(array)  # Uses defaults

        assert windowed.shape == array.shape
        assert windowed.dtype == np.uint8


class TestSubtractImages:
    """Test image subtraction."""

    def test_subtract_simple(self):
        """Post - Pre should work element-wise."""
        post = np.array([[100, 200], [150, 250]], dtype=np.float32)
        pre = np.array([[50, 100], [50, 100]], dtype=np.float32)

        subtracted = subtract_images(post, pre)

        expected = np.array([[50, 100], [100, 150]], dtype=np.float32)
        np.testing.assert_array_equal(subtracted, expected)

    def test_subtract_clipping(self):
        """Subtraction should clip negative values to 0."""
        post = np.array([[100, 50]], dtype=np.float32)
        pre = np.array([[150, 100]], dtype=np.float32)

        subtracted = subtract_images(post, pre)

        # 100 - 150 = -50 → clips to 0
        # 50 - 100 = -50 → clips to 0
        expected = np.array([[0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(subtracted, expected)

    def test_subtract_shape_mismatch(self):
        """Should raise error if shapes don't match."""
        post = np.ones((100, 100))
        pre = np.ones((100, 99))

        with pytest.raises(ValueError, match="same shape"):
            subtract_images(post, pre)


class TestNormalizeForDisplay:
    """Test normalization to [0, 255]."""

    def test_normalize_basic(self):
        """Normalize to uint8 [0, 255]."""
        array = np.array([[0.0, 50.0, 100.0]])
        normalized = normalize_for_display(array)

        assert normalized.dtype == np.uint8
        assert normalized.min() == 0
        assert normalized.max() == 255

    def test_normalize_constant_array(self):
        """Constant array should map to 128 (mid-gray)."""
        array = np.full((10, 10), 50.0)
        normalized = normalize_for_display(array)

        assert normalized.dtype == np.uint8
        assert np.all(normalized == 128)

    def test_normalize_empty_array(self):
        """Empty array should return valid result."""
        array = np.array([], dtype=np.float32).reshape(0, 0)
        normalized = normalize_for_display(array)

        assert normalized.dtype == np.uint8


class TestArrayToPNGBase64:
    """Test conversion to base64-encoded PNG."""

    def test_encode_basic(self):
        """Encode grayscale array to base64 PNG."""
        array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        b64 = array_to_png_base64(array)

        # Should be valid data URL
        assert b64.startswith("data:image/png;base64,")

        # Extract and decode
        b64_part = b64.replace("data:image/png;base64,", "")
        png_bytes = base64.b64decode(b64_part)

        # Verify it's a valid PNG
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (100, 100)
        assert img.mode == "L"  # Grayscale

    def test_encode_converts_dtype(self):
        """Should convert non-uint8 arrays."""
        array = np.random.rand(50, 50).astype(np.float32)
        b64 = array_to_png_base64(array)

        assert b64.startswith("data:image/png;base64,")


class TestParseDICOMFile:
    """Test DICOM file parsing (integration test).

    NOTE: These tests require actual DICOM files from HCC_010 dataset.
    They will be skipped if the test data is not available.
    """

    @pytest.fixture(scope="class")
    def dicom_test_data(self):
        """Find HCC_010 DICOM files for testing."""
        data_path = Path(__file__).parents[3] / "HCC 010 final" / "DATA HCC 010" / "HCC 010" / "05-03-1998-NA-ABDPEL LIVER-46678" / "2.000000-PRE LIVER-34910"

        if not data_path.exists():
            return None

        dcm_files = sorted(data_path.glob("1-*.dcm"))
        return dcm_files[:3] if dcm_files else None

    def test_parse_real_dicom(self, dicom_test_data):
        """Parse actual HCC_010 DICOM files."""
        if not dicom_test_data:
            pytest.skip("HCC_010 test data not found")

        for dcm_file in dicom_test_data:
            pixel_array, metadata = parse_dicom_file(str(dcm_file))

            # Check array properties
            assert isinstance(pixel_array, np.ndarray)
            assert pixel_array.ndim == 2
            assert pixel_array.size > 0

            # Check metadata
            assert "filename" in metadata
            assert "width" in metadata
            assert "height" in metadata
            assert metadata["filename"] == dcm_file.name

    def test_parse_nonexistent_file(self):
        """Should raise error for nonexistent file."""
        with pytest.raises(DICOMParseError):
            parse_dicom_file("/nonexistent/file.dcm")


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.fixture(scope="class")
    def hcc_010_files(self):
        """Get HCC_010 DICOM files for integration testing."""
        data_path = Path(__file__).parents[3] / "HCC 010 final" / "DATA HCC 010" / "HCC 010" / "05-03-1998-NA-ABDPEL LIVER-46678" / "2.000000-PRE LIVER-34910"

        if not data_path.exists():
            return None

        dcm_files = sorted(data_path.glob("1-*.dcm"))
        return [str(f) for f in dcm_files] if dcm_files else None

    def test_subtraction_pipeline_synthetic(self):
        """Test full subtraction pipeline with synthetic data."""
        # Create synthetic post/pre images
        post = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        pre = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Subtract
        subtracted = subtract_images(post.astype(np.float32), pre.astype(np.float32))

        # Normalize
        normalized = normalize_for_display(subtracted)

        # Encode
        b64 = array_to_png_base64(normalized)

        # Verify
        assert b64.startswith("data:image/png;base64,")
        assert normalized.dtype == np.uint8

    def test_series_alignment_and_subtraction(self, hcc_010_files):
        """Test full pipeline: align and subtract real DICOM series."""
        if not hcc_010_files:
            pytest.skip("HCC_010 test data not found")

        # We only have pre-contrast, so test with subset
        if len(hcc_010_files) < 2:
            pytest.skip("Need at least 2 DICOM files")

        # Use same files as both post and pre for testing
        # (In real use, these would be different phases)
        post_files = hcc_010_files[:5]
        pre_files = hcc_010_files[:5]

        # Parse both
        post_arrays = [parse_dicom_file(f)[0] for f in post_files]
        pre_arrays = [parse_dicom_file(f)[0] for f in pre_files]

        # Subtract first pair
        subtracted = subtract_images(post_arrays[0], pre_arrays[0])

        # Should produce valid result
        assert isinstance(subtracted, np.ndarray)
        assert subtracted.shape == post_arrays[0].shape
        assert subtracted.min() >= 0
