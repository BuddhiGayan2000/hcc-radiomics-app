"""
Tests for DICOM series subtraction endpoint.

Tests cover:
- Valid series subtraction
- Error handling (missing sessions, no matching slices, etc.)
- Base64 PNG image generation
- Slice alignment
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import base64
import io
from PIL import Image

from app.main import app
from app.models.session_store import get_session_store

client = TestClient(app)
session_store = get_session_store()


@pytest.fixture
def hcc_010_dicom_files():
    """Get HCC_010 DICOM files for testing."""
    data_path = (
        Path(__file__).parents[2]
        / "HCC 010 final"
        / "DATA HCC 010"
        / "HCC 010"
        / "05-03-1998-NA-ABDPEL LIVER-46678"
        / "2.000000-PRE LIVER-34910"
    )

    if not data_path.exists():
        return None

    dcm_files = sorted(data_path.glob("1-*.dcm"))
    return dcm_files if dcm_files else None


class TestSubtractSeriesEndpoint:
    """Test POST /series/subtract endpoint."""

    def test_subtract_missing_post_session(self):
        """Should reject if post-contrast session not found."""
        response = client.post(
            "/series/subtract",
            json={
                "post_session_id": "invalid-post-id",
                "pre_session_id": "invalid-pre-id",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_subtract_missing_pre_session(self):
        """Should reject if pre-contrast session not found."""
        # Create a valid post session
        post_id = session_store.create_session("post-contrast")

        response = client.post(
            "/series/subtract",
            json={
                "post_session_id": post_id,
                "pre_session_id": "invalid-pre-id",
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

        # Cleanup
        session_store.delete_session(post_id)

    def test_subtract_empty_post_session(self):
        """Should reject if post-contrast session is empty."""
        post_id = session_store.create_session("post-contrast")
        pre_id = session_store.create_session("pre-contrast")

        response = client.post(
            "/series/subtract",
            json={
                "post_session_id": post_id,
                "pre_session_id": pre_id,
            },
        )

        assert response.status_code == 400
        assert "no slices" in response.json()["detail"]

        # Cleanup
        session_store.delete_session(post_id)
        session_store.delete_session(pre_id)

    def test_subtract_no_matching_slices(self):
        """Should reject if no slices match between series."""
        from app.models.session_store import SeriesSlice
        import numpy as np

        post_id = session_store.create_session("post-contrast")
        pre_id = session_store.create_session("pre-contrast")

        # Add slices with different indices
        fake_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(
            np.float32
        )

        post_session = session_store.get_session(post_id)
        pre_session = session_store.get_session(pre_id)

        post_session.add_slice(
            SeriesSlice(
                index="01",
                filename="1-01.dcm",
                pixel_array=fake_array,
                width=100,
                height=100,
                min_intensity=0,
                max_intensity=256,
            )
        )

        pre_session.add_slice(
            SeriesSlice(
                index="02",  # Different index
                filename="1-02.dcm",
                pixel_array=fake_array,
                width=100,
                height=100,
                min_intensity=0,
                max_intensity=256,
            )
        )

        response = client.post(
            "/series/subtract",
            json={
                "post_session_id": post_id,
                "pre_session_id": pre_id,
            },
        )

        assert response.status_code == 400
        assert "No matching slices" in response.json()["detail"]

        # Cleanup
        session_store.delete_session(post_id)
        session_store.delete_session(pre_id)

    @pytest.mark.skipif(
        Path(
            "HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910"
        ).exists()
        is False,
        reason="HCC_010 test data not found",
    )
    def test_subtract_real_dicom_series(self, hcc_010_dicom_files):
        """Test subtraction with real HCC_010 DICOM files."""
        if not hcc_010_dicom_files:
            pytest.skip("HCC_010 data not available")

        # Upload post-contrast
        test_files = hcc_010_dicom_files[:5]
        files = [
            ("files", (dcm_path.name, open(dcm_path, "rb")))
            for dcm_path in test_files
        ]

        try:
            post_response = client.post(
                "/series/upload",
                data={"phase": "post-contrast"},
                files=files,
            )

            assert post_response.status_code == 200
            post_id = post_response.json()["session_id"]

            # Upload pre-contrast (same files for testing)
            files = [
                ("files", (dcm_path.name, open(dcm_path, "rb")))
                for dcm_path in test_files
            ]

            pre_response = client.post(
                "/series/upload",
                data={"phase": "pre-contrast"},
                files=files,
            )

            assert pre_response.status_code == 200
            pre_id = pre_response.json()["session_id"]

            # Subtract
            subtract_response = client.post(
                "/series/subtract",
                json={
                    "post_session_id": post_id,
                    "pre_session_id": pre_id,
                },
            )

            assert subtract_response.status_code == 200

            data = subtract_response.json()
            assert data["total"] == 5
            assert len(data["subtracted_series"]) == 5

            # Check each subtracted slice
            for slice_info in data["subtracted_series"]:
                assert slice_info["index"].isdigit()
                assert slice_info["filename"].endswith(".dcm")
                assert slice_info["width"] == 512
                assert slice_info["height"] == 512

                # Verify base64 PNG
                assert slice_info["image_data_b64"].startswith("data:image/png;base64,")

                # Decode and verify it's valid PNG
                b64_part = slice_info["image_data_b64"].replace(
                    "data:image/png;base64,", ""
                )
                png_bytes = base64.b64decode(b64_part)

                img = Image.open(io.BytesIO(png_bytes))
                assert img.size == (512, 512)
                assert img.mode == "L"  # Grayscale

            # Cleanup
            session_store.delete_session(post_id)
            session_store.delete_session(pre_id)

        finally:
            # Close file handles
            for _, (_, f) in files:
                try:
                    f.close()
                except:
                    pass

    @pytest.mark.skipif(
        Path(
            "HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910"
        ).exists()
        is False,
        reason="HCC_010 test data not found",
    )
    def test_subtract_preserves_slice_data(self, hcc_010_dicom_files):
        """Test that subtraction preserves raw_subtracted_array for feature extraction."""
        if not hcc_010_dicom_files:
            pytest.skip("HCC_010 data not available")

        # Upload post and pre
        test_files = hcc_010_dicom_files[:3]

        files = [
            ("files", (dcm_path.name, open(dcm_path, "rb")))
            for dcm_path in test_files
        ]

        try:
            post_response = client.post(
                "/series/upload",
                data={"phase": "post-contrast"},
                files=files,
            )

            post_id = post_response.json()["session_id"]

            files = [
                ("files", (dcm_path.name, open(dcm_path, "rb")))
                for dcm_path in test_files
            ]

            pre_response = client.post(
                "/series/upload",
                data={"phase": "pre-contrast"},
                files=files,
            )

            pre_id = pre_response.json()["session_id"]

            # Subtract
            subtract_response = client.post(
                "/series/subtract",
                json={
                    "post_session_id": post_id,
                    "pre_session_id": pre_id,
                },
            )

            assert subtract_response.status_code == 200

            # Verify that raw_subtracted_array is stored in post session
            post_session = session_store.get_session(post_id)
            assert post_session is not None

            for slice_obj in post_session.get_sorted_slices():
                assert slice_obj.raw_subtracted_array is not None
                assert slice_obj.raw_subtracted_array.shape == (512, 512)

            # Cleanup
            session_store.delete_session(post_id)
            session_store.delete_session(pre_id)

        finally:
            for _, (_, f) in files:
                try:
                    f.close()
                except:
                    pass


class TestSubtractionIntegration:
    """Integration tests for full upload → subtract workflow."""

    @pytest.mark.skipif(
        Path(
            "HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910"
        ).exists()
        is False,
        reason="HCC_010 test data not found",
    )
    def test_full_upload_and_subtract_workflow(self, hcc_010_dicom_files):
        """Test complete workflow: upload post → upload pre → subtract."""
        if not hcc_010_dicom_files:
            pytest.skip("HCC_010 data not available")

        test_files = hcc_010_dicom_files[:4]

        # Step 1: Upload post-contrast
        files = [
            ("files", (dcm_path.name, open(dcm_path, "rb")))
            for dcm_path in test_files
        ]

        try:
            post_response = client.post(
                "/series/upload",
                data={"phase": "post-contrast"},
                files=files,
            )

            assert post_response.status_code == 200
            post_data = post_response.json()
            post_id = post_data["session_id"]

            assert post_data["slice_count"] == 4
            assert post_data["phase"] == "post-contrast"

            # Step 2: Upload pre-contrast
            files = [
                ("files", (dcm_path.name, open(dcm_path, "rb")))
                for dcm_path in test_files
            ]

            pre_response = client.post(
                "/series/upload",
                data={"phase": "pre-contrast"},
                files=files,
            )

            assert pre_response.status_code == 200
            pre_data = pre_response.json()
            pre_id = pre_data["session_id"]

            assert pre_data["slice_count"] == 4

            # Step 3: Compute subtraction
            subtract_response = client.post(
                "/series/subtract",
                json={
                    "post_session_id": post_id,
                    "pre_session_id": pre_id,
                },
            )

            assert subtract_response.status_code == 200
            subtract_data = subtract_response.json()

            assert subtract_data["total"] == 4
            assert len(subtract_data["subtracted_series"]) == 4

            # Verify all slices are present and valid
            indices = [s["index"] for s in subtract_data["subtracted_series"]]
            assert len(indices) == 4
            assert all(idx.isdigit() for idx in indices)

            # Cleanup
            session_store.delete_session(post_id)
            session_store.delete_session(pre_id)

        finally:
            for _, (_, f) in files:
                try:
                    f.close()
                except:
                    pass
