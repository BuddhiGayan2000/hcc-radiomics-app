"""
Tests for DICOM series feature extraction endpoint.

Tests cover:
- Extracting features from valid slice + ROI
- Error handling (missing session, invalid ROI, etc.)
- Feature completeness (25 features)
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.models.session_store import get_session_store, SeriesSlice

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


class TestExtractSeriesEndpoint:
    """Test POST /series/extract endpoint."""

    def test_extract_missing_session(self):
        """Should reject if post-contrast session not found."""
        response = client.post(
            "/series/extract",
            json={
                "post_session_id": "invalid-post-id",
                "pre_session_id": "invalid-pre-id",
                "slice_index": "01",
                "roi": [{"x": 100, "y": 100}, {"x": 150, "y": 100}, {"x": 150, "y": 150}],
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_extract_missing_slice(self):
        """Should reject if slice not found in session."""
        post_id = session_store.create_session("post-contrast")
        pre_id = session_store.create_session("pre-contrast")

        response = client.post(
            "/series/extract",
            json={
                "post_session_id": post_id,
                "pre_session_id": pre_id,
                "slice_index": "99",  # Doesn't exist
                "roi": [{"x": 100, "y": 100}, {"x": 150, "y": 100}, {"x": 150, "y": 150}],
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

        session_store.delete_session(post_id)
        session_store.delete_session(pre_id)

    def test_extract_no_subtracted_array(self):
        """Should reject if subtracted array not available."""
        post_id = session_store.create_session("post-contrast")

        post_session = session_store.get_session(post_id)
        fake_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(np.float32)

        # Add slice WITHOUT subtracted array
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

        response = client.post(
            "/series/extract",
            json={
                "post_session_id": post_id,
                "pre_session_id": "dummy-pre",
                "slice_index": "01",
                "roi": [{"x": 10, "y": 10}, {"x": 90, "y": 10}, {"x": 90, "y": 90}],
            },
        )

        assert response.status_code == 400
        assert "not available" in response.json()["detail"]

        session_store.delete_session(post_id)

    def test_extract_invalid_roi(self):
        """Should reject ROI with <3 points."""
        post_id = session_store.create_session("post-contrast")

        post_session = session_store.get_session(post_id)
        fake_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(np.float32)

        # Add slice WITH subtracted array
        slice_obj = SeriesSlice(
            index="01",
            filename="1-01.dcm",
            pixel_array=fake_array,
            width=100,
            height=100,
            min_intensity=0,
            max_intensity=256,
        )
        slice_obj.raw_subtracted_array = fake_array
        post_session.add_slice(slice_obj)

        # Invalid ROI (only 2 points)
        response = client.post(
            "/series/extract",
            json={
                "post_session_id": post_id,
                "pre_session_id": "dummy-pre",
                "slice_index": "01",
                "roi": [{"x": 10, "y": 10}, {"x": 90, "y": 90}],
            },
        )

        assert response.status_code == 400
        assert "at least 3 points" in response.json()["detail"]

        session_store.delete_session(post_id)

    def test_extract_empty_roi(self):
        """Should reject ROI with no pixels."""
        post_id = session_store.create_session("post-contrast")

        post_session = session_store.get_session(post_id)
        fake_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8).astype(np.float32)

        slice_obj = SeriesSlice(
            index="01",
            filename="1-01.dcm",
            pixel_array=fake_array,
            width=100,
            height=100,
            min_intensity=0,
            max_intensity=256,
        )
        slice_obj.raw_subtracted_array = fake_array
        post_session.add_slice(slice_obj)

        # ROI completely outside image
        response = client.post(
            "/series/extract",
            json={
                "post_session_id": post_id,
                "pre_session_id": "dummy-pre",
                "slice_index": "01",
                "roi": [
                    {"x": 500, "y": 500},
                    {"x": 600, "y": 500},
                    {"x": 600, "y": 600},
                ],
            },
        )

        assert response.status_code == 400
        assert "no pixels" in response.json()["detail"]

        session_store.delete_session(post_id)

    def test_extract_valid_roi_synthetic(self):
        """Test feature extraction with valid ROI on synthetic data."""
        post_id = session_store.create_session("post-contrast")

        post_session = session_store.get_session(post_id)
        fake_array = np.random.randint(50, 200, (512, 512), dtype=np.uint8).astype(np.float32)

        slice_obj = SeriesSlice(
            index="05",
            filename="1-05.dcm",
            pixel_array=fake_array,
            width=512,
            height=512,
            min_intensity=50,
            max_intensity=200,
        )
        slice_obj.raw_subtracted_array = fake_array
        post_session.add_slice(slice_obj)

        # Valid ROI
        response = client.post(
            "/series/extract",
            json={
                "post_session_id": post_id,
                "pre_session_id": "dummy-pre",
                "slice_index": "05",
                "roi": [
                    {"x": 100, "y": 100},
                    {"x": 400, "y": 100},
                    {"x": 400, "y": 400},
                    {"x": 100, "y": 400},
                ],
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["slice_index"] == "05"
        assert data["roi_point_count"] == 4
        assert "features" in data

        # Check features
        features = data["features"]
        assert isinstance(features, dict)
        assert len(features) == 25  # Should have all 25 features

        # Check expected feature names
        expected_features = [
            "Volume", "Area", "MaxDiameter", "SurfaceArea",
            "Sphericity", "Compactness", "Elongation",
            "Mean", "Median", "Min", "Max", "Std", "Skewness", "Kurtosis", "Entropy",
            "GLCM_Contrast", "GLCM_Correlation", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Entropy",
            "SRE", "LRE", "GLN", "LiverEntropy", "TumorLiverContrast"
        ]

        for fname in expected_features:
            assert fname in features
            assert isinstance(features[fname], (int, float))

        session_store.delete_session(post_id)

    @pytest.mark.skipif(
        Path(
            "HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910"
        ).exists()
        is False,
        reason="HCC_010 test data not found",
    )
    def test_extract_real_dicom_workflow(self, hcc_010_dicom_files):
        """Test full workflow: upload → subtract → extract."""
        if not hcc_010_dicom_files:
            pytest.skip("HCC_010 data not available")

        test_files = hcc_010_dicom_files[:2]

        # Step 1: Upload both series
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

            # Step 2: Subtract
            subtract_response = client.post(
                "/series/subtract",
                json={
                    "post_session_id": post_id,
                    "pre_session_id": pre_id,
                },
            )

            assert subtract_response.status_code == 200
            subtracted_data = subtract_response.json()
            assert subtracted_data["total"] == 2

            # Step 3: Extract features
            first_slice_idx = subtracted_data["subtracted_series"][0]["index"]

            extract_response = client.post(
                "/series/extract",
                json={
                    "post_session_id": post_id,
                    "pre_session_id": pre_id,
                    "slice_index": first_slice_idx,
                    "roi": [
                        {"x": 150, "y": 150},
                        {"x": 350, "y": 150},
                        {"x": 350, "y": 350},
                        {"x": 150, "y": 350},
                    ],
                },
            )

            assert extract_response.status_code == 200

            extract_data = extract_response.json()
            assert extract_data["slice_index"] == first_slice_idx
            assert extract_data["roi_point_count"] == 4
            assert len(extract_data["features"]) == 25

            session_store.delete_session(post_id)
            session_store.delete_session(pre_id)

        finally:
            for _, (_, f) in files:
                try:
                    f.close()
                except:
                    pass
