"""
Tests for DICOM series upload endpoint.

Tests cover:
- Uploading valid DICOM series
- Error handling (invalid phase, no files, parse errors)
- Session management
- Slice metadata extraction
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

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


class TestUploadSeriesEndpoint:
    """Test POST /series/upload endpoint."""

    def test_upload_invalid_phase(self):
        """Should reject invalid phase."""
        response = client.post(
            "/series/upload",
            data={"phase": "invalid-phase"},
            files={"files": ("test.dcm", b"fake data")},
        )

        assert response.status_code == 400
        assert "Invalid phase" in response.json()["detail"]

    def test_upload_no_files(self):
        """Should reject upload with no files."""
        response = client.post(
            "/series/upload",
            data={"phase": "post-contrast"},
        )

        # FastAPI returns 422 (Unprocessable Entity) for missing required parameter
        assert response.status_code == 422

    def test_upload_too_many_files(self):
        """Should reject upload with >100 files."""
        files = [
            ("files", (f"file{i}.dcm", b"fake data"))
            for i in range(101)
        ]

        response = client.post(
            "/series/upload",
            data={"phase": "post-contrast"},
            files=files,
        )

        assert response.status_code == 413
        assert "Too many files" in response.json()["detail"]

    def test_upload_with_invalid_dcm_file(self):
        """Should reject invalid DICOM files."""
        response = client.post(
            "/series/upload",
            data={"phase": "post-contrast"},
            files={"files": ("not_a_dicom.dcm", b"not a real DICOM file")},
        )

        assert response.status_code == 400
        assert "Failed to parse any DICOM files" in response.json()["detail"]

    @pytest.mark.skipif(
        Path(
            "HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910"
        ).exists()
        is False,
        reason="HCC_010 test data not found",
    )
    def test_upload_real_dicom_files(self, hcc_010_dicom_files):
        """Test uploading real HCC_010 DICOM files."""
        if not hcc_010_dicom_files:
            pytest.skip("HCC_010 data not available")

        # Use first 5 files for testing
        test_files = hcc_010_dicom_files[:5]

        # Prepare files for upload
        files = [
            ("files", (dcm_path.name, open(dcm_path, "rb")))
            for dcm_path in test_files
        ]

        try:
            response = client.post(
                "/series/upload",
                data={"phase": "post-contrast"},
                files=files,
            )

            # Should succeed
            assert response.status_code == 200

            data = response.json()
            assert "session_id" in data
            assert data["phase"] == "post-contrast"
            assert data["slice_count"] == 5

            # Check slice info
            assert len(data["slices"]) == 5
            for slice_info, dcm_file in zip(data["slices"], sorted(test_files)):
                assert slice_info["filename"] == dcm_file.name
                assert slice_info["width"] == 512
                assert slice_info["height"] == 512
                assert slice_info["index"].isdigit()

        finally:
            # Close file handles
            for _, (_, f) in files:
                f.close()

    @pytest.mark.skipif(
        Path(
            "HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910"
        ).exists()
        is False,
        reason="HCC_010 test data not found",
    )
    def test_upload_and_query_session(self, hcc_010_dicom_files):
        """Test uploading and then querying session info."""
        if not hcc_010_dicom_files:
            pytest.skip("HCC_010 data not available")

        test_files = hcc_010_dicom_files[:3]

        files = [
            ("files", (dcm_path.name, open(dcm_path, "rb")))
            for dcm_path in test_files
        ]

        try:
            # Upload
            upload_response = client.post(
                "/series/upload",
                data={"phase": "pre-contrast"},
                files=files,
            )

            assert upload_response.status_code == 200
            session_id = upload_response.json()["session_id"]

            # Query session info
            info_response = client.get(f"/series/session/{session_id}")

            assert info_response.status_code == 200
            info = info_response.json()
            assert info["session_id"] == session_id
            assert info["phase"] == "pre-contrast"
            assert info["slice_count"] == 3

        finally:
            for _, (_, f) in files:
                f.close()

    @pytest.mark.skipif(
        Path(
            "HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/2.000000-PRE LIVER-34910"
        ).exists()
        is False,
        reason="HCC_010 test data not found",
    )
    def test_upload_delete_session(self, hcc_010_dicom_files):
        """Test uploading and then deleting session."""
        if not hcc_010_dicom_files:
            pytest.skip("HCC_010 data not available")

        test_files = hcc_010_dicom_files[:2]

        files = [
            ("files", (dcm_path.name, open(dcm_path, "rb")))
            for dcm_path in test_files
        ]

        try:
            # Upload
            upload_response = client.post(
                "/series/upload",
                data={"phase": "post-contrast"},
                files=files,
            )

            session_id = upload_response.json()["session_id"]

            # Delete session
            delete_response = client.delete(f"/series/session/{session_id}")

            assert delete_response.status_code == 200
            assert delete_response.json()["status"] == "deleted"

            # Verify session is gone
            info_response = client.get(f"/series/session/{session_id}")
            assert info_response.status_code == 404

        finally:
            for _, (_, f) in files:
                f.close()

    def test_query_nonexistent_session(self):
        """Should return 404 for nonexistent session."""
        response = client.get("/series/session/invalid-session-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_nonexistent_session(self):
        """Should return 404 when deleting nonexistent session."""
        response = client.delete("/series/session/invalid-session-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestSessionStore:
    """Test session storage functionality."""

    def test_create_and_retrieve_session(self):
        """Test creating and retrieving a session."""
        session_id = session_store.create_session("post-contrast")

        session = session_store.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.phase == "post-contrast"
        assert session.slice_count() == 0

        # Cleanup
        session_store.delete_session(session_id)

    def test_session_expiration(self):
        """Test that expired sessions are cleaned up."""
        # Create session with very short timeout
        session_id = session_store.create_session("pre-contrast")

        # Force expiration
        session = session_store.get_session(session_id)
        session.last_accessed = 0  # Set to epoch (way past)

        # Should return None (expired and deleted)
        expired = session_store.get_session(session_id)
        assert expired is None

    def test_multiple_sessions(self):
        """Test managing multiple sessions."""
        session_id1 = session_store.create_session("post-contrast")
        session_id2 = session_store.create_session("pre-contrast")

        session1 = session_store.get_session(session_id1)
        session2 = session_store.get_session(session_id2)

        assert session1.phase == "post-contrast"
        assert session2.phase == "pre-contrast"

        # Cleanup
        session_store.delete_session(session_id1)
        session_store.delete_session(session_id2)
