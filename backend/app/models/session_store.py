"""
Session storage for DICOM series uploads.

Each upload session stores:
- Uploaded DICOM pixel data (as numpy arrays)
- Metadata about each slice
- Slice index mapping

Sessions are identified by UUID and cleaned up after prediction or timeout.
"""

import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time


@dataclass
class SeriesSlice:
    """Metadata and pixel data for a single slice in a series."""

    index: str  # Slice index (01, 02, ..., 20)
    filename: str  # Original filename (1-01.dcm)
    pixel_array: np.ndarray  # Raw pixel data
    width: int  # Image width
    height: int  # Image height
    min_intensity: float  # Min pixel value
    max_intensity: float  # Max pixel value
    metadata: Dict = field(default_factory=dict)  # DICOM metadata (patient_id, etc.)
    timestamp: float = field(default_factory=time.time)
    raw_subtracted_array: Optional[np.ndarray] = None  # Subtracted image (computed later)


@dataclass
class SeriesSession:
    """Session for a DICOM series upload."""

    session_id: str
    phase: str  # "post-contrast" or "pre-contrast"
    slices: Dict[str, SeriesSlice] = field(default_factory=dict)  # Keyed by index
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def add_slice(self, slice_obj: SeriesSlice) -> None:
        """Add a slice to this session."""
        self.slices[slice_obj.index] = slice_obj
        self.last_accessed = time.time()

    def get_slice(self, index: str) -> Optional[SeriesSlice]:
        """Get a slice by index."""
        self.last_accessed = time.time()
        return self.slices.get(index)

    def get_sorted_slices(self) -> List[SeriesSlice]:
        """Get all slices sorted by index."""
        self.last_accessed = time.time()
        return sorted(self.slices.values(), key=lambda s: s.index)

    def slice_count(self) -> int:
        """Number of slices in session."""
        return len(self.slices)

    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """Check if session has expired (default: 1 hour)."""
        return (time.time() - self.last_accessed) > timeout_seconds


class SessionStore:
    """In-memory store for DICOM series upload sessions."""

    def __init__(self, cleanup_interval_seconds: int = 600, timeout_seconds: int = 3600):
        """
        Initialize session store.

        Args:
            cleanup_interval_seconds: How often to clean up expired sessions (default 10 min)
            timeout_seconds: How long before a session expires if not accessed (default 1 hour)
        """
        self.sessions: Dict[str, SeriesSession] = {}
        self.cleanup_interval = cleanup_interval_seconds
        self.timeout = timeout_seconds
        self.last_cleanup = time.time()

    def create_session(self, phase: str) -> str:
        """
        Create a new upload session.

        Args:
            phase: "post-contrast" or "pre-contrast"

        Returns:
            Session ID (UUID)
        """
        self._cleanup_if_needed()

        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SeriesSession(
            session_id=session_id,
            phase=phase,
        )
        return session_id

    def get_session(self, session_id: str) -> Optional[SeriesSession]:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        if session and session.is_expired(self.timeout):
            del self.sessions[session_id]
            return None
        return session

    def add_slice_to_session(self, session_id: str, slice_obj: SeriesSlice) -> bool:
        """
        Add a slice to a session.

        Args:
            session_id: Session ID
            slice_obj: Slice to add

        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.add_slice(slice_obj)
        return True

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get summary info about a session.

        Returns:
            Dict with phase, slice_count, indices, or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session_id,
            "phase": session.phase,
            "slice_count": session.slice_count(),
            "slices": [
                {
                    "index": s.index,
                    "filename": s.filename,
                    "width": s.width,
                    "height": s.height,
                }
                for s in session.get_sorted_slices()
            ],
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and free memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def _cleanup_if_needed(self) -> None:
        """Remove expired sessions if cleanup interval has passed."""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return

        expired_ids = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.timeout)
        ]

        for sid in expired_ids:
            del self.sessions[sid]

        self.last_cleanup = time.time()

    def cleanup_all(self) -> int:
        """Clean up all expired sessions. Returns count removed."""
        self._cleanup_if_needed()
        return len(self.sessions)


# Global session store (singleton)
_session_store = SessionStore()


def get_session_store() -> SessionStore:
    """Get the global session store."""
    return _session_store
