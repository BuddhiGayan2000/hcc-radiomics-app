"""
Request and response schemas for DICOM series endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class SeriesSliceInfo(BaseModel):
    """Information about a single slice in a series."""

    index: str = Field(..., description="Slice index (01, 02, ..., 20)")
    filename: str = Field(..., description="Original DICOM filename")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class UploadSeriesResponse(BaseModel):
    """Response from POST /upload/series endpoint."""

    session_id: str = Field(
        ..., description="Unique session ID for this upload (store for later use)"
    )
    phase: str = Field(
        ..., description="Contrast phase (post-contrast or pre-contrast)"
    )
    slice_count: int = Field(..., description="Number of slices uploaded")
    slices: List[SeriesSliceInfo] = Field(
        ..., description="List of slices with metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "phase": "post-contrast",
                "slice_count": 20,
                "slices": [
                    {"index": "01", "filename": "1-01.dcm", "width": 512, "height": 512},
                    {"index": "02", "filename": "1-02.dcm", "width": 512, "height": 512},
                ],
            }
        }


class SubtractSeriesRequest(BaseModel):
    """Request for POST /subtract/series endpoint."""

    post_session_id: str = Field(..., description="Session ID from post-contrast upload")
    pre_session_id: str = Field(..., description="Session ID from pre-contrast upload")


class SubtractedSliceInfo(BaseModel):
    """Information about a subtracted slice."""

    index: str = Field(..., description="Slice index (01, 02, ..., 20)")
    filename: str = Field(..., description="Original DICOM filename")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    image_data_b64: str = Field(
        ..., description="Base64-encoded PNG data URL (data:image/png;base64,...)"
    )


class SubtractSeriesResponse(BaseModel):
    """Response from POST /subtract/series endpoint."""

    subtracted_series: List[SubtractedSliceInfo] = Field(
        ..., description="List of subtracted slices with base64 images"
    )
    total: int = Field(..., description="Total number of subtracted slices")

    class Config:
        json_schema_extra = {
            "example": {
                "subtracted_series": [
                    {
                        "index": "01",
                        "filename": "1-01.dcm",
                        "width": 512,
                        "height": 512,
                        "image_data_b64": "data:image/png;base64,iVBORw0KGgo...",
                    }
                ],
                "total": 20,
            }
        }


class ExtractFromSeriesRequest(BaseModel):
    """Request for POST /extract-from-series endpoint."""

    post_session_id: str = Field(..., description="Session ID from post-contrast upload")
    pre_session_id: str = Field(..., description="Session ID from pre-contrast upload")
    slice_index: str = Field(..., description="Slice to extract from (01, 02, ..., 20)")
    roi: List[Dict[str, float]] = Field(
        ..., description="ROI polygon as list of {x, y} points"
    )


class ExtractFeaturesResponse(BaseModel):
    """Response from POST /extract-from-series endpoint."""

    features: Dict[str, float] = Field(
        ..., description="25 extracted radiomic features"
    )
    slice_index: str = Field(..., description="Slice that was analyzed")
    roi_point_count: int = Field(..., description="Number of points in ROI")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Volume": 3895.0,
                    "Area": 1558.0,
                    "MaxDiameter": 48.28,
                    "Mean": 285.69,
                },
                "slice_index": "05",
                "roi_point_count": 12,
            }
        }
