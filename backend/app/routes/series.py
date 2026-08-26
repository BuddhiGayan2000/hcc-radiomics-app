"""
API routes for DICOM series upload and processing.

Endpoints:
- POST /upload/series - Upload entire DICOM folder
- POST /subtract/series - Compute subtracted series
- POST /extract-from-series - Extract features from selected slice
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Tuple

import numpy as np

from app.models.dicom_handler import (
    parse_dicom_file,
    DICOMParseError,
    assign_slice_indices,
    best_slice_position,
    subtract_images,
    array_to_png_base64,
    apply_windowing,
    normalize_for_display,
)
from app.models.session_store import get_session_store, SeriesSlice
from app.schemas.series import (
    UploadSeriesResponse,
    SeriesSliceInfo,
    SubtractSeriesRequest,
    SubtractSeriesResponse,
    SubtractedSliceInfo,
    ExtractFromSeriesRequest,
    ExtractFeaturesResponse,
)
from app.models.feature_extractor import extract_all_features, rasterize_mask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/series", tags=["series"])

# Get session store
session_store = get_session_store()

# Max physical distance (mm) between a post-contrast slice and a pre-contrast
# slice for them to be considered the same anatomical location.
ALIGNMENT_TOLERANCE_MM = 5.0


def _align_slices_by_position(
    post_slices: List[SeriesSlice],
    pre_slices: List[SeriesSlice],
    tolerance_mm: float = ALIGNMENT_TOLERANCE_MM,
) -> Tuple[List[Tuple[SeriesSlice, SeriesSlice]], str]:
    """
    Pair each post-contrast slice with its closest pre-contrast counterpart.

    Matches by physical slice position (SliceLocation, or the z component of
    ImagePositionPatient) within tolerance_mm — the only correspondence
    that's true regardless of filename or either series' slice count. Falls
    back to ordinal pairing (i-th post to i-th pre, using each series' own
    DICOM-derived order — see assign_slice_indices) only when position
    metadata is missing from every slice in either series.

    Returns:
        (pairs, method) where method is "position" or "ordinal".
    """
    post_positions = [(s, best_slice_position(s.metadata)) for s in post_slices]
    pre_positions = [(s, best_slice_position(s.metadata)) for s in pre_slices]

    have_positions = all(p is not None for _, p in post_positions) and all(
        p is not None for _, p in pre_positions
    )

    if not have_positions:
        return list(zip(post_slices, pre_slices)), "ordinal"

    pairs = []
    used_pre_indices = set()
    for post_slice, post_pos in post_positions:
        best_pre, best_dist = None, None
        for pre_slice, pre_pos in pre_positions:
            if pre_slice.index in used_pre_indices:
                continue
            dist = abs(post_pos - pre_pos)
            if best_dist is None or dist < best_dist:
                best_pre, best_dist = pre_slice, dist
        if best_pre is not None and best_dist <= tolerance_mm:
            pairs.append((post_slice, best_pre))
            used_pre_indices.add(best_pre.index)

    return pairs, "position"


@router.post(
    "/upload",
    response_model=UploadSeriesResponse,
    summary="Upload DICOM series folder",
    description="Upload multiple DICOM files (entire pre/post-contrast series). "
                "Returns session ID to use for subtraction and feature extraction.",
)
async def upload_series(
    phase: str = Form(..., description="Contrast phase: 'post-contrast' or 'pre-contrast'"),
    files: List[UploadFile] = File(..., description="DICOM files (.dcm format)"),
) -> UploadSeriesResponse:
    """
    Upload an entire DICOM series folder.

    Accepts multiple DICOM files, parses them, and stores in a session.
    Returns a session ID to use for subsequent operations.

    Args:
        phase: "post-contrast" or "pre-contrast"
        files: List of DICOM files to upload

    Returns:
        UploadSeriesResponse with session_id and slice metadata

    Raises:
        HTTPException 400: Invalid phase, no files, or parsing error
        HTTPException 413: Too many files or file too large
    """

    # Validate phase
    if phase not in ["post-contrast", "pre-contrast"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid phase '{phase}'. Must be 'post-contrast' or 'pre-contrast'.",
        )

    # Validate file count
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    if len(files) > 100:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files ({len(files)}). Maximum 100 files per upload.",
        )

    logger.info(f"Uploading {len(files)} DICOM files for {phase} phase")

    # Create session
    session_id = session_store.create_session(phase)
    session = session_store.get_session(session_id)

    # Pass 1: parse every file. Filenames are never inspected here — real
    # DICOM exports use every naming convention imaginable ("1-01.dcm",
    # "1-012.dcm", "IMG1000012.dcm", ...), so a file is only rejected if its
    # DICOM content itself can't be read.
    import os
    import tempfile

    parsed: List[Tuple[np.ndarray, dict]] = []
    errors = []

    for file in files:
        try:
            content = await file.read()

            with tempfile.NamedTemporaryFile(
                suffix=".dcm", delete=False, prefix="dicom_"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                pixel_array, metadata = parse_dicom_file(tmp_path)
                metadata["filename"] = file.filename  # preserve the original name, not the temp path's
                parsed.append((pixel_array, metadata))
            finally:
                os.unlink(tmp_path)

        except DICOMParseError as e:
            logger.warning(f"Failed to parse {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")

    if not parsed:
        session_store.delete_session(session_id)
        error_msg = "; ".join(errors) if errors else "Could not parse any DICOM files"
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse any DICOM files. {error_msg}",
        )

    if errors:
        logger.warning(f"Upload partially succeeded: {len(errors)} files failed to parse")

    # Pass 2: order slices using each file's own DICOM position metadata
    # (SliceLocation / ImagePositionPatient > InstanceNumber > filename as a
    # last resort) rather than any assumed filename pattern.
    metadata_list = [m for _, m in parsed]
    indices = assign_slice_indices(metadata_list)

    series_uids = {m["series_instance_uid"] for _, m in parsed if "series_instance_uid" in m}
    if len(series_uids) > 1:
        logger.warning(
            f"Upload for {phase} contains {len(series_uids)} distinct SeriesInstanceUIDs "
            f"({len(parsed)} files) — this folder may mix multiple series/phases, which can "
            f"produce a misleading subtraction. Consider uploading a single series only."
        )

    for (pixel_array, metadata), slice_idx in zip(parsed, indices):
        session.add_slice(
            SeriesSlice(
                index=slice_idx,
                filename=metadata["filename"],
                pixel_array=pixel_array,
                width=metadata["width"],
                height=metadata["height"],
                min_intensity=metadata["min_intensity"],
                max_intensity=metadata["max_intensity"],
                metadata=metadata,
            )
        )

    # Build response
    slices = session.get_sorted_slices()
    response = UploadSeriesResponse(
        session_id=session_id,
        phase=phase,
        slice_count=len(slices),
        slices=[
            SeriesSliceInfo(
                index=s.index,
                filename=s.filename,
                width=s.width,
                height=s.height,
            )
            for s in slices
        ],
    )

    logger.info(
        f"Upload successful: {len(slices)} slices, session_id={session_id[:8]}..."
    )

    return response


@router.get(
    "/session/{session_id}",
    summary="Get session info",
    description="Get information about an uploaded series session.",
)
async def get_session_info(session_id: str):
    """
    Get information about a series session.

    Args:
        session_id: Session ID from upload response

    Returns:
        Session info (phase, slice count, indices)

    Raises:
        HTTPException 404: Session not found or expired
    """
    info = session_store.get_session_info(session_id)

    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id[:8]}...' not found or expired.",
        )

    return info


@router.delete(
    "/session/{session_id}",
    summary="Delete session",
    description="Delete a session and free memory.",
)
async def delete_session(session_id: str):
    """
    Delete a session.

    Args:
        session_id: Session ID to delete

    Returns:
        Success message
    """
    success = session_store.delete_session(session_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id[:8]}...' not found.",
        )

    logger.info(f"Deleted session {session_id[:8]}...")
    return {"status": "deleted", "session_id": session_id}


@router.post(
    "/subtract",
    response_model=SubtractSeriesResponse,
    summary="Compute subtracted DICOM series",
    description="Align pre/post-contrast series and compute subtraction (Post - Pre). "
                "Returns base64-encoded PNG images for all subtracted slices.",
)
async def subtract_series(request: SubtractSeriesRequest) -> SubtractSeriesResponse:
    """
    Compute subtracted images from pre/post-contrast DICOM series.

    Aligns slices by physical position (never by filename — see
    _align_slices_by_position), computes Post - Pre for each matched pair,
    and returns as base64-encoded PNG images.

    Args:
        request: Contains post_session_id and pre_session_id

    Returns:
        SubtractSeriesResponse with subtracted images (base64 PNG)

    Raises:
        HTTPException 404: Session not found or expired
        HTTPException 400: Series cannot be aligned or other processing error
    """

    # Retrieve sessions
    post_session = session_store.get_session(request.post_session_id)
    pre_session = session_store.get_session(request.pre_session_id)

    if not post_session:
        raise HTTPException(
            status_code=404,
            detail=f"Post-contrast session '{request.post_session_id[:8]}...' not found or expired.",
        )

    if not pre_session:
        raise HTTPException(
            status_code=404,
            detail=f"Pre-contrast session '{request.pre_session_id[:8]}...' not found or expired.",
        )

    # Verify both sessions have data
    if post_session.slice_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Post-contrast session has no slices.",
        )

    if pre_session.slice_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Pre-contrast session has no slices.",
        )

    logger.info(
        f"Computing subtraction: "
        f"{post_session.slice_count()} post slices vs "
        f"{pre_session.slice_count()} pre slices"
    )

    pairs, alignment_method = _align_slices_by_position(
        post_session.get_sorted_slices(), pre_session.get_sorted_slices()
    )

    if not pairs:
        raise HTTPException(
            status_code=400,
            detail="No matching slices between post and pre series — none of the "
                   "post-contrast slices had a pre-contrast slice at the same "
                   f"physical position within {ALIGNMENT_TOLERANCE_MM}mm.",
        )

    if alignment_method == "ordinal":
        logger.warning(
            "Neither series has usable slice-position metadata; aligning "
            "post/pre slices by ordinal position instead — verify the "
            "resulting slice pairs look anatomically correct."
        )
    logger.info(f"Aligned {len(pairs)} slice pairs by {alignment_method}")

    # Compute subtraction for each matching pair
    subtracted_slices = []
    errors = []

    for post_slice, pre_slice in pairs:
        idx = post_slice.index
        try:
            # Ensure same dimensions
            if post_slice.pixel_array.shape != pre_slice.pixel_array.shape:
                errors.append(
                    f"Slice {idx}: shape mismatch "
                    f"(post {post_slice.pixel_array.shape} vs "
                    f"pre {pre_slice.pixel_array.shape})"
                )
                continue

            # Apply windowing to both (liver CT standard: window=400, center=50)
            post_windowed = apply_windowing(post_slice.pixel_array)
            pre_windowed = apply_windowing(pre_slice.pixel_array)

            # Compute subtraction
            subtracted_windowed = subtract_images(
                post_windowed.astype(np.float32),
                pre_windowed.astype(np.float32),
            )

            # Also compute on raw data for feature extraction later
            subtracted_raw = subtract_images(
                post_slice.pixel_array.astype(np.float32),
                pre_slice.pixel_array.astype(np.float32),
            )

            # Normalize for display
            subtracted_display = normalize_for_display(subtracted_windowed)

            # Encode as base64 PNG
            image_b64 = array_to_png_base64(subtracted_display)

            # Store raw array in post_slice for later feature extraction
            # (we'll retrieve it by session_id + slice_index)
            post_slice.raw_subtracted_array = subtracted_raw

            subtracted_slices.append(
                SubtractedSliceInfo(
                    index=idx,
                    filename=post_slice.filename,
                    width=post_slice.width,
                    height=post_slice.height,
                    image_data_b64=image_b64,
                )
            )

            logger.debug(f"Subtracted slice {idx}: {post_slice.filename}")

        except Exception as e:
            logger.warning(f"Failed to subtract slice {idx}: {e}")
            errors.append(f"Slice {idx}: {str(e)}")

    # Check if any subtractions succeeded
    if not subtracted_slices:
        error_msg = "; ".join(errors) if errors else "Unknown error"
        raise HTTPException(
            status_code=400,
            detail=f"Failed to compute any subtractions. {error_msg}",
        )

    # Log warnings if some slices failed
    if errors:
        logger.warning(
            f"Subtraction partially succeeded: {len(errors)} slices failed"
        )

    response = SubtractSeriesResponse(
        subtracted_series=subtracted_slices,
        total=len(subtracted_slices),
    )

    logger.info(
        f"Subtraction complete: {len(subtracted_slices)} slices computed"
    )

    return response


@router.post(
    "/extract",
    response_model=ExtractFeaturesResponse,
    summary="Extract radiomic features from selected slice",
    description="Extract 25 radiomic features from a selected subtracted slice "
                "using a user-drawn ROI polygon. Returns feature dict ready for prediction.",
)
async def extract_from_series(request: ExtractFromSeriesRequest) -> ExtractFeaturesResponse:
    """
    Extract radiomic features from a selected subtracted slice.

    Retrieves the pre-computed subtracted image, applies the ROI mask,
    and extracts all 25 radiomic features from the masked region.

    Args:
        request: Contains session IDs, slice index, and ROI polygon points

    Returns:
        ExtractFeaturesResponse with 25 extracted features

    Raises:
        HTTPException 404: Session not found or slice not found
        HTTPException 400: Invalid ROI, processing error
    """

    # Retrieve post-contrast session (stores the subtracted arrays)
    post_session = session_store.get_session(request.post_session_id)

    if not post_session:
        raise HTTPException(
            status_code=404,
            detail=f"Post-contrast session '{request.post_session_id[:8]}...' not found or expired.",
        )

    # Get the selected slice
    selected_slice = post_session.get_slice(request.slice_index)

    if not selected_slice:
        raise HTTPException(
            status_code=404,
            detail=f"Slice '{request.slice_index}' not found in post-contrast session.",
        )

    # Check that subtracted array exists (computed during subtraction step)
    if selected_slice.raw_subtracted_array is None:
        raise HTTPException(
            status_code=400,
            detail=f"Subtracted image for slice '{request.slice_index}' not available. "
                   "Run /series/subtract first.",
        )

    # Validate ROI
    if not request.roi or len(request.roi) < 3:
        raise HTTPException(
            status_code=400,
            detail="ROI must have at least 3 points.",
        )

    logger.info(
        f"Extracting features from slice {request.slice_index} "
        f"with {len(request.roi)}-point ROI"
    )

    try:
        # Rasterize ROI polygon to binary mask
        mask = rasterize_mask(
            request.roi,
            selected_slice.width,
            selected_slice.height,
        )

        # Check if mask has any pixels
        mask_pixel_count = int(np.sum(mask))
        if mask_pixel_count == 0:
            raise HTTPException(
                status_code=400,
                detail="ROI has no pixels. Try drawing a larger ROI.",
            )

        # Extract features from subtracted image
        features = extract_all_features(
            selected_slice.raw_subtracted_array,
            mask,
            selected_slice.width,
            selected_slice.height,
        )

        logger.info(f"Extracted {len(features)} features from slice {request.slice_index}")

        response = ExtractFeaturesResponse(
            features=features,
            slice_index=request.slice_index,
            roi_point_count=len(request.roi),
        )

        return response

    except Exception as e:
        logger.error(f"Feature extraction failed for slice {request.slice_index}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Feature extraction failed: {str(e)}",
        )
