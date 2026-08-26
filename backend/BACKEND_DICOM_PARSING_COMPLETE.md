# Backend DICOM Parsing — Complete ✓

**Status:** Implementation Complete & Verified  
**Date:** 2026-08-23  
**Tests:** 22/22 Passed | Real Data Verification: Passed

> **Update (2026-08-26):** `extract_slice_index` and `align_series`
> (filename-regex based) described below have been removed — real DICOM
> exports use every naming convention imaginable, not just "1-XX.dcm".
> Ordering and cross-series alignment now use `slice_sort_key` /
> `assign_slice_indices` / `best_slice_position` in this same module, based
> on each slice's own `SliceLocation` / `ImagePositionPatient` /
> `InstanceNumber` metadata. Everything else below (windowing, subtraction,
> PNG encoding) is unchanged.

---

## What Was Implemented

### Module: `app/models/dicom_handler.py` (460 lines)

A complete DICOM parsing and series processing module with:

#### Core Functions

1. **`parse_dicom_file(file_path)`**
   - Reads DICOM files with pydicom
   - Extracts pixel data as numpy array
   - Returns metadata (shape, pixel range, patient info)

2. **`apply_windowing(pixel_array, window_width=400, window_center=50)`**
   - Applies CT windowing/leveling (HU → display range)
   - Standard liver CT params: window=400, center=50
   - Returns uint8 array [0, 255]

3. **`extract_slice_index(filename)`**
   - Extracts slice index from DICOM filename (e.g., "1-05.dcm" → "05")
   - Enables pre/post series matching

4. **`align_series(post_files, pre_files)`**
   - Matches pre-contrast and post-contrast DICOMs by filename
   - Returns list of (post_path, pre_path, slice_index) tuples
   - Validates that both series have same slices

5. **`subtract_images(post_array, pre_array)`**
   - Computes Post - Pre subtraction
   - Clips negative values to 0
   - Returns subtracted array

6. **`normalize_for_display(array)`**
   - Normalizes any array to [0, 255] uint8 range
   - Handles constant arrays (maps to 128)
   - Suitable for PNG encoding

7. **`array_to_png_base64(array)`**
   - Converts numpy array to PNG
   - Encodes as base64 data URL
   - Example: `data:image/png;base64,iVBORw0KG...`

8. **`process_dicom_series(post_files, pre_files)`**
   - Complete pipeline function
   - Loads, aligns, subtracts, and encodes all slices
   - Returns list of results with base64-encoded images

#### Exceptions

- **`DICOMParseError`** — Raised when file cannot be read
- **`SeriesAlignmentError`** — Raised when pre/post cannot be aligned

---

## Testing

### Unit Tests: `tests/test_dicom_handler.py` (22 tests)

**Test Coverage:**
- ✓ Filename parsing (valid/invalid/case-insensitive)
- ✓ Windowing/leveling (clipping, default params)
- ✓ Image subtraction (element-wise, clipping, shape validation)
- ✓ Normalization (basic, constant arrays, empty arrays)
- ✓ Base64 PNG encoding (basic, dtype conversion, PNG validation)
- ✓ Series alignment (simple, partial overlap, no match, sorted output)
- ✓ DICOM file parsing (real HCC_010 files, nonexistent files)
- ✓ Integration tests (synthetic pipeline, real series alignment)

**Result:** All 22 tests passed ✓

### Real Data Verification: `scripts/verify_dicom_handler.py`

Tested with actual HCC_010 DICOM dataset:

```
✓ Found HCC_010 dataset
✓ Found 25 DICOM files
✓ Parsed 3 DICOM files (512×512, -1024 to +1218 HU range)
✓ Extracted 5 slice indices (01, 02, 03, 04, 05)
✓ Aligned 5 slice pairs
✓ Computed subtraction (range 0-1380)
✓ Normalized to [0, 255]
✓ Encoded as base64 PNG (99.9 KB per slice)
```

**Result:** Verification PASSED ✓

---

## Technical Details

### DICOM Windowing

Standard for liver CT:
- **Window Width:** 400 HU
- **Window Center:** 50 HU
- **Display Range:** [50-200, 50+200] = [-150, 250] HU → [0, 255]

This emphasizes liver tissue and tumors, suppresses background.

### Series Alignment

Matching by filename pattern: `1-XX.dcm`
- Post-contrast: `1-01.dcm`, `1-02.dcm`, ..., `1-20.dcm`
- Pre-contrast: `1-01.dcm`, `1-02.dcm`, ..., `1-20.dcm`
- Index "01" in both → matched pair

Simple, robust, assumes DICOM series already aligned anatomically (standard in PACS).

### Image Subtraction

```
subtracted = max(0, post - pre)
```

Regions brighter in post-contrast (tumors that enhance with contrast) show as bright areas in subtraction. Normal tissue cancels out.

### Base64 Encoding

Each subtracted image (512×512) encodes to ~100KB as PNG data URL:
```
data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAA...
```

Suitable for transmission to frontend via JSON response.

---

## Dependencies Added

Updated `backend/requirements.txt`:
```
pydicom==2.4.4      # DICOM file parsing
Pillow==10.1.0      # PNG encoding
```

Both already compatible with existing dependencies (numpy, scikit-learn).

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `backend/app/models/dicom_handler.py` | 460 | Core DICOM parsing module |
| `backend/tests/test_dicom_handler.py` | 380 | Unit tests + integration tests |
| `scripts/verify_dicom_handler.py` | 160 | Real data verification script |

---

## Next Steps

The DICOM handler is **ready for integration** into backend endpoints:

### Phase 2: Backend Series Endpoints

Need to create `app/routes/series.py` with:

1. **`POST /upload/series`**
   - Accept multi-file DICOM upload
   - Parse with dicom_handler
   - Return slice metadata

2. **`POST /subtract/series`**
   - Match and align pre/post
   - Compute all subtractions
   - Return base64-encoded images

3. **`POST /extract-from-series`**
   - Receive selected slice + ROI
   - Extract 25 features from subtracted image
   - Return features dict

### Phase 3: Frontend Integration

- Multi-file upload UI
- Series browser (thumbnails + full viewer)
- ROI drawing on selected slice
- Feature extraction & prediction

---

## Verification Commands

To re-run verification:

```bash
cd ./hcc-radiomics-app/backend

# Run all DICOM handler tests
python -m pytest tests/test_dicom_handler.py -v

# Verify with real HCC_010 data
PYTHONIOENCODING=utf-8 python ../scripts/verify_dicom_handler.py
```

---

## Success Criteria Met

- ✓ Parse actual HCC_010 DICOM files
- ✓ Apply medical-grade windowing
- ✓ Align pre/post series by filename
- ✓ Compute subtraction (Post - Pre)
- ✓ Encode as base64 PNG for transmission
- ✓ Handle edge cases (misaligned, missing slices)
- ✓ Full test coverage (22 tests)
- ✓ Real data verification passed

---

## Summary

**Backend DICOM parsing is complete and ready for integration.**

The module correctly:
1. Reads DICOM files from HCC_010 dataset
2. Applies medical windowing (liver CT standard)
3. Aligns pre-contrast and post-contrast series
4. Computes subtracted images
5. Encodes for transmission to frontend

All 22 unit tests pass. Real data verification with HCC_010 DICOMs passed. Ready to implement backend endpoints in Phase 2.

**Proceed to:** Backend series endpoints (`/upload/series`, `/subtract/series`, `/extract-from-series`)
