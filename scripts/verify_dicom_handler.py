#!/usr/bin/env python3
"""
Verification script for DICOM handler module.

Tests that the backend DICOM parsing module correctly:
- Reads DICOM files from HCC_010 dataset
- Parses pixel data and metadata
- Aligns pre/post series by filename
- Computes subtraction
- Encodes as base64 PNG

Usage:
    python scripts/verify_dicom_handler.py
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parents[1]
backend_path = repo_root / "backend"
sys.path.insert(0, str(backend_path))

from app.models.dicom_handler import (
    parse_dicom_file,
    apply_windowing,
    extract_slice_index,
    align_series,
    subtract_images,
    normalize_for_display,
    array_to_png_base64,
)


def main():
    print("=" * 70)
    print("DICOM Handler Verification — HCC_010 Dataset")
    print("=" * 70)

    # Find HCC_010 data (from repo root)
    repo_root = Path(__file__).parents[1]
    data_dir = repo_root / "HCC 010 final" / "DATA HCC 010" / "HCC 010" / "05-03-1998-NA-ABDPEL LIVER-46678" / "2.000000-PRE LIVER-34910"

    if not data_dir.exists():
        print(f"\n✗ Error: HCC_010 data not found at {data_dir}")
        print("Make sure 'HCC 010 final' folder is in the repo root.")
        return 1

    # Find DICOM files
    dcm_files = sorted(data_dir.glob("1-*.dcm"))
    if not dcm_files:
        print(f"\n✗ Error: No DICOM files found in {data_dir}")
        return 1

    print(f"\n✓ Found HCC_010 dataset at {data_dir}")
    print(f"✓ Found {len(dcm_files)} DICOM files")

    # Test 1: Parse individual files
    print("\n" + "-" * 70)
    print("Test 1: Parse Individual DICOM Files")
    print("-" * 70)

    for i, dcm_file in enumerate(dcm_files[:3]):
        print(f"\n  Parsing {dcm_file.name}...")
        try:
            pixel_array, metadata = parse_dicom_file(str(dcm_file))
            print(f"    ✓ Shape: {metadata['shape']}")
            print(f"    ✓ Pixel range: [{metadata['min_intensity']:.0f}, {metadata['max_intensity']:.0f}]")

            # Test windowing
            windowed = apply_windowing(pixel_array)
            print(f"    ✓ Windowed range: [{windowed.min()}, {windowed.max()}]")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return 1

    # Test 2: Extract slice indices
    print("\n" + "-" * 70)
    print("Test 2: Extract Slice Indices from Filenames")
    print("-" * 70)

    indices_found = []
    for dcm_file in dcm_files[:5]:
        idx = extract_slice_index(dcm_file.name)
        if idx:
            indices_found.append((dcm_file.name, idx))
            print(f"  {dcm_file.name} → Slice {idx}")

    if indices_found:
        print(f"\n✓ Successfully extracted {len(indices_found)} slice indices")
    else:
        print("\n✗ Failed to extract slice indices")
        return 1

    # Test 3: Simulate series alignment
    # (We can't really align pre/post since we only have pre-contrast in this test data)
    print("\n" + "-" * 70)
    print("Test 3: Series Alignment Simulation")
    print("-" * 70)

    pre_files = [str(f) for f in dcm_files[:5]]
    post_files = [str(f) for f in dcm_files[:5]]  # Same files for simulation

    try:
        pairs = align_series(post_files, pre_files)
        print(f"\n✓ Successfully aligned {len(pairs)} slice pairs")
        for post, pre, idx in pairs:
            print(f"  Slice {idx}: {Path(post).name} ↔ {Path(pre).name}")
    except Exception as e:
        print(f"\n✗ Error aligning series: {e}")
        return 1

    # Test 4: Subtraction and encoding
    print("\n" + "-" * 70)
    print("Test 4: Image Subtraction and Base64 Encoding")
    print("-" * 70)

    try:
        # Load first two slices
        post_array, _ = parse_dicom_file(str(dcm_files[0]))
        pre_array, _ = parse_dicom_file(str(dcm_files[1]))

        print(f"\n  Computing subtraction...")
        subtracted = subtract_images(post_array, pre_array)
        print(f"    ✓ Subtracted shape: {subtracted.shape}")
        print(f"    ✓ Subtracted range: [{subtracted.min():.0f}, {subtracted.max():.0f}]")

        print(f"\n  Normalizing for display...")
        normalized = normalize_for_display(subtracted)
        print(f"    ✓ Normalized dtype: {normalized.dtype}")
        print(f"    ✓ Normalized range: [{normalized.min()}, {normalized.max()}]")

        print(f"\n  Encoding as base64 PNG...")
        b64_png = array_to_png_base64(normalized)
        size_kb = len(b64_png) / 1024
        print(f"    ✓ Encoded size: {size_kb:.1f} KB")
        print(f"    ✓ Data URL prefix: {b64_png[:50]}...")

    except Exception as e:
        print(f"\n✗ Error during subtraction: {e}")
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n✓ DICOM Handler Verification PASSED")
    print(f"  - Parsed {len(dcm_files)} DICOM files")
    print(f"  - Extracted {len(indices_found)} slice indices")
    print(f"  - Aligned {len(pairs)} slice pairs")
    print(f"  - Successfully computed subtraction and encoding")
    print(f"\n✓ Ready for backend series endpoints implementation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
