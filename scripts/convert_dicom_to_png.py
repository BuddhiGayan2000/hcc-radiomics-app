#!/usr/bin/env python3
"""
Convert DICOM images from the HCC 010 test dataset to PNG format for web app testing.

This script:
1. Reads DICOM files from the test dataset
2. Applies windowing/leveling suitable for liver CT (window=400, level=50)
3. Exports as PNG to data/test_images/ for upload to the web app

Usage:
    python scripts/convert_dicom_to_png.py --all          # Convert all 20 slices
    python scripts/convert_dicom_to_png.py --slices 10 12  # Convert specific slices
"""
import sys
import argparse
import warnings
from pathlib import Path

try:
    import pydicom
    import numpy as np
    from PIL import Image
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install pydicom pillow numpy")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "HCC 010 final" / "DATA HCC 010" / "HCC 010" / "05-03-1998-NA-ABDPEL LIVER-46678" / "2.000000-PRE LIVER-34910"
OUTPUT_DIR = REPO_ROOT / "data" / "test_images"

# CT windowing for liver (standard)
WINDOW_WIDTH = 400
WINDOW_CENTER = 50


def convert_dicom_to_png(dcm_path, output_png_path):
    """
    Convert a DICOM file to PNG with windowing applied.

    Args:
        dcm_path (Path): Path to .dcm input file
        output_png_path (Path): Path to save .png output

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        dcm = pydicom.dcmread(dcm_path)
        pixel_array = dcm.pixel_array

        # Apply Hounsfield window/level transformation
        # This makes the image suitable for visual inspection
        min_val = WINDOW_CENTER - WINDOW_WIDTH / 2
        max_val = WINDOW_CENTER + WINDOW_WIDTH / 2
        windowed = np.clip(pixel_array, min_val, max_val)
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Save as PNG
        img = Image.fromarray(windowed)
        img.save(output_png_path)
        return True
    except Exception as e:
        print(f"  ✗ {dcm_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert HCC 010 DICOM images to PNG for web app testing"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Convert all 20 slices"
    )
    parser.add_argument(
        "--slices", type=int, nargs="+",
        help="Convert specific slices (e.g., --slices 10 12 15)"
    )
    args = parser.parse_args()

    # Default to slices 10 and 12 if no args
    if not args.all and not args.slices:
        args.slices = [10, 12]

    # Determine which slices to convert
    if args.all:
        slices = list(range(1, 21))
    else:
        slices = args.slices

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check data directory exists
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Make sure HCC 010 final folder is in the repo root.")
        sys.exit(1)

    # Convert
    print(f"Converting {len(slices)} DICOM slices...")
    print(f"Windowing: width={WINDOW_WIDTH}, center={WINDOW_CENTER}")
    print(f"Output: {OUTPUT_DIR}\n")

    success = 0
    failed = 0

    for slice_num in sorted(slices):
        dcm_filename = f"1-{slice_num:02d}.dcm"
        dcm_path = DATA_DIR / dcm_filename

        if not dcm_path.exists():
            print(f"✗ {dcm_filename} not found")
            failed += 1
            continue

        png_filename = f"HCC_010_slice_{slice_num:02d}_pre_contrast.png"
        png_path = OUTPUT_DIR / png_filename

        if convert_dicom_to_png(dcm_path, png_path):
            print(f"✓ {dcm_filename} → {png_filename}")
            success += 1
        else:
            failed += 1

    print(f"\n{success} converted, {failed} failed")
    print(f"\nTest images ready in: {OUTPUT_DIR}")
    print("To test in the web app:")
    print("  1. Open http://localhost:5173 (make sure servers are running)")
    print("  2. Click 'Upload Pre-Contrast Image'")
    print(f"  3. Select an image from {OUTPUT_DIR.name}/")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
