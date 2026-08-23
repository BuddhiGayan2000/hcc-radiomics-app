#!/usr/bin/env python3
"""
Validate extracted radiomic features against ground truth CSV.

This script compares features extracted by the web app with the ground-truth
values from HCC_010_tumor_features.csv, calculating differences and flagging
potential feature extraction issues.

Usage:
    python scripts/validate_features.py --extracted extracted.csv --ground-truth "HCC 010 final/FEATURES HCC 010/FINAL FEATURES_HCC_010/HCC_010_tumor_features.csv"
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd


# Thresholds for feature difference validation (%)
MORPHOLOGY_THRESHOLD = 5.0  # Volume, Area, Diameter should match closely
INTENSITY_THRESHOLD = 5.0   # Mean, Median, intensity stats
TEXTURE_THRESHOLD = 10.0    # GLCM, RLE features have more variation


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and handle missing files."""
    p = Path(path)
    if not p.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)
    return pd.read_csv(p)


def calculate_percentage_diff(extracted: float, expected: float) -> float:
    """Calculate percentage difference, handling division by zero."""
    if expected == 0:
        # If expected is 0 and extracted is 0, diff is 0%; otherwise inf
        return 0.0 if extracted == 0 else float('inf')
    return abs((extracted - expected) / expected) * 100


def get_threshold_for_feature(feature_name: str) -> float:
    """Return appropriate threshold for this feature type."""
    if feature_name in ["Volume", "Area", "MaxDiameter", "Perimeter"]:
        return MORPHOLOGY_THRESHOLD
    elif feature_name in ["Sphericity", "Compactness", "Elongation"]:
        return MORPHOLOGY_THRESHOLD
    elif feature_name in ["Mean", "Median", "Min", "Max", "Std"]:
        return INTENSITY_THRESHOLD
    elif feature_name in ["Skewness", "Kurtosis", "Entropy"]:
        return INTENSITY_THRESHOLD
    else:  # Texture features (GLCM, SRE, LRE, etc.)
        return TEXTURE_THRESHOLD


def validate_features(
    extracted_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    slice_name: str = None
) -> Tuple[List[Dict], float]:
    """
    Compare extracted vs ground-truth features.

    Args:
        extracted_df: DataFrame with extracted features (single row)
        ground_truth_df: DataFrame with ground-truth features (may have multiple rows)
        slice_name: Optional slice identifier to match (e.g., "Slice_neg100.38")

    Returns:
        (list of issue dicts, average percentage difference)
    """
    issues = []

    # If slice_name provided, filter ground truth to that slice
    if slice_name and "Slice" in ground_truth_df.columns:
        ground_truth_df = ground_truth_df[ground_truth_df["Slice"] == slice_name]
        if ground_truth_df.empty:
            print(f"Error: No ground-truth data for slice {slice_name}")
            return [], float('inf')

    if ground_truth_df.shape[0] > 1:
        print("Warning: Multiple ground-truth rows; using first")
    ground_truth = ground_truth_df.iloc[0]

    # Get feature columns (exclude metadata like PatientID, Slice)
    feature_cols = [c for c in extracted_df.columns if c not in ["PatientID", "Slice"]]

    diffs = []
    for col in feature_cols:
        if col not in ground_truth.index:
            issues.append({
                "feature": col,
                "severity": "warning",
                "message": f"Feature not in ground truth"
            })
            continue

        extracted_val = extracted_df[col].iloc[0]
        expected_val = ground_truth[col]

        # Handle NaN, None, "N/A"
        try:
            extracted_val = float(extracted_val)
            expected_val = float(expected_val)
        except (ValueError, TypeError):
            issues.append({
                "feature": col,
                "severity": "error",
                "message": f"Non-numeric value: extracted={extracted_val}, expected={expected_val}"
            })
            continue

        # Calculate difference
        diff_pct = calculate_percentage_diff(extracted_val, expected_val)
        diffs.append(diff_pct)

        # Check against threshold
        threshold = get_threshold_for_feature(col)
        if diff_pct > threshold:
            severity = "warning" if diff_pct < threshold * 2 else "error"
            issues.append({
                "feature": col,
                "severity": severity,
                "extracted": extracted_val,
                "expected": expected_val,
                "diff_pct": diff_pct,
                "threshold": threshold,
                "message": f"{diff_pct:.1f}% diff (threshold: {threshold}%)"
            })

    avg_diff = sum(diffs) / len(diffs) if diffs else float('inf')
    return issues, avg_diff


def print_report(issues: List[Dict], avg_diff: float, slice_name: str = None):
    """Pretty-print validation report."""
    if slice_name:
        print(f"\n{'='*70}")
        print(f"Feature Validation Report: {slice_name}")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"Feature Validation Report")
        print(f"{'='*70}")

    if not issues:
        print(f"\n✓ All features within acceptable tolerance (avg diff: {avg_diff:.1f}%)")
        return True

    # Separate by severity
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    if errors:
        print(f"\n✗ ERRORS ({len(errors)}):")
        for issue in errors:
            if "diff_pct" in issue:
                print(f"  {issue['feature']:20} {issue['message']}")
                print(f"    Extracted: {issue['extracted']:.4f}, Expected: {issue['expected']:.4f}")
            else:
                print(f"  {issue['feature']:20} {issue['message']}")

    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for issue in warnings:
            if "diff_pct" in issue:
                print(f"  {issue['feature']:20} {issue['message']}")
                print(f"    Extracted: {issue['extracted']:.4f}, Expected: {issue['expected']:.4f}")
            else:
                print(f"  {issue['feature']:20} {issue['message']}")

    print(f"\nAverage difference: {avg_diff:.1f}%")

    if errors:
        print("\nStatus: ✗ FAILED — Feature extraction has issues")
        return False
    else:
        print("\nStatus: ⚠ PASSED WITH WARNINGS — Check differences above")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate radiomic feature extraction against ground truth"
    )
    parser.add_argument(
        "--extracted", type=str, required=True,
        help="Path to CSV with extracted features (from web app)"
    )
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to ground-truth CSV (HCC_010_tumor_features.csv)"
    )
    parser.add_argument(
        "--slice", type=str,
        help="Optional: slice identifier to match (e.g., 'Slice_neg100.38')"
    )
    args = parser.parse_args()

    # Load CSVs
    print("Loading feature CSVs...")
    extracted = load_csv(args.extracted)
    ground_truth = load_csv(args.ground_truth)

    print(f"Extracted features: {extracted.shape[0]} rows, {extracted.shape[1]} columns")
    print(f"Ground truth: {ground_truth.shape[0]} rows, {ground_truth.shape[1]} columns")

    # Validate
    if extracted.shape[0] != 1:
        print(f"\nWarning: Expected 1 row in extracted features, got {extracted.shape[0]}")
        print("Using first row for comparison")

    issues, avg_diff = validate_features(
        extracted,
        ground_truth,
        slice_name=args.slice
    )

    # Report
    success = print_report(issues, avg_diff, slice_name=args.slice)

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)

    if not issues:
        print("""
✓ FEATURE EXTRACTION IS WORKING CORRECTLY
  → Feature extraction matches expected values
  → ROI drawing and calculations are accurate
  → Ready to test model predictions

Next steps:
  1. Export features for all 4 test slices
  2. Compare all slices
  3. Check model predictions make sense
  4. Proceed to full dataset validation
""")
    elif all(i.get("severity") != "error" for i in issues):
        print("""
⚠ MINOR DIFFERENCES DETECTED
  → Small differences are expected due to:
    - ROI drawing variation (±5-10% normal)
    - Floating-point precision differences
    - Minor algorithm differences

How to improve:
  1. Redraw ROI more carefully, matching the overlay image
  2. Ensure ROI boundary aligns with tumor edge
  3. Check that ROI is completely within image bounds

If differences persist >10%, see PARITY_TESTING.md
""")
    else:
        print("""
✗ SIGNIFICANT ISSUES DETECTED
  → Feature extraction may have bugs
  → Check ROI drawing carefully first
  → If ROI is correct, feature extraction algorithm needs investigation

Troubleshooting:
  1. Verify ROI matches the overlay image from FEATURES folder
  2. Check that all 25 features are being calculated
  3. Review PARITY_TESTING.md for detailed diagnosis steps
  4. Compare JavaScript feature extraction with original Python

Critical checks:
  - Volume/Area should be nearly identical (error suggests wrong units?)
  - Mean/Median should match HU values in image
  - GLCM should be stable (texture features are less sensitive to ROI variation)
""")

    return 0 if success or not errors else 1


if __name__ == "__main__":
    sys.exit(main())
