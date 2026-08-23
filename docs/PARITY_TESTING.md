# Feature-Extraction Parity Testing

This is the single biggest risk to trusting any prediction from this app: the trained models learned from **Python-computed** radiomic features, but the deployed app computes those same 25 features in **JavaScript**, in the browser (`extractAllFeatures` in `frontend/src/App.jsx`). If the two implementations disagree — even slightly, on GLCM quantization, entropy bin counts, or a rounding difference — the model receives numbers unlike anything it was trained on, and its predictions are not trustworthy, even though the model itself is fine.

**This has not yet been run in this repository**, because it requires two things this repo does not currently have:
1. The original Python feature-extraction code used at training time.
2. A handful of CT slices with known ground-truth feature values (e.g. saved feature CSVs from training).

Neither was part of the delivered files inspected during setup (only the trained `.joblib` models, `run_summary.json`, and the frontend prototype were present). **Do not treat any prediction from this app as clinically meaningful until this test has been run and passed.**

## How to run it, once you have the missing pieces

1. **Select 5–10 slices** from the original training/test set for which exact feature values are already known.
2. **Run the same slices through the frontend.** Upload the same post/pre-contrast images, and draw an ROI matching the original mask as closely as possible.
3. **Record the extracted values** shown in Step 4 of the UI ("Features extracted from your drawn ROI").
4. **Compare feature-by-feature** against the known Python values. Flag anything with a relative difference over ~5% for investigation.
5. **Likely causes of mismatch**, if any turn up:
   - Different GLCM gray-level quantization (`nLevels` in `glcmFeatures` — currently 24)
   - Different histogram bin count for entropy (`nBins` in `shannonEntropy` — currently 32)
   - Different rounding/precision at some intermediate step
   - A mask drawn slightly differently than the original (freehand ROI vs. the original segmentation mask)
6. **Document the result.** Either the values match within tolerance, or discrepancies are resolved, or they're explicitly accepted with a written rationale — but this decision must be made and recorded before real-world use, not skipped.

## A lightweight sanity check you *can* do right now

Even without ground-truth data, you can catch gross bugs:
- Upload the **same image** as both post-contrast and pre-contrast. The subtraction should be all zeros/black, and most first-order features should come out as `0` or `NaN` — if you see large nonzero values, something in the subtraction or feature code is broken.
- Draw a tiny ROI (a few pixels) vs. a large one on the same image, and confirm `Volume`/`Area` scale accordingly and shape features (`Sphericity`, `Compactness`) stay in a sane 0–2 range.

## Where the reference data should live once available

`data/expected_features/` is set up to hold known feature-value CSVs/JSON for this purpose — see [../data/README.md](../data/README.md).
