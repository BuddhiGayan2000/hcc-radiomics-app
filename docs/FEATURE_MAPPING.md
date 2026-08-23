# Feature Mapping — All 25 Radiomic Features

These are computed by the frontend (`extractAllFeatures` in `frontend/src/App.jsx`) from the pixels inside the ROI you draw on the subtracted (post − pre contrast) image. `*` marks the two "liver-context" features, which additionally use the rest of the image as a proxy for whole-liver tissue.

## Morphological (7) — shape of the drawn ROI

| Feature | Meaning |
|---|---|
| `Volume` | Pixel count inside the ROI (2D "area", named for parity with the original 3D radiomics vocabulary) |
| `Area` | Same as Volume in this 2D implementation |
| `MaxDiameter` | Largest pairwise distance between sampled boundary points |
| `SurfaceArea` | Perimeter — count of boundary pixels |
| `Sphericity` | `4π·Area / Perimeter²` — how close to a circle |
| `Compactness` | `Area / Perimeter²` |
| `Elongation` | Ratio of the two principal axis lengths (PCA of pixel coordinates) |

## First-order intensity (8) — pixel value statistics inside the ROI

| Feature | Meaning |
|---|---|
| `Mean`, `Median`, `Min`, `Max`, `Std` | Standard descriptive statistics of subtracted-image pixel values inside the ROI |
| `Skewness`, `Kurtosis` | Third/fourth standardized moments |
| `Entropy` | Shannon entropy over a 32-bin histogram of ROI pixel values |

## GLCM / run-length (10) — texture

| Feature | Meaning |
|---|---|
| `GLCM_Contrast` | Gray-Level Co-occurrence Matrix: local intensity variation |
| `GLCM_Correlation` | Linear dependency of gray levels between neighboring pixels |
| `GLCM_Homogeneity` | Closeness of the GLCM distribution to its diagonal |
| `GLCM_Energy` | Sum of squared GLCM elements (textural uniformity) |
| `GLCM_Entropy` | Entropy of the GLCM itself |
| `SRE` | Short Run Emphasis (gray-level run-length matrix) |
| `LRE` | Long Run Emphasis |
| `GLN` | Gray-Level Nonuniformity |
| `LiverEntropy` * | Entropy of non-ROI pixels above the 10th percentile — proxy for surrounding liver tissue texture |
| `TumorLiverContrast` * | `\|tumor mean − liver-context mean\| / \|liver-context mean\|` |

## The 12 SHAP-selected features

Per `backend/models_storage/metadata.json` (from the original study's model selection), these 12 carry the most predictive signal for BCLC staging and are what three of the four staging models (`LightGBM`, `RandomForest`, `GradientBoosting`) are trained on:

```
Std, TumorLiverContrast, LiverEntropy, Compactness, SurfaceArea, GLCM_Correlation,
Mean, Skewness, GLCM_Homogeneity, MaxDiameter, Kurtosis, Entropy
```

The XGBoost model uses the same 12 with one swap (`GLCM_Energy` instead of `Kurtosis`) — see [backend/docs/MODEL_LOADING.md](../backend/docs/MODEL_LOADING.md) for the exact per-model list. The necrosis model uses all 25.

## Where the implementation lives

- **Frontend (JavaScript):** `frontend/src/App.jsx` — functions `firstOrderStats`, `shapeStats`, `glcmFeatures`, `glrlmFeatures`, `liverContextFeatures`.
- **Original training-time (Python):** not included in this repo — see [PARITY_TESTING.md](PARITY_TESTING.md) for why the two must be validated against each other before trusting predictions.
