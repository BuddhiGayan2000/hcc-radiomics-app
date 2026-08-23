# Testing the Web App with HCC 010 Dataset

## What You Have

You have received a complete test dataset for patient **HCC_010** containing:

1. **DICOM Images** (`DATA HCC 010/`) — Raw CT scans (pre-contrast phase)
   - Patient: HCC_010 (date: 05-03-1998)
   - ~20 slices showing the tumor at different anatomical levels
   - Format: DICOM (.dcm files)

2. **Ground-Truth Features** (`FEATURES HCC 010/HCC_010_tumor_features.csv`) — Expected radiomic values
   - 4 tumor slices with all 25 pre-extracted features
   - Slices: neg100.38, neg110.38, neg120.38, neg130.38
   - Can be used to validate feature extraction parity

3. **Visual Reference** (`FEATURES HCC 010/`) — Mask and overlay images
   - `Slice_*_mask.png` — Binary mask showing tumor ROI location (small files)
   - `Slice_*_overlay.png` — Visual overlay of tumor on actual CT slice (helps you know where to draw ROI)

4. **Anatomical Reference** (`SAME SLICE LOCATION HCC 010/`) — Pre/post-contrast slice pairs
   - Shows same anatomical location in both pre- and post-contrast phases
   - Useful for understanding multi-phase imaging context

---

## Clinical Workflow (Critical Understanding)

The app implements the **standard HCC imaging protocol**:

1. **Pre-contrast CT scan** — baseline liver attenuation
2. **Post-contrast CT scan** — same anatomical position, after IV contrast injection
3. **Subtraction image** — Post minus Pre reveals enhanced (vascularized) areas
4. **User reviews subtraction** — Picks the clearest slice showing tumor enhancement
5. **User draws ROI** — On the subtracted image
6. **Models predict** — Based on 25 features from subtracted ROI

This is why the dataset has BOTH pre and post-contrast images — they must be **aligned** (same patient, same slice position).

---

## Step-by-Step Testing Workflow

### Phase 1: Environment Setup (5 minutes)

**Make sure both servers are running:**

```bash
# Terminal 1: Backend (from repo root)
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Frontend (from repo root)  
cd frontend
npm run dev
```

Open browser: http://localhost:5173

---

### Phase 2: Prepare Test Image Pairs (5 minutes)

**The web app requires PAIRED pre/post-contrast PNG files** (same slice, different phases).

Your DICOM files need conversion. The test dataset has PRE-contrast images. You'll need to:
1. Find the matching POST-contrast DICOM series
2. Convert both to PNG
3. Use them as a pair in the app

**⚠️ CRITICAL:** You need BOTH pre and post-contrast images for the same slice.

The test data has **pre-contrast** series. You'll need to find the **post-contrast** series in the same patient folder. Check:

```
HCC 010 final/DATA HCC 010/HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/
```

Look for folders like:
- `2.000000-PRE LIVER-34910` ← Pre-contrast (you have this)
- `3.000000-POST LIVER-???` OR similar ← Post-contrast (need to find)
- `4.000000-VENOUS-??` OR similar ← Other phases

**Until you get the post-contrast series, you can:**
1. Convert two similar pre-contrast slices (simulating pre/post)
2. Use one as "post" and another as "pre" for testing flow
3. Note: Results won't be clinically meaningful, but feature extraction flow will work

**Option A: Using the provided conversion script**

```bash
# Install dependencies
pip install pydicom pillow numpy

# Run the script to convert slices
cd hcc-radiomics-app
python scripts/convert_dicom_to_png.py --slices 10 11
# Creates: data/test_images/HCC_010_slice_10_pre_contrast.png
#          data/test_images/HCC_010_slice_11_pre_contrast.png
```

This creates PNG files. For the app to work properly, you would need:
- `HCC_010_slice_10_post_contrast.png` (post-contrast of same slice)
- `HCC_010_slice_10_pre_contrast.png` (pre-contrast of same slice)

Then the app subtracts: Post - Pre = Enhancement image showing tumor.

**Option B: Get full DICOM series from your research team**
The complete dataset should include:
- Pre-contrast series (already have)
- Post-contrast series (need to request)
- Both aligned to same slices

Ask your team for the full multi-phase DICOM dataset for HCC_010.

---

### Phase 3: Understand the ROI (3 minutes)

Open the overlay images to see **where the tumor is in the subtracted image**:

```bash
# View overlay for one of the test slices
# On Windows:
start "HCC 010 final/FEATURES HCC 010/FINAL FEATURES_HCC_010/Slice_neg100.38_overlay.png"

# On Mac/Linux:
open "HCC 010 final/FEATURES HCC 010/FINAL FEATURES_HCC_010/Slice_neg100.38_overlay.png"
```

**What you see:**
- A subtracted CT image (appears as enhancement/bright regions)
- The red/colored outline shows the tumor ROI boundary
- This is what the app will show you after computing Post - Pre
- Your ROI should match this boundary

---

### Phase 4: Test in the Web App (15 minutes)

#### Step 4.1: Upload Post-Contrast Image (Step 1)

1. App opens with **"Step 1 — Select post-contrast image"**
2. Click **"Upload Image"** or drag/drop
3. Select one of your converted PNG files (e.g., `HCC_010_slice_10_post_contrast.png`)
4. ✓ Expected: Image displays in canvas preview

#### Step 4.2: Upload Pre-Contrast Image (Step 2)

1. Click **"Next"** → App moves to **"Step 2 — Select pre-contrast image"**
2. Upload the matching pre-contrast PNG (e.g., `HCC_010_slice_10_pre_contrast.png`)
3. Should be the **exact same slice position** as post-contrast
4. ✓ Expected: Image displays in canvas preview

#### Step 4.3: Review Subtracted Image (Step 3)

1. Click **"Next"** → App automatically computes **Post - Pre**
2. You see the **subtracted enhancement image** displayed
3. The tumor region should appear brighter (high contrast enhancement)
4. ✓ Expected: Subtracted image shows tumor enhancement clearly

#### Step 4.4: Draw the Tumor ROI (Step 3)

1. **Compare with overlay image** to see where tumor should be
2. Use the **freehand drawing tool** to trace tumor boundary on subtracted image
   - Click to start, move mouse to trace
   - Release to finish
   - The app closes the polygon automatically
3. Aim to match the red outline from the overlay image
4. ✓ Expected: ROI polygon displays on subtracted image

#### Step 4.3: Auto-Extract Features

1. Once ROI is drawn, click **"Extract Features"**
2. The app extracts all 25 radiomic features from the ROI
3. Features appear in a table below
4. ✓ Expected: All 25 features have numeric values (no errors)

#### Step 4.4: Compare with Ground Truth

This is **critical for validation**.

**Get expected features:**
- Open `HCC_010_tumor_features.csv`
- Find the row matching your slice (check the overlay image name, e.g., "Slice_neg100.38")
- Extract that row's feature values

**Compare in the app:**
1. Write down extracted values from the web app
2. Write down expected values from the CSV
3. Calculate differences:
   - **Acceptable**: ±10% for most features (due to ROI drawing variation)
   - **Concerning**: >10% difference for basic morphology (Volume, Area, Diameter)
   - **Concerning**: >5% difference for mean intensity (Mean, Median)

**Example comparison:**
```
Feature              | Extracted | Expected | Difference | Status
Volume               | 3850      | 3895     | -1.2%      | ✓ Good
Area                 | 1540      | 1558     | -1.2%      | ✓ Good
MaxDiameter          | 47.5      | 48.28    | -1.6%      | ✓ Good
Mean                 | 285.6     | 285.69   | -0.03%     | ✓ Excellent
GLCM_Entropy         | 6.84      | 6.84     | 0.0%       | ✓ Perfect
```

### Step 4.5: Make Predictions (5 minutes)

Once features are extracted:

1. The **Prediction Panel** appears on the right showing:
   - **BCLC Stage** (4-class: Advanced, StageA, StageB, Healthy)
   - **Necrosis Probability** (binary: has/no necrosis)
   - Confidence scores for each class
   - **SHAP Evidence Strip** — top contributing features

2. For each model, you should see:
   - Stage prediction probabilities summing to ~1.0
   - Individual confidence per class
   - For GradientBoosting: Evidence strip will be **empty** (known limitation — SHAP incompatibility)
   - For other models: Evidence strip shows top 5 contributing features

3. **Check results make sense:**
   - No negative probabilities
   - No NaN or infinite values
   - Predictions differ between models (expected, different algorithms)

---

## Expected Results and Interpretation

### Feature Extraction Accuracy

| Scenario | Interpretation | Next Steps |
|----------|---|---|
| Most features ±10%, morphology ±5% | ✓ **GOOD** — Feature extraction is working | Proceed to full dataset validation |
| Consistent 15–20% difference on all features | ⚠️ **Check ROI** — Is your ROI smaller/larger than the overlay? | Redraw ROI more carefully, match overlay boundary |
| Random, unpredictable differences >20% | ✗ **FEATURE EXTRACTION BUG** — Algorithm mismatch between JS and Python | See [PARITY_TESTING.md](./PARITY_TESTING.md) for diagnosis |
| All features = 0 or NaN | ✗ **CRITICAL BUG** — Feature extractor crashed silently | Check browser console for errors |

### Model Predictions

| Prediction | What It Means | How to Validate |
|---|---|---|
| BCLC Stage: 95% StageA, 5% StageB | Clear prediction for StageA | Ask clinician: is this reasonable for this patient? |
| BCLC Stage: 40% Healthy, 35% StageA, 20% StageB | Uncertain prediction (borderline case) | Model is appropriately uncertain; useful for clinical review |
| BCLC Stage: 0.1% Healthy, 99.9% Advanced | Extremely confident | Check: is tumor truly very large/advanced? |
| Necrosis: 98% No Necrosis | High confidence no necrosis present | Should match pathology report (if available) |

### SHAP Explanations (Evidence Strip)

**For XGBoost, LightGBM, RandomForest models:**
- Shows top 5 features that **pushed prediction toward the predicted class**
- Feature value + contribution value
- Example: "Volume +450: +0.23" means "higher volume increased Stage A probability by 0.23"

**For GradientBoosting:**
- Evidence strip will be **empty** (documented limitation)
- Predictions still work; just no explanation

---

## Testing Checklist

Use this to ensure the web app is working:

### Environment
- [ ] Backend running on `http://localhost:8000`
- [ ] Frontend running on `http://localhost:5173`
- [ ] No CORS errors in browser console
- [ ] API health check returns `{"status": "healthy"}`

### Upload & ROI Drawing
- [ ] Can upload pre-contrast PNG image
- [ ] Image displays correctly (not inverted/distorted)
- [ ] ROI drawing tool responds to clicks
- [ ] Can complete and close a polygon ROI
- [ ] ROI displays on the image

### Feature Extraction
- [ ] Feature extraction completes without crashing
- [ ] All 25 features have numeric values
- [ ] No NaN, Infinity, or error messages
- [ ] Feature values match expected CSV (±10% acceptable)
- [ ] Extracted features are reasonable (e.g., Volume > 0, Mean in image range)

### Predictions
- [ ] Stage prediction probabilities display (4 classes)
- [ ] Probabilities sum to ~1.0
- [ ] Necrosis probability displays (binary: 0–1 range)
- [ ] All predictions are numeric (no NaN)
- [ ] At least one model has SHAP evidence strip (XGBoost, LightGBM, or RF)

### Error Handling
- [ ] If ROI is invalid (overlaps itself), app shows error
- [ ] If image is too small, app shows warning
- [ ] Browser console is clean (no unhandled exceptions)

---

## Troubleshooting

### "API Unreachable" Error
**Problem:** Frontend shows "Cannot connect to API at http://localhost:8000"
**Diagnosis:**
1. Check backend is running: `curl http://localhost:8000/health`
2. Check CORS settings: Backend should allow `http://localhost:5173`
3. Check firewall (Windows): Allow Python.exe on port 8000

**Fix:**
```bash
# Kill all Python processes
taskkill /F /IM python.exe

# Restart backend cleanly
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Image Doesn't Display
**Problem:** Image loads but doesn't show in canvas
**Diagnosis:**
1. Check image format (must be PNG, JPG, or GIF)
2. Check image size (should be reasonable, e.g., 512×512)
3. Open browser DevTools → Console, check for errors

**Fix:**
- Re-convert DICOM with correct windowing (window=400, level=50)
- Try a different test image

### ROI Drawing is Laggy
**Problem:** Clicking to add ROI points is slow/unresponsive
**Diagnosis:** 
1. Image is very large (>2048×2048 pixels)
2. Browser is using CPU instead of GPU

**Fix:**
- Resize image before uploading: `512×512` or `1024×1024`
- Use Chrome instead of Firefox (better GPU acceleration)

### Features Are All NaN or Zero
**Problem:** After ROI extraction, all feature values are NaN or 0
**Diagnosis:**
1. ROI is outside image bounds
2. ROI has <10 pixels (too small)
3. JavaScript feature extraction crashed

**Fix:**
1. Check browser console (F12 → Console tab)
2. Look for error messages
3. See [PARITY_TESTING.md](./PARITY_TESTING.md) for debugging

### Predictions Are Identical Between Models
**Problem:** All 4 stage models return the same prediction
**Diagnosis:**
1. Features are unrealistic (all zeros, all identical)
2. Scaler or pipeline is misconfigured

**Fix:**
1. Verify features match expected CSV (Step 4.4)
2. Check backend logs for errors during prediction

---

## Next Steps: Full Validation

Once you've successfully tested 1–2 slices:

### 1. Convert All Test Slices (Optional)
```bash
# Convert all 20 slices from the full DICOM series
python scripts/convert_dicom_to_png.py --all
```

Then test with multiple slices to check consistency.

### 2. Feature Extraction Parity Validation
Once you have feature values for all 4 test slices, create a comparison CSV:

```csv
Slice,Feature,Extracted,Expected,Difference_%,Status
Slice_neg100.38,Volume,3850,3895,-1.2,PASS
Slice_neg100.38,Area,1540,1558,-1.2,PASS
...
```

See [PARITY_TESTING.md](./PARITY_TESTING.md) for detailed guidance.

### 3. Model Accuracy Assessment
Create a summary:
```
Patient: HCC_010
Slices Tested: 4
Feature Accuracy: 98% (avg difference 2.1%)
Model Predictions:
  - XGBoost Stage: [probabilities]
  - LightGBM Stage: [probabilities]
  - RandomForest Stage: [probabilities]
  - GradientBoosting Stage: [probabilities]
  - Necrosis RF: [probability]
```

Present to your research team with questions:
- Do predictions align with clinical assessment?
- Are differences between models expected (algorithm variety)?
- Is necrosis prediction reasonable given pathology data?

### 4. Ask Research Team
Before clinical use, get answers to:
1. **Ground truth labels**: What is the actual BCLC stage for HCC_010?
2. **Necrosis status**: Was necrosis present on pathology?
3. **Feature source**: Which radiomics library extracted the ground-truth features? (PyRadiomics?)
4. **CV method**: Were models trained with patient-level or slice-level cross-validation?

---

## Files Reference

```
HCC 010 final/
├── DATA HCC 010/
│   └── HCC 010/05-03-1998-NA-ABDPEL LIVER-46678/
│       └── 2.000000-PRE LIVER-34910/          ← Pre-contrast DICOMs
│           └── 1-01.dcm, 1-02.dcm, ... (20 slices)
│
├── FEATURES HCC 010/
│   └── FINAL FEATURES_HCC_010/
│       ├── HCC_010_tumor_features.csv         ← Ground truth features (USE THIS!)
│       ├── Slice_neg100.38_mask.png           ← Binary ROI mask
│       ├── Slice_neg100.38_overlay.png        ← Visual reference (WHERE TO DRAW)
│       └── ...more slices...
│
└── SAME SLICE LOCATION HCC 010/
    └── HCC_010/
        ├── Slice_neg10.38/
        │   ├── PreLiver_Slice3_Phase3Slice3.dcm     (Pre-contrast)
        │   └── Phase3_Slice3_PreLiverSlice3.dcm     (Post-contrast)
        └── ...more locations...
```

---

## Summary

Your testing workflow:

1. **Setup**: Both servers running ✓
2. **Convert**: 1–2 DICOM slices → PNG ✓
3. **Upload**: PNG to web app ✓
4. **Draw**: ROI matching overlay image ✓
5. **Extract**: Get 25 features ✓
6. **Compare**: Features match CSV (±10%) ✓
7. **Predict**: 4 stage models + necrosis ✓
8. **Explain**: SHAP evidence strip (except GradientBoosting) ✓
9. **Report**: Summary of results to research team ✓

**Expected time: 30–45 minutes for complete validation of 1–2 slices.**

See [PARITY_TESTING.md](./PARITY_TESTING.md) for deeper validation steps.
