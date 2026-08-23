# HCC 010 Test Results Template

**Date:** YYYY-MM-DD  
**Tester:** [Your Name]  
**Test Environment:** Windows 11, localhost (5173/8000)

---

## Test Overview

| Item | Result |
|------|--------|
| **Backend Status** | ✓ Running / ⚠ Errors / ✗ Failed |
| **Frontend Status** | ✓ Running / ⚠ Errors / ✗ Failed |
| **Test Slices** | 2 slices (default) / 4 slices / All 20 slices |
| **Overall Result** | ✓ PASS / ⚠ MINOR ISSUES / ✗ FAIL |

---

## Environment Verification

```bash
Backend health check:
$ curl http://localhost:8000/health
Response: {"status": "healthy"}  ✓

Frontend loads:
http://localhost:5173  ✓

CORS working:
No console errors on page load  ✓
```

---

## Test Slice 1: Slice_neg100.38

### Setup
- [ ] DICOM converted to PNG
- [ ] PNG uploaded to web app
- [ ] Image displays correctly

### ROI Drawing
- [ ] Overlay image reviewed
- [ ] ROI drawn and closed
- [ ] ROI visually matches overlay

### Feature Extraction
- [ ] No errors during extraction
- [ ] All 25 features have numeric values
- [ ] No NaN or Infinity values

### Feature Validation

| Feature | Extracted | Expected | Difference | Status |
|---------|-----------|----------|------------|--------|
| Volume | ___ | 3895 | __% | ✓/⚠/✗ |
| Area | ___ | 1558 | __% | ✓/⚠/✗ |
| MaxDiameter | ___ | 48.28 | __% | ✓/⚠/✗ |
| Mean | ___ | 285.69 | __% | ✓/⚠/✗ |
| Median | ___ | 225 | __% | ✓/⚠/✗ |
| GLCM_Entropy | ___ | 6.84 | __% | ✓/⚠/✗ |
| ... | ... | ... | ... | ... |

**Average Feature Difference:** ____%

**Status:** ✓ All ±10% / ⚠ Some 10-20% / ✗ Major differences >20%

### Model Predictions

**BCLC Stage Predictions:**
```
XGBoost:            Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid]
LightGBM:           Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid]
RandomForest:       Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid]
GradientBoosting:   Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid / ✗ NaN]
```

**Necrosis Prediction:**
```
RandomForest:       No Necrosis: __%, Necrosis: __% [✓ Valid / ✗ Invalid]
```

**SHAP Explanations:**
```
XGBoost:            ✓ Top 5 features shown / ✗ Empty
LightGBM:           ✓ Top 5 features shown / ✗ Empty
RandomForest:       ✓ Top 5 features shown / ✗ Empty
GradientBoosting:   ✓ Top 5 features shown / ✗ Empty (expected limitation)
```

### Notes for This Slice
```
[Any observations, issues, or notes about this test]
```

---

## Test Slice 2: Slice_neg110.38

### Setup
- [ ] DICOM converted to PNG
- [ ] PNG uploaded to web app
- [ ] Image displays correctly

### ROI Drawing
- [ ] Overlay image reviewed
- [ ] ROI drawn and closed
- [ ] ROI visually matches overlay

### Feature Extraction
- [ ] No errors during extraction
- [ ] All 25 features have numeric values
- [ ] No NaN or Infinity values

### Feature Validation

| Feature | Extracted | Expected | Difference | Status |
|---------|-----------|----------|------------|--------|
| Volume | ___ | 5637.5 | __% | ✓/⚠/✗ |
| Area | ___ | 2255 | __% | ✓/⚠/✗ |
| MaxDiameter | ___ | 54.91 | __% | ✓/⚠/✗ |
| Mean | ___ | 177.18 | __% | ✓/⚠/✗ |
| Median | ___ | 50 | __% | ✓/⚠/✗ |
| GLCM_Entropy | ___ | 6.84 | __% | ✓/⚠/✗ |
| ... | ... | ... | ... | ... |

**Average Feature Difference:** ____%

**Status:** ✓ All ±10% / ⚠ Some 10-20% / ✗ Major differences >20%

### Model Predictions

**BCLC Stage Predictions:**
```
XGBoost:            Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid]
LightGBM:           Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid]
RandomForest:       Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid]
GradientBoosting:   Advanced: _%, StageA: _%, StageB: _%, Healthy: _% [✓ Valid / ✗ NaN]
```

**Necrosis Prediction:**
```
RandomForest:       No Necrosis: __%, Necrosis: __% [✓ Valid / ✗ Invalid]
```

### Notes for This Slice
```
[Any observations, issues, or notes about this test]
```

---

## Summary

### Feature Extraction Accuracy (across all tested slices)

| Metric | Result | Status |
|--------|--------|--------|
| **Average % Difference** | ___% | ✓ <5% / ⚠ 5-10% / ✗ >10% |
| **Morphology Accuracy** | ___% | ✓ <5% / ⚠ 5-10% / ✗ >10% |
| **Intensity Accuracy** | ___% | ✓ <5% / ⚠ 5-10% / ✗ >10% |
| **Texture Accuracy** | ___% | ✓ <10% / ⚠ 10-15% / ✗ >15% |
| **Zero/NaN Values** | _____ | ✓ None / ⚠ Few / ✗ Many |

### Model Prediction Consistency

| Model | Passes QC | Notes |
|-------|-----------|-------|
| XGBoost | ✓ / ✗ | [Any issues with predictions] |
| LightGBM | ✓ / ✗ | [Any issues with predictions] |
| RandomForest (Stage) | ✓ / ✗ | [Any issues with predictions] |
| RandomForest (Necrosis) | ✓ / ✗ | [Any issues with predictions] |
| GradientBoosting | ✓ / ✗ | SHAP empty (expected), predictions valid |

### SHAP Explainability

- XGBoost evidence strip: ✓ Working / ✗ Empty / ✗ Error
- LightGBM evidence strip: ✓ Working / ✗ Empty / ✗ Error
- RandomForest evidence strip: ✓ Working / ✗ Empty / ✗ Error
- GradientBoosting evidence strip: ✓ Empty (expected) / ✗ Error

---

## Issues Found

### Issue 1
**Severity:** ✓ Low / ⚠ Medium / ✗ High  
**Feature/Component:** [e.g., "Feature extraction", "SHAP explanation", "Model prediction"]  
**Description:** [What went wrong]  
**Impact:** [What this means for clinical use]  
**Reproduction:** [Steps to reproduce]  
**Next Steps:**  
- [ ] Retest with different ROI
- [ ] Check browser console for errors
- [ ] See PARITY_TESTING.md for deeper diagnosis
- [ ] Report to development team

### Issue 2
[Repeat for additional issues]

---

## Overall Assessment

### ✓ PASS (Ready for Further Testing)
- Feature extraction accurate (within ±10%)
- All models producing valid predictions
- No critical errors or data corruption
- SHAP explanations working (where expected)

**Recommendation:** Proceed to full dataset validation and clinical review

### ⚠ PASS WITH RESERVATIONS (Minor Issues, Verify Before Clinical Use)
- Some features off by 10-20% (likely ROI variation)
- One or two prediction anomalies (expected variation between models)
- SHAP working for most models

**Recommendation:** Retrain with careful ROI drawing, validate with research team

### ✗ FAIL (Issues Must Be Resolved)
- Feature extraction differs >20% from ground truth
- Predictions contain NaN or invalid values
- Critical errors during any phase

**Recommendation:** Debug per PARITY_TESTING.md before further testing

---

## Validation Against Spec

| Requirement | Status | Notes |
|---|---|---|
| Feature extraction ✓ | ✓/✗ | [Pass/fail] |
| BCLC stage prediction (4-class) | ✓/✗ | [Pass/fail] |
| Necrosis prediction (binary) | ✓/✗ | [Pass/fail] |
| SHAP top-K features | ✓/⚠/✗ | [Works for all models except GradientBoosting] |
| Backend API responds correctly | ✓/✗ | [Pass/fail] |
| Frontend renders cleanly | ✓/✗ | [Pass/fail] |
| No privacy violations (logging) | ✓/✗ | [Check logs only contain metadata] |

---

## Next Steps

- [ ] Test remaining slices (neg120.38, neg130.38)
- [ ] Run validation script on all results
- [ ] Prepare summary for research team
- [ ] Ask team about ground-truth BCLC stage and necrosis status
- [ ] Proceed to full dataset validation
- [ ] Plan clinical pilot if all tests pass

---

## Attachments

- [ ] Browser console screenshot (if errors)
- [ ] Feature comparison CSV
- [ ] ROI overlay images from web app
- [ ] Model prediction screenshots
- [ ] SHAP evidence strip screenshots

