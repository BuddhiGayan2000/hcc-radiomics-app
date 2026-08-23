# Model Loading — Internals & Findings

This doc records what was actually discovered by inspecting the five delivered `.joblib` files directly (not assumed from the original spec). If you're extending or debugging the backend, read this before touching `app/models/loader.py` or `config.py`.

## The models are full pipelines, not bare classifiers

The original technical spec assumed the delivered models were bare classifiers requiring a separately-exported `StandardScaler`. Loading the actual files shows otherwise:

```python
>>> import joblib
>>> pipe = joblib.load("best_model_XGBoost.joblib")
>>> type(pipe)
<class 'imblearn.pipeline.Pipeline'>
>>> pipe.steps
[('scaler', StandardScaler()), ('smote', SMOTE(k_neighbors=3, random_state=42)), ('clf', XGBClassifier(...))]
```

Every one of the five files (`best_model_XGBoost`, `model_LightGBM`, `model_RandomForest`, `model_GradientBoosting`, `best_necrotic_vs_others_model_RandomForest`) follows this same `scaler → smote → clf` shape.

**Consequence:** at prediction time, calling `pipeline.predict_proba(X)` is correct and complete. SMOTE only implements `fit_resample`, not `transform`, so imblearn's `Pipeline.predict()` skips it automatically — the scaler still runs. **Never manually re-scale or re-fit a scaler; the pipeline already does the right thing.**

## Each model has its own feature subset, in its own order

```
best_model_XGBoost.joblib          → 12 features: Std, TumorLiverContrast, LiverEntropy, Compactness,
                                      SurfaceArea, GLCM_Correlation, Mean, Skewness, GLCM_Homogeneity,
                                      MaxDiameter, Kurtosis, Entropy
model_LightGBM.joblib               → 12 features: same list but GLCM_Energy instead of Kurtosis
model_RandomForest.joblib           → 12 features: same as LightGBM's set
model_GradientBoosting.joblib       → 12 features: same as LightGBM's set
best_necrotic_vs_others_model_RandomForest.joblib → all 25 features
```

`app/models/loader.py` reads each pipeline's `feature_names_in_` at load time rather than hard-coding these lists — `app/models/inference.py` always selects exactly the right columns, in the right order, per model.

## Class-index → label mapping (read this before trusting a prediction)

The saved pipelines expose only integer `classes_` (`[0,1,2,3]` for staging, `[0,1]` for necrosis) — the original `LabelEncoder`/string mapping used at training time was **not saved** alongside them.

This mapping was reconstructed by cross-referencing multiple plots from the research team's training notebook (confusion matrices, SHAP summary titles, calibration curves — kept as reference screenshots with the original deliverables):

- The **XGBoost confusion matrix** ("Confusion matrix — XGBoost (test set)") shows axis order `Advanced, Healthy, StageA, StageB` — ascending order of the integer class index.
- The **SHAP per-class summary plots** are titled in the same sequence (`class: Advanced`, `class: Healthy`, `class: StageA`, `class: StageB`).
- The **necrosis confusion matrix** ("RandomForest discrimination of Necrotic vs. Others") shows axis order `Others, Necrotic` — i.e. index 0 = Others, index 1 = Necrotic.

Resulting mapping, encoded in `config.py`:

```python
BCLC_CLASS_MAP = {0: "Advanced", 1: "Healthy", 2: "A", 3: "B"}
NECROSIS_POSITIVE_CLASS_INDEX = 1  # predict_proba[:, 1] = P(Necrotic)
```

**Confidence: high, but not verified against a saved encoder.** If you get access to the original training notebook or a saved `LabelEncoder`, confirm this mapping directly and remove this caveat. If predictions look clinically implausible (e.g. an obviously advanced-looking case predicted as "Healthy" with high confidence), this mapping is the first thing to re-check.

## Version warnings

On startup you will see repeated warnings like:

```
InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.6.1 when using version 1.8.0.
```

This is expected — the five models were saved from slightly different scikit-learn versions during the original study (version numbers seen: 1.4.2, 1.6.1, 1.9.0), and this environment runs 1.8.0. All five models were loaded and produced sane `predict_proba` output when this was verified directly (see below) — this warning is noise, not a functional problem, unless you see actual prediction errors alongside it.

## GradientBoosting has no SHAP explainer

```
shap.utils._exceptions.InvalidModelError: GradientBoostingClassifier is only supported for binary classification right now!
```

`shap.TreeExplainer` (shap 0.52.0) does not support multiclass `sklearn.ensemble.GradientBoostingClassifier`. This is a SHAP library limitation — XGBoost, LightGBM, and RandomForest classifiers all work fine for the 4-class case.

`app/models/loader.py` catches this at startup and sets `explainer = None` for that one model. `app/models/inference.py` checks for `None` and returns `contributions: []` instead of crashing. The GradientBoosting model's actual predictions (`stageProbs`) are unaffected — only its evidence strip is empty.

## SHAP output shape (verified directly)

```python
>>> shap.TreeExplainer(clf).shap_values(X_scaled).shape
(1, 12, 4)   # (n_samples, n_features, n_classes) — for the 4-class staging models
(1, 25, 2)   # for the binary necrosis model
```

`app/models/explainer.py` asserts this exact shape (`ndim == 3`) so that a future shap/scikit-learn upgrade that changes this convention fails loudly at the assertion rather than silently returning wrong contributions.
