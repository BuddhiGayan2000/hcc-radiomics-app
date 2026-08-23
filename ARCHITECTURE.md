# Architecture

## The shape of the system

```
┌─────────────────────────────┐        HTTP (localhost only)        ┌──────────────────────────────┐
│   Frontend (browser)         │ ───────────────────────────────────▶│   Backend (Python process)    │
│   React + Vite, port 5173    │  POST /predict/stage                │   FastAPI, port 8000           │
│                               │  POST /predict/necrotic             │                                │
│  • image upload               │ ◀─────────────────────────────────── │  • loads 5 trained .joblib     │
│  • pixel subtraction          │  { stageProbs, contributions }      │    pipelines once at startup   │
│  • freehand ROI drawing       │  { necroticProb, contributions }    │  • scales + predicts            │
│  • 25-feature extraction      │                                     │  • SHAP explains the prediction │
│  • results UI                 │                                     │                                │
└─────────────────────────────┘                                      └──────────────────────────────┘
```

Both processes run on your machine. There is no cloud component, no shared database, and no third party in the loop — the only network traffic is the browser talking to `localhost:8000`.

## Why a backend at all, if everything else runs in the browser?

The trained models are Python objects (`.joblib` pickles of scikit-learn/XGBoost/LightGBM pipelines). A browser cannot execute them. Two options existed:

1. **Re-implement each model's tree-traversal logic in JavaScript.** Rejected — every model family (XGBoost, LightGBM, RandomForest, GradientBoosting) needs its own hand-written inference engine, and any retraining means re-validating numerical parity all over again.
2. **Run a small local Python server that loads the real models and answers over HTTP.** Chosen — the frontend keeps doing everything it already does well (image handling, ROI drawing, feature extraction), and the backend's only job is "take 25 numbers, return probabilities and SHAP values using the exact model that was validated in the study."

This is why the backend is a thin API, not a rewrite of anything: [backend/app/models/inference.py](backend/app/models/inference.py) is ~50 lines because the hard work (feature extraction) already happened in the browser before the request is sent.

## Request flow, step by step

1. User draws an ROI → `frontend/src/App.jsx` computes all 25 radiomic features from the pixels inside it (pure JavaScript, see `extractAllFeatures`).
2. User clicks "Run prediction" → the frontend fires two parallel requests:
   - `POST /predict/stage` with the chosen model name + all 25 features
   - `POST /predict/necrotic` with the same 25 features
3. Each backend route ([backend/app/routes/predict.py](backend/app/routes/predict.py)):
   - Validates that every feature key the requested model needs is present ([backend/app/utils/validators.py](backend/app/utils/validators.py))
   - Selects exactly that model's feature subset, in the order it was trained on ([backend/app/models/inference.py](backend/app/models/inference.py))
   - Calls `pipeline.predict_proba()` — the pipeline already contains the fitted `StandardScaler`, so no separate scaling step is needed
   - Builds a SHAP explanation for the predicted class ([backend/app/models/explainer.py](backend/app/models/explainer.py))
4. The frontend renders the stage gauge, necrosis readout, and evidence strip from the JSON response.

## Why the models don't need a separate scaler file

The original spec assumed bare classifiers plus separately-exported `StandardScaler` files. Inspecting the actual delivered `.joblib` files showed they are complete `imblearn.pipeline.Pipeline` objects with three steps: `scaler → smote → clf`. At prediction time, SMOTE (a training-only oversampling step) is a no-op, so calling `pipeline.predict_proba(X)` on raw feature values automatically applies the correct fitted scaler first. This simplified [backend/app/models/loader.py](backend/app/models/loader.py) considerably — see [backend/docs/MODEL_LOADING.md](backend/docs/MODEL_LOADING.md) for the full explanation.

## Known limitation: GradientBoosting has no SHAP contributions

`shap.TreeExplainer` does not support multiclass `GradientBoostingClassifier` (a SHAP library limitation, not something specific to this project). The backend detects this at startup and degrades gracefully: the GradientBoosting staging model still predicts normally, it simply returns an empty `contributions` array. See [backend/docs/MODEL_LOADING.md](backend/docs/MODEL_LOADING.md).

## Everything is local — what that simplifies

Per the current scope, there is no hosting, no Docker, and no production deployment:
- CORS is locked to `http://localhost:5173` (the Vite dev server), not a wildcard.
- The API binds to `127.0.0.1`, not `0.0.0.0` — it is not reachable from other machines.
- There's no build/CI pipeline to maintain — `pip install` + `npm install` is the entire setup.

If this ever needs to run somewhere other than a developer's own machine, that's a deliberate future decision, not something this codebase assumes.
