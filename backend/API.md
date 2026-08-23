# API Reference

Base URL (local dev): `http://localhost:8000`

All examples below are real output captured from a running instance of this backend.

---

## `GET /health`

Returns 200 once all five models are loaded in memory.

**Response**
```json
{
  "status": "ok",
  "models_loaded": ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "Necrosis-RandomForest"]
}
```

---

## `POST /predict/stage`

Predicts BCLC stage (4-class) using the chosen model.

**Request**
```json
{
  "model": "XGBoost",
  "features": {
    "Volume": 4200.1, "Area": 850.2, "MaxDiameter": 42.0, "SurfaceArea": 150.0,
    "Sphericity": 0.9, "Compactness": 0.25, "Elongation": 1.2,
    "Mean": 1200.0, "Median": 1180.0, "Min": 900.0, "Max": 1900.0, "Std": 210.0,
    "Skewness": 0.5, "Kurtosis": 3.0, "Entropy": 6.5,
    "GLCM_Contrast": 2.0, "GLCM_Correlation": 0.8, "GLCM_Homogeneity": 0.9,
    "GLCM_Energy": 0.1, "GLCM_Entropy": 5.0, "SRE": 0.5, "LRE": 10.0, "GLN": 0.3,
    "LiverEntropy": 6.0, "TumorLiverContrast": 0.6
  }
}
```

- `model`: one of `"XGBoost"`, `"LightGBM"`, `"RandomForest"`, `"GradientBoosting"`. Defaults to `"XGBoost"` if omitted.
- `features`: all 25 radiomic feature values (see [docs/FEATURE_MAPPING.md](../docs/FEATURE_MAPPING.md)). The backend selects only the subset the chosen model was trained on — sending all 25 regardless of model is correct and is what the frontend does.

**Response**
```json
{
  "stageProbs": {
    "Advanced": 0.3294806480407715,
    "Healthy": 0.008683989755809307,
    "A": 0.004106617532670498,
    "B": 0.6577287316322327
  },
  "predicted_stage": "B",
  "contributions": [
    { "name": "GLCM_Correlation", "value": -1.0125916004180908 },
    { "name": "GLCM_Homogeneity", "value": 0.8964572548866272 },
    { "name": "MaxDiameter", "value": -0.3705296516418457 }
  ]
}
```

- `contributions` is empty (`[]`) when `model` is `"GradientBoosting"` — see [docs/MODEL_LOADING.md](docs/MODEL_LOADING.md) for why.
- Positive contribution values push toward the predicted class; negative values push away from it (standard SHAP convention).

**Errors**
- `422` — a required feature key for the chosen model is missing: `{"detail": "Missing required feature keys: [...]"}`
- `500` — unexpected internal error; check the backend log (never returns a stack trace to the client)

---

## `POST /predict/necrotic`

Predicts necrotic vs. non-necrotic tissue.

**Request**
```json
{ "features": { /* all 25 features, same shape as above */ } }
```

Unlike `/predict/stage`, this endpoint always uses all 25 features (the necrosis model was trained on the full feature set, not the 12-feature SHAP-selected subset).

**Response**
```json
{
  "necroticProb": 0.29,
  "contributions": [
    { "name": "SRE", "value": 0.11460616403411328 },
    { "name": "LRE", "value": 0.10616670631936444 },
    { "name": "Std", "value": -0.07368005855665698 }
  ]
}
```

`necroticProb` is `P(Necrotic)` — see the class-mapping note in [backend/config.py](config.py) and [docs/MODEL_LOADING.md](docs/MODEL_LOADING.md) for how this was determined.

---

## Notes for frontend integration

- Both endpoints are called in parallel by the frontend (`Promise.all`) — see `runRealModel()` in `frontend/src/App.jsx`.
- CORS only allows the origins listed in the backend's `CORS_ORIGINS` env var. If you run the frontend on a different port, update it there.
- There is no authentication. This API is designed to be reachable only from `localhost` during local development.
