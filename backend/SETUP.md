# Backend Setup

## 1. Create a virtual environment

```bash
cd backend
python -m venv .venv
```

Activate it:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows cmd: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, uvicorn, scikit-learn, imbalanced-learn, xgboost, lightgbm, shap, joblib, pandas, numpy — everything needed to load and run the five trained model pipelines.

## 3. (Optional) Configure environment

```bash
cp .env.example .env
```

The defaults work for local development unchanged — only edit `.env` if you need a different port or a different frontend origin for CORS.

## 4. Run

```bash
python run.py
```

Expected output ends with:

```
All models loaded: stage=['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting'], necrosis=ok
Uvicorn running on http://127.0.0.1:8000
```

You will also see a block of `InconsistentVersionWarning` messages — expected, see [docs/MODEL_LOADING.md](docs/MODEL_LOADING.md#version-warnings).

## 5. Verify

```bash
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","models_loaded":[...]}`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | Running from the wrong directory | Run `python run.py` from inside `backend/`, not the repo root |
| `Address already in use` | A previous instance is still running | Find and stop it, or change `API_PORT` in `.env` |
| Frontend shows "API unreachable" | Backend not running, or CORS origin mismatch | Check the backend terminal; confirm `CORS_ORIGINS` in `.env` includes the frontend's actual origin |
| `InvalidModelError` on startup | A model file is corrupted or from an incompatible library version | Re-run `scripts/verify_models.py` from the repo root to isolate which file |
