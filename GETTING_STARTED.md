# Getting Started

Two things need to run at the same time, in two separate terminals: the **backend** (Python) and the **frontend** (Vite/React). Neither needs internet access, Docker, or an account — everything runs on `localhost`.

## Prerequisites

- Python 3.10+ (`python --version`)
- Node.js 18+ (`node --version`)

## 1. Backend — Python API

```bash
cd backend
python -m venv .venv
```

Activate it, then install and run:

```bash
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

You should see `All models loaded` and `Uvicorn running on http://127.0.0.1:8000`. Leave this terminal open.

> You will see several `InconsistentVersionWarning` lines on startup — these are expected (the delivered models were trained across slightly different scikit-learn versions) and do not affect predictions. See [backend/docs/MODEL_LOADING.md](backend/docs/MODEL_LOADING.md).

Verify it's alive:

```bash
curl http://127.0.0.1:8000/health
```

## 2. Frontend — React app

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`).

## 3. Try it end to end

1. Upload any two similar images as "post-contrast" and "pre-contrast" (for a first smoke test, any two photos work — the pipeline just needs pixels to subtract).
2. Draw a rough ROI on the subtracted image.
3. Click "Extract features from ROI".
4. Click "Test connection" — you should see "API reachable".
5. Click "Run prediction" — you should see stage probabilities, a necrosis readout, and an evidence strip of SHAP contributions.

If step 4 fails, the backend terminal is either not running or crashed on startup — check its output first.

## What you just ran

- The **frontend** did all image handling and radiomic feature extraction in your browser — nothing is uploaded anywhere.
- The **backend** received only the 25 numeric feature values, ran them through the actual trained model pipelines, and returned probabilities + SHAP explanations.

For how these two pieces are wired together, see [ARCHITECTURE.md](ARCHITECTURE.md).
