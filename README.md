# HCC Radiomics Clinical Decision-Support Application

A research prototype that predicts **BCLC liver cancer stage** (Healthy / A / B / Advanced) and **tumor necrosis** from CT radiomic features, with per-prediction SHAP explanations.

> ⚠️ **Research and educational tool only.** This is not a certified medical device and must never be presented, marketed, or used as one. All outputs must be confirmed by a qualified radiologist before any clinical use.

---

## What this project does

1. **Frontend (browser):** you upload a post-contrast and pre-contrast CT slice, the app computes the pixel-wise subtraction image, you draw a freehand ROI (region of interest) around the tumor, and the app extracts 25 radiomic features directly from the pixels inside that ROI.
2. **Backend (local Python server):** the 25 features are sent to a local API which runs the actual trained models (XGBoost, Random Forest, etc. — the ones validated in the original research study) and returns stage probabilities, a necrosis probability, and SHAP values explaining *why* the model predicted what it did.
3. **Everything runs on your machine.** No cloud hosting, no Docker — just two local processes (a Vite dev server and a Python API) talking to each other over `localhost`.

## Quick links

| I want to... | Go to |
|---|---|
| Get both servers running in under 10 minutes | [GETTING_STARTED.md](GETTING_STARTED.md) |
| Understand how the pieces fit together | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Look up an API endpoint | [backend/API.md](backend/API.md) |
| Look up what a specific radiomic feature means | [docs/FEATURE_MAPPING.md](docs/FEATURE_MAPPING.md) |
| Understand a medical/technical term | [docs/GLOSSARY.md](docs/GLOSSARY.md) |
| Validate that JS feature extraction matches the original Python pipeline | [docs/PARITY_TESTING.md](docs/PARITY_TESTING.md) |
| Contribute code | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Project status

This repository already contains, unmodified from the original research prototype:
- The React frontend (image upload, pixel subtraction, ROI drawing, feature extraction) — [frontend/src/App.jsx](frontend/src/App.jsx)
- Five trained model pipelines (`.joblib`) — [backend/models_storage/](backend/models_storage/)

What was built on top, in this repo, to make it a working application:
- The backend API that loads those models and serves real predictions — [backend/app/](backend/app/)
- Wiring, environment configuration, and documentation for running both pieces locally

See [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) for the day-to-day dev loop and [docs/PARITY_TESTING.md](docs/PARITY_TESTING.md) for the one validation step that must be done before treating any prediction as trustworthy.

## Folder structure

```
hcc-radiomics-app/
├── backend/     Python FastAPI server — loads the trained models, serves /predict/* endpoints
├── frontend/    React + Vite app — image handling, ROI drawing, feature extraction, results UI
├── docs/        Architecture, feature reference, glossary, diagrams — shared across both sides
├── data/        Sample test images + known feature values, used for parity testing
└── scripts/     One-off utility scripts (verify models load, run parity test, etc.)
```

Each folder above has its own `README.md` — start there when you land in a folder you haven't touched before.
