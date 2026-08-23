# Backend — Inference API

A small FastAPI server with one job: load the trained model pipelines once, and answer `/predict/*` requests with real predictions and SHAP explanations.

## Run it

See [SETUP.md](SETUP.md) for first-time setup. Once dependencies are installed:

```bash
python run.py
```

Serves on `http://127.0.0.1:8000` (configurable — see `.env.example`).

## Layout

```
backend/
├── run.py                 Entry point — `python run.py` starts the server
├── config.py               All constants: model file paths, feature lists, class mappings
├── requirements.txt
├── .env.example
│
├── app/
│   ├── main.py              FastAPI app: CORS, routes, exception handlers, startup model loading
│   ├── models/
│   │   ├── loader.py         Loads every .joblib pipeline + builds a SHAP explainer for each
│   │   ├── inference.py      Builds the feature vector, calls predict_proba, gets SHAP contributions
│   │   └── explainer.py      Turns raw SHAP arrays into the {name, value} list the UI renders
│   ├── routes/
│   │   ├── predict.py         POST /predict/stage, POST /predict/necrotic
│   │   └── health.py          GET /health
│   ├── schemas/               Pydantic request/response models
│   ├── utils/                 Validation, logging, error handlers
│   └── middleware/cors.py     CORS configuration
│
├── models_storage/          The five trained .joblib pipelines (see its own README)
├── tests/                   pytest suite
└── docs/                    Deep-dive docs (model loading internals, feature mapping)
```

## API reference

See [API.md](API.md) for the full request/response contract with examples.

## Security & privacy

- **Never log feature values or images.** Route handlers log only metadata (model name, predicted class, timing) — see `app/routes/predict.py` and `app/utils/logger.py`. If you add a new log line, ask "would this leak patient-derived data?" before committing it.
- **CORS is not a wildcard.** It's restricted to `CORS_ORIGINS` in `.env` (default: the Vite dev server origin only).
- **No server-side storage.** This API is stateless — it does not persist uploaded features, images, or predictions anywhere. If that ever needs to change (e.g. an audit trail), it requires an explicit, approved data-retention policy first (see the original spec, Section 7).
- **The API binds to `127.0.0.1`, not `0.0.0.0`.** It is not reachable from other machines on your network by default.

## Known limitations

- **GradientBoosting has no SHAP contributions.** `shap.TreeExplainer` doesn't support multiclass `GradientBoostingClassifier`. Predictions still work; `contributions` comes back as `[]`. See [docs/MODEL_LOADING.md](docs/MODEL_LOADING.md).
- **Class-label mapping was reconstructed, not read from a saved encoder.** See the comment block in `config.py` above `BCLC_CLASS_MAP` and [docs/MODEL_LOADING.md](docs/MODEL_LOADING.md) for the evidence and how to double-check it.
