# Development Workflow

## Daily loop

Two terminals, both left running while you work:

```bash
# Terminal 1
cd backend && python run.py       # auto-reloads on file changes (ENV=development)

# Terminal 2
cd frontend && npm run dev         # Vite hot-reloads on file changes
```

Edit backend Python files → uvicorn reloads automatically. Edit `frontend/src/App.jsx` → Vite hot-reloads the browser automatically. Neither requires a manual restart in normal development.

## Making a backend change

1. Edit the relevant file under `backend/app/`.
2. Check the terminal running `python run.py` for startup errors (a broken import fails loudly there).
3. Hit the endpoint with `curl` before touching the frontend — faster feedback loop:
   ```bash
   curl -X POST http://localhost:8000/predict/stage -H "Content-Type: application/json" -d '{"model":"XGBoost","features":{...}}'
   ```
4. Once the API responds as expected, verify in the browser end-to-end.

## Making a frontend change

Most changes are inside `frontend/src/App.jsx`. Vite's hot-reload means you'll usually see the change instantly without losing UI state. If you change feature-extraction logic specifically, re-read [PARITY_TESTING.md](PARITY_TESTING.md) first — that code has to stay numerically faithful to the original Python implementation.

## Debugging "API unreachable"

1. Is the backend terminal actually running, and did it print `Uvicorn running on http://127.0.0.1:8000`?
2. Does `curl http://localhost:8000/health` succeed from your own terminal? If not, the backend itself is the problem, not CORS.
3. If `curl` works but the browser shows "API unreachable" — open the browser console (F12) and look for a CORS error specifically. Fix: confirm `CORS_ORIGINS` in `backend/.env` includes the frontend's actual origin (`http://localhost:5173` by default).

## Debugging a wrong-looking prediction

1. First check: is this the `GradientBoosting` model? Its `contributions` are always empty — expected, not a bug (see [backend/docs/MODEL_LOADING.md](../backend/docs/MODEL_LOADING.md)).
2. Second check: does the *ranking* of stage probabilities look plausible at all (e.g. is "Healthy" ever the top prediction for a large, irregular ROI)? If predictions seem systematically inverted, revisit the class-index mapping in `backend/config.py` — see the confidence caveat in [backend/docs/MODEL_LOADING.md](../backend/docs/MODEL_LOADING.md).
3. Third check: has the parity test (see [PARITY_TESTING.md](PARITY_TESTING.md)) actually been run? An un-validated feature pipeline is the most likely source of "the model must be wrong" symptoms that are actually a feature-extraction mismatch.

## Running the backend test suite

```bash
cd backend
pytest
```

(See `backend/tests/` — add new tests alongside the route/model code they cover.)
