# Contributing

This is a research prototype, not a public open-source project — "contributing" here means "how a teammate should work on this codebase."

## Ground rules

1. **Don't touch the feature-extraction math in `frontend/src/App.jsx` without re-running the parity test.** Every function between `shannonEntropy` and `extractAllFeatures` is a direct re-implementation of the original Python feature extraction. Changing rounding, bin counts, or formulas invalidates the trained models' assumptions. See [docs/PARITY_TESTING.md](docs/PARITY_TESTING.md).
2. **Never re-fit or replace the `StandardScaler` inside a model pipeline.** The scaler is part of the saved `.joblib` pipeline and must stay paired with the classifier it was fit alongside.
3. **Don't log feature values or images.** See the Security section of [backend/README.md](backend/README.md).
4. **Keep the "not a diagnostic device" banner as-is.** If it needs to change, that's a clinical/ethics decision, not an engineering one.

## Day-to-day workflow

See [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) for the run/edit/test loop.

## Code style

- Backend: standard PEP 8, type hints on function signatures, docstrings only where the *why* isn't obvious from the code.
- Frontend: match the existing prototype's style (functional components, inline style objects via the `COLORS` token map) rather than introducing a new styling approach.

## Adding a new model

If a new staging model is trained later:
1. Export it the same way as the existing ones — a full `imblearn.pipeline.Pipeline` with `scaler → smote → clf` steps, saved via `joblib.dump`.
2. Drop the `.joblib` file into `backend/models_storage/`.
3. Add an entry to `STAGE_MODEL_FILES` in [backend/config.py](backend/config.py).
4. Add the model name to the `Literal[...]` in [backend/app/schemas/requests.py](backend/app/schemas/requests.py) and to `STAGING_MODEL_OPTIONS` in `frontend/src/App.jsx`.
5. Confirm `shap.TreeExplainer` supports the new classifier type — see the GradientBoosting caveat in [backend/docs/MODEL_LOADING.md](backend/docs/MODEL_LOADING.md).
