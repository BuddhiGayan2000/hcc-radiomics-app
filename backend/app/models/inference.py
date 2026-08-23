"""
Runs a single prediction through a loaded pipeline: build the exact feature
vector the model expects (right columns, right order), score it, and explain
it. Called by the route handlers in app/routes/predict.py.
"""
import numpy as np
import pandas as pd

import config
from app.models.explainer import contributions_for_class
from app.models.loader import LoadedModel


def _feature_frame(features: dict, feature_names: list[str]) -> pd.DataFrame:
    """Select exactly the columns this model was trained on, in that order.

    Passing a DataFrame (not a bare array) matters here: these pipelines were
    fit with named columns (feature_names_in_), and scikit-learn will warn —
    on some versions, raise — if given an unlabeled array instead.
    """
    return pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)


def predict_stage(model: LoadedModel, features: dict) -> tuple[dict, str, list[dict]]:
    X = _feature_frame(features, model.feature_names)
    probs = model.pipeline.predict_proba(X)[0]

    stage_probs = {
        label: float(probs[idx]) for idx, label in config.BCLC_CLASS_MAP.items()
    }
    predicted_idx = int(np.argmax(probs))
    predicted_stage = config.BCLC_CLASS_MAP[predicted_idx]

    contributions = []
    if model.explainer is not None:
        X_scaled = model.scaler.transform(X)
        shap_values = model.explainer.shap_values(X_scaled)
        contributions = contributions_for_class(shap_values, model.feature_names, predicted_idx)

    return stage_probs, predicted_stage, contributions


def predict_necrosis(model: LoadedModel, features: dict) -> tuple[float, list[dict]]:
    X = _feature_frame(features, model.feature_names)
    probs = model.pipeline.predict_proba(X)[0]
    necrotic_prob = float(probs[config.NECROSIS_POSITIVE_CLASS_INDEX])

    contributions = []
    if model.explainer is not None:
        X_scaled = model.scaler.transform(X)
        shap_values = model.explainer.shap_values(X_scaled)
        contributions = contributions_for_class(
            shap_values, model.feature_names, config.NECROSIS_POSITIVE_CLASS_INDEX
        )

    return necrotic_prob, contributions
