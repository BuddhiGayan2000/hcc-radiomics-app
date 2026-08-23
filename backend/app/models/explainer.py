"""
Turns raw SHAP values into the flat {name, value} list the frontend's
EvidenceStrip component renders (see frontend/src/App.jsx, EvidenceStrip).

shap.TreeExplainer(clf).shap_values(X) on these models returns an array of
shape (n_samples, n_features, n_classes) for both the 4-class staging models
and the binary necrosis model (verified directly against the delivered
.joblib files with shap==0.52.0 — see backend/docs/MODEL_LOADING.md). This
module assumes that shape; if a future scikit-learn/shap upgrade changes it,
the assertion below will fail loudly instead of silently returning garbage.
"""
import numpy as np

import config


def contributions_for_class(shap_values: np.ndarray, feature_names: list[str], class_index: int) -> list[dict]:
    """Return the top-K {name, value} contributions for one predicted class."""
    arr = np.asarray(shap_values)
    assert arr.ndim == 3, f"expected (n_samples, n_features, n_classes), got shape {arr.shape}"
    row = arr[0, :, class_index]  # one sample, all features, one class

    order = np.argsort(-np.abs(row))[: config.TOP_K_CONTRIBUTIONS]
    return [{"name": feature_names[i], "value": float(row[i])} for i in order]
