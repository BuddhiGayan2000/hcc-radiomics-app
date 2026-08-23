"""Request-payload validation. Reject early rather than substitute defaults."""
from app.utils.errors import InvalidFeaturePayload


def require_features(features: dict, required: list[str]) -> None:
    """Raise InvalidFeaturePayload (-> HTTP 422) if any required key is absent.

    Deliberately does not check for NaN/None values: a feature can legitimately
    be NaN (e.g. GLCM features on a degenerate ROI) and the model pipelines
    handle that the same way they did at training time. We only guard against
    the key being missing outright, which would otherwise silently become a
    NaN column and produce a misleading prediction.
    """
    missing = [k for k in required if k not in features]
    if missing:
        raise InvalidFeaturePayload(missing)
