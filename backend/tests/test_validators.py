import pytest

from app.utils.errors import InvalidFeaturePayload
from app.utils.validators import require_features


def test_require_features_passes_when_all_present():
    require_features({"a": 1, "b": 2}, ["a", "b"])  # should not raise


def test_require_features_raises_on_missing():
    with pytest.raises(InvalidFeaturePayload) as exc_info:
        require_features({"a": 1}, ["a", "b", "c"])
    assert exc_info.value.missing == ["b", "c"]


def test_require_features_allows_nan_values():
    # A present-but-NaN feature is a modeling concern, not a payload-shape one.
    require_features({"a": float("nan")}, ["a"])
