import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_features():
    """A structurally valid (not clinically meaningful) 25-feature payload."""
    return {
        "Volume": 4200.1, "Area": 850.2, "MaxDiameter": 42.0, "SurfaceArea": 150.0,
        "Sphericity": 0.9, "Compactness": 0.25, "Elongation": 1.2,
        "Mean": 1200.0, "Median": 1180.0, "Min": 900.0, "Max": 1900.0, "Std": 210.0,
        "Skewness": 0.5, "Kurtosis": 3.0, "Entropy": 6.5,
        "GLCM_Contrast": 2.0, "GLCM_Correlation": 0.8, "GLCM_Homogeneity": 0.9,
        "GLCM_Energy": 0.1, "GLCM_Entropy": 5.0, "SRE": 0.5, "LRE": 10.0, "GLN": 0.3,
        "LiverEntropy": 6.0, "TumorLiverContrast": 0.6,
    }
