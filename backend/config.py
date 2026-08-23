"""
Central configuration for the inference API. Every value a new developer
would need to look up — which file backs which model, what the 25 feature
names are, how class indices map to clinical labels — lives here so it
is never duplicated or guessed in route/inference code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models_storage"

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
IS_DEV = os.getenv("ENV", "development") != "production"

# ---------------------------------------------------------------------------
# The 25 radiomic features, in the same order the frontend's FEATURE_META
# object defines them (frontend/src/App.jsx). This is the canonical feature
# list used for request validation.
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "Volume", "Area", "MaxDiameter", "SurfaceArea", "Sphericity", "Compactness", "Elongation",
    "Mean", "Median", "Min", "Max", "Std", "Skewness", "Kurtosis", "Entropy",
    "GLCM_Contrast", "GLCM_Correlation", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Entropy",
    "SRE", "LRE", "GLN", "LiverEntropy", "TumorLiverContrast",
]

# ---------------------------------------------------------------------------
# BCLC staging models. Each is a full imblearn Pipeline (scaler -> smote -> clf)
# saved from the original study. Every model was trained on its own SHAP-selected
# feature subset — see model.feature_names_in_ at load time, do not assume they
# match. The scaler is already inside the pipeline: never re-fit or re-apply one.
# ---------------------------------------------------------------------------
STAGE_MODEL_FILES = {
    "XGBoost": "best_model_XGBoost.joblib",
    "LightGBM": "model_LightGBM.joblib",
    "RandomForest": "model_RandomForest.joblib",
    "GradientBoosting": "model_GradientBoosting.joblib",
}
DEFAULT_STAGE_MODEL = "XGBoost"  # best CV/test performance per run_summary.json

NECROSIS_MODEL_FILE = "best_necrotic_vs_others_model_RandomForest.joblib"

# ---------------------------------------------------------------------------
# Class-index -> label mapping.
#
# The saved pipelines only expose integer classes_ (e.g. [0,1,2,3]); the
# original LabelEncoder/mapping used at training time was not saved alongside
# them. This mapping was reconstructed from the confusion-matrix and SHAP
# summary plots in the research team's training notebook (see the reference
# screenshots kept with the original deliverables), where axis/plot order
# consistently matched ascending class index:
#   BCLC:     0=Advanced, 1=Healthy, 2=StageA, 3=StageB
#   Necrosis: 0=Others,   1=Necrotic
#
# CONFIDENCE: high, but not verified against a saved encoder. If predictions
# look inverted or clinically implausible during testing, this mapping is the
# first thing to check — see docs/PARITY_TESTING.md.
# ---------------------------------------------------------------------------
BCLC_CLASS_MAP = {0: "Advanced", 1: "Healthy", 2: "A", 3: "B"}
NECROSIS_POSITIVE_CLASS_INDEX = 1  # predict_proba[:, 1] = P(Necrotic)

# Number of top SHAP contributors returned per prediction (evidence strip).
TOP_K_CONTRIBUTIONS = 8
